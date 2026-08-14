//! Prefix cache: reuse KV of a common prompt prefix across slots.
//!
//! P3 §4.0 spike confirmed `llama_memory_seq_cp` / `llama_memory_seq_keep` /
//! `llama_memory_seq_rm` are bound (llama.cpp SHA `5b474eb69`). We use the
//! **slot-local reuse** path: each slot keeps the KV of its own prefix so a
//! re-bound job that shares the same prefix does not re-prefill it.
//!
//! P5: Stop/Length/Cancel/Timeout leave that KV resident. `remember` stores
//! prompt + generated token IDs so the next bind's LCP can include the
//! previous assistant turn, or a mid-prefill prefix after a client drop.
//! A global cross-slot store is still NOT implemented — `PrefixStore` stays
//! inactive; no `llama_memory_seq_cp` between slots.

use std::collections::HashMap;

use crate::tokenizer::Token;

/// Longest common prefix length between two token slices.
pub fn common_prefix_len(a: &[Token], b: &[Token]) -> usize {
    let n = a.len().min(b.len());
    let mut i = 0;
    while i < n && a[i] == b[i] {
        i += 1;
    }
    i
}

/// Per-slot cache of the last bound prompt's prefix KV.
///
/// `tokens` is the prefix whose KV is already resident in the slot's seq.
/// When a new job shares this prefix, we skip prefilling those tokens and
/// start from `prefix_len`.
#[derive(Clone, Debug, Default)]
pub struct SlotPrefixCache {
    pub tokens: Vec<Token>,
    pub prefix_len: usize,
}

impl SlotPrefixCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Given the new job's prompt tokens, decide how many leading tokens can
    /// be reused from the cached prefix. Returns (reused_len, cache).
    ///
    /// The caller then starts prefill at `reused_len` (the KV for tokens
    /// `[0..reused_len)` is already resident in the slot's sequence).
    pub fn reuse(&mut self, prompt: &[Token]) -> usize {
        let len = common_prefix_len(&self.tokens, prompt);
        // Keep the matched prefix as the new cache (so it grows/shrinks with
        // the observed workload).
        if len > 0 {
            self.tokens = prompt[..len].to_vec();
        }
        self.prefix_len = len;
        len
    }

    pub fn reset(&mut self) {
        self.tokens.clear();
        self.prefix_len = 0;
    }

    /// After a successful Stop/Length, record the tokens whose KV is actually
    /// resident (`n_past` == `llama_memory_seq_pos_max + 1`). The last sampled
    /// token may not have been decoded yet, so `n_past` can be shorter than
    /// `prompt.len() + generated.len()`.
    pub fn remember(&mut self, prompt: &[Token], generated: &[Token], n_past: u32) {
        self.tokens.clear();
        self.tokens.extend_from_slice(prompt);
        self.tokens.extend_from_slice(generated);
        self.tokens.truncate(n_past as usize);
        self.prefix_len = self.tokens.len();
    }

    /// Bind-time reuse. Returns how many leading prompt tokens keep their KV.
    ///
    /// Always leaves at least one prompt token to prefill so the last cell has
    /// logits (llama-server `TAG_PROMPT_LOGITS`). Returns 0 when the GPU must
    /// start from position 0 (`clear_seq`).
    pub fn reuse_for_bind(&mut self, prompt: &[Token], gpu_n_past: u32) -> usize {
        if prompt.is_empty() {
            self.reset();
            return 0;
        }
        let mut n = common_prefix_len(&self.tokens, prompt);
        n = n.min(prompt.len() - 1);
        if n == 0 || gpu_n_past < n as u32 {
            self.reset();
            return 0;
        }
        self.prefix_len = n;
        n
    }
}

/// Optional global prefix store (disabled by default — see module doc).
#[derive(Clone, Debug, Default)]
pub struct PrefixStore {
    pub entries: HashMap<Vec<Token>, usize>,
}

impl PrefixStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(v: &[i32]) -> Vec<Token> {
        v.iter().map(|&x| x as Token).collect()
    }

    #[test]
    fn common_prefix_reuses_shared_head() {
        let a = t(&[1, 2, 3, 4]);
        let b = t(&[1, 2, 5, 6]);
        assert_eq!(common_prefix_len(&a, &b), 2);
    }

    #[test]
    fn common_prefix_empty_when_no_shared() {
        assert_eq!(common_prefix_len(&t(&[1]), &t(&[9])), 0);
    }

    #[test]
    fn reuse_updates_cache_to_matched_prefix() {
        let mut c = SlotPrefixCache::new();
        c.tokens = t(&[1, 2, 3, 4]);
        assert_eq!(c.reuse(&t(&[1, 2, 5])), 2);
        assert_eq!(c.tokens, t(&[1, 2]));
    }

    #[test]
    fn reuse_zero_keeps_prior_cache() {
        let mut c = SlotPrefixCache::new();
        c.tokens = t(&[1, 2]);
        assert_eq!(c.reuse(&t(&[9, 8])), 0);
        // reset clears; a no-match does not overwrite the cache
        assert_eq!(c.tokens, t(&[1, 2]));
    }

    #[test]
    fn remember_truncates_to_resident_n_past() {
        let mut c = SlotPrefixCache::new();
        c.remember(&t(&[1, 2, 3, 4]), &t(&[5, 6]), 5);
        assert_eq!(c.tokens, t(&[1, 2, 3, 4, 5]));
        assert_eq!(c.prefix_len, 5);
    }

    #[test]
    fn second_bind_reuses_common_prefix() {
        let mut c = SlotPrefixCache::new();
        assert_eq!(c.reuse_for_bind(&t(&[1, 2, 3, 4]), 0), 0);
        c.remember(&t(&[1, 2, 3, 4]), &t(&[5, 6]), 6);
        let n = c.reuse_for_bind(&t(&[1, 2, 3, 4, 5, 6, 7, 8]), 6);
        assert!(n > 0, "shared prefix must reuse, got {n}");
        assert_eq!(n, 6);
    }

    #[test]
    fn bind_without_common_prefix_clears() {
        let mut c = SlotPrefixCache::new();
        c.remember(&t(&[1, 2, 3]), &[], 3);
        assert_eq!(c.reuse_for_bind(&t(&[9, 8, 7]), 3), 0);
        assert!(c.tokens.is_empty());
    }

    #[test]
    fn generated_tokens_extend_next_lcp() {
        let mut c = SlotPrefixCache::new();
        c.remember(&t(&[1, 2, 3]), &t(&[4, 5]), 5);
        // Next prompt includes the previous assistant ids [4,5].
        let n = c.reuse_for_bind(&t(&[1, 2, 3, 4, 5, 6]), 5);
        assert_eq!(n, 5, "LCP must include generated assistant tokens");
    }

    #[test]
    fn reuse_for_bind_leaves_one_token_for_logits() {
        let mut c = SlotPrefixCache::new();
        c.remember(&t(&[1, 2, 3]), &[], 3);
        // Exact prompt match would be LCP==len; clamp so we still prefill.
        assert_eq!(c.reuse_for_bind(&t(&[1, 2, 3]), 3), 2);
    }

    #[test]
    fn reuse_for_bind_rejects_when_gpu_shorter_than_lcp() {
        let mut c = SlotPrefixCache::new();
        c.remember(&t(&[1, 2, 3, 4]), &[], 4);
        assert_eq!(c.reuse_for_bind(&t(&[1, 2, 3, 4, 5]), 2), 0);
        assert!(c.tokens.is_empty());
    }

    #[test]
    fn cancel_mid_prefill_reuses_decoded_prefix() {
        let mut c = SlotPrefixCache::new();
        let prompt = t(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        // Same request cancelled after 6 prompt tokens landed in KV.
        c.remember(&prompt, &[], 6);
        assert_eq!(c.reuse_for_bind(&prompt, 6), 6);
    }

    #[test]
    fn cancel_after_full_prefill_leaves_logits_token() {
        let mut c = SlotPrefixCache::new();
        let prompt = t(&[1, 2, 3, 4, 5]);
        c.remember(&prompt, &t(&[6, 7]), 7);
        // Retry of the same prompt: keep all but one cell for logits.
        assert_eq!(c.reuse_for_bind(&prompt, 7), 4);
    }
}
