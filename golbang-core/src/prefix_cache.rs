//! Prefix cache: reuse KV of a common prompt prefix across slots.
//!
//! P3 §4.0 spike confirmed `llama_memory_seq_cp` / `llama_memory_seq_keep` /
//! `llama_memory_seq_rm` are bound (llama.cpp SHA `5b474eb69`). We use the
//! **slot-local reuse** path: each slot keeps the KV of its own prefix so a
//! re-bound job that shares the same prefix does not re-prefill it.
//!
//! A global cross-slot store is intentionally NOT implemented (see §4.1 note):
//! copying a whole seq with `llama_memory_seq_cp` requires a reserved sequence
//! slot and shifts position bookkeeping; the spike showed the simpler
//! slot-local path already removes the dominant TTFT cost for repeated
//! system prompts.

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
}
