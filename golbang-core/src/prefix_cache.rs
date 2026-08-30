//! Prefix cache: reuse KV of a common prompt prefix across slots.
//!
//! P3 §4.0 spike confirmed `llama_memory_seq_cp` / `llama_memory_seq_keep` /
//! `llama_memory_seq_rm` are bound (llama.cpp SHA `3ac5658c7`). We use the
//! **slot-local reuse** path: each slot keeps the KV of its own prefix so a
//! re-bound job that shares the same prefix does not re-prefill it.
//!
//! P5: Stop/Length/Cancel/Timeout leave that KV resident. `remember` stores
//! prompt + generated token IDs so the next bind's LCP can include the
//! previous assistant turn, or a mid-prefill prefix after a client drop.
//! P8-B: `PrefixStore` holds host `seq_state` dumps of the shared tool/system
//! head. Empty-slot bind restores the longest matching dump (`seq_state_set`).
//! There is still no `llama_memory_seq_cp` between slots.
//!
//! Vision (Qwen3.8 `--mmproj`): `VisionSeq` is llama-server `server_tokens`
//! for one slot. Image cells compare FNV chunk ids, not vocab tokens. M-RoPE
//! uses `n_pos != n_tokens`; `pos_next` is what `seq_rm` / `n_past` need.

use std::collections::HashMap;

use crate::slot::SeqCheckpoint;
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

/// Prefix tokens `[0..n_tokens)` used as a [`PrefixStore`] key.
///
/// `None` when `n_tokens` is 0 or longer than `prompt` — a store entry must
/// be an exact head of the prompt that produced the dump.
pub fn snapshot_key(prompt: &[Token], n_tokens: u32) -> Option<Vec<Token>> {
    let n = n_tokens as usize;
    if n == 0 || n > prompt.len() {
        None
    } else {
        Some(prompt[..n].to_vec())
    }
}

/// Search length for a bind-time store lookup.
///
/// `reuse_len == 0` is an empty slot (no local LCP). Search up to
/// `prompt.len() - 1` so a 12k tool-head dump can still hit, while leaving
/// one logits token.
pub fn host_search_len(prompt_len: usize, reuse_len: usize) -> usize {
    if reuse_len == 0 {
        prompt_len.saturating_sub(1)
    } else {
        reuse_len
    }
}

/// One mtmd chunk after `mtmd_tokenize`. Image/audio ids are FNV hashes
/// from `mtmd_helper_bitmap_init_from_buf`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VisionChunk {
    Text(Vec<Token>),
    Media {
        id: String,
        n_tokens: u32,
        n_pos: u32,
    },
}

/// One image/audio span in a flattened vision prompt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisionImageSpan {
    pub start: usize,
    pub id: String,
    pub n_tokens: u32,
    pub n_pos: u32,
}

/// Flattened text tokens + media spans. Image cells use token id 0; LCP
/// compares span ids, not those placeholders. Qwen3.8 M-RoPE: `n_pos`
/// for an image is `max(nx, ny)`, smaller than `n_tokens`.
#[derive(Clone, Debug, Default)]
pub struct VisionSeq {
    pub tokens: Vec<Token>,
    pub images: Vec<VisionImageSpan>,
}

impl VisionSeq {
    pub fn from_chunks(chunks: impl IntoIterator<Item = VisionChunk>) -> Self {
        let mut seq = Self::default();
        for chunk in chunks {
            match chunk {
                VisionChunk::Text(toks) => seq.tokens.extend(toks),
                VisionChunk::Media {
                    id,
                    n_tokens,
                    n_pos,
                } => {
                    let start = seq.tokens.len();
                    seq.tokens
                        .extend(std::iter::repeat(0).take(n_tokens as usize));
                    seq.images.push(VisionImageSpan {
                        start,
                        id,
                        n_tokens,
                        n_pos,
                    });
                }
            }
        }
        seq
    }

    pub fn n_tokens(&self) -> usize {
        self.tokens.len()
    }

    pub fn n_pos(&self) -> u32 {
        self.pos_next(self.tokens.len())
    }

    pub fn image_at(&self, idx: usize) -> Option<&VisionImageSpan> {
        self.images.iter().find(|im| im.start == idx)
    }

    pub fn image_covering(&self, idx: usize) -> Option<&VisionImageSpan> {
        self.images
            .iter()
            .find(|im| idx >= im.start && idx < im.start.saturating_add(im.n_tokens as usize))
    }

    /// Position after the first `n_tokens` cells (M-RoPE-aware).
    pub fn pos_next(&self, n_tokens: usize) -> u32 {
        let n = n_tokens.min(self.tokens.len());
        let mut idx = 0;
        let mut pos = 0u32;
        while idx < n {
            if let Some(img) = self.image_at(idx) {
                pos = pos.saturating_add(img.n_pos);
                idx = idx.saturating_add(img.n_tokens as usize);
            } else {
                pos = pos.saturating_add(1);
                idx += 1;
            }
        }
        pos
    }

    /// Token index whose position is `max_pos` (llama-server `size_up_to_pos`).
    pub fn size_up_to_pos(&self, max_pos: u32) -> usize {
        let mut idx = 0;
        let mut pos = 0u32;
        while idx < self.tokens.len() && pos < max_pos {
            if let Some(img) = self.image_at(idx) {
                pos = pos.saturating_add(img.n_pos);
                idx = idx.saturating_add(img.n_tokens as usize);
            } else {
                pos = pos.saturating_add(1);
                idx += 1;
            }
        }
        idx
    }

    /// llama-server `get_common_prefix` with media: whole image or none.
    pub fn common_prefix(&self, other: &Self) -> usize {
        let max = self.tokens.len().min(other.tokens.len());
        let mut i = 0;
        while i < max {
            match (self.image_at(i), other.image_at(i)) {
                (Some(a), Some(b)) => {
                    if !a.id.is_empty() && a.id == b.id && a.n_tokens == b.n_tokens {
                        i = i.saturating_add(a.n_tokens as usize);
                        continue;
                    }
                    return i;
                }
                (None, None) => {
                    if self.tokens[i] != other.tokens[i] {
                        return i;
                    }
                    i += 1;
                }
                _ => return i,
            }
        }
        max
    }

    /// Bind-time reuse in **token cells**. Leaves one cell for logits and
    /// never splits an image. 0 → caller `clear_seq`.
    pub fn reuse_for_bind(&self, prompt: &Self, gpu_n_past: u32) -> usize {
        if prompt.tokens.is_empty() {
            return 0;
        }
        let mut n = self.common_prefix(prompt);
        n = n.min(prompt.tokens.len() - 1);
        if let Some(img) = prompt.image_covering(n) {
            n = img.start;
        }
        if n == 0 || gpu_n_past < prompt.pos_next(n) {
            return 0;
        }
        n
    }

    pub fn append_generated(&mut self, generated: &[Token]) {
        self.tokens.extend_from_slice(generated);
    }

    /// Drop cells whose position is at or past `n_pos` (cancel mid-decode).
    /// A partial image at the cut is dropped whole.
    pub fn truncate_to_pos(&mut self, n_pos: u32) {
        let mut idx = 0;
        let mut pos = 0u32;
        while idx < self.tokens.len() && pos < n_pos {
            if let Some(img) = self.image_at(idx) {
                if pos.saturating_add(img.n_pos) > n_pos {
                    break;
                }
                pos = pos.saturating_add(img.n_pos);
                idx = idx.saturating_add(img.n_tokens as usize);
            } else {
                pos = pos.saturating_add(1);
                idx += 1;
            }
        }
        self.tokens.truncate(idx);
        self.images.retain(|im| im.start < idx);
    }
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
    /// Last vision prompt (+ generated text cells). `None` after a text bind
    /// or a full reset. Next `--mmproj` bind LCPs against this, not `tokens`.
    pub vision: Option<VisionSeq>,
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
        self.vision = None;
    }

    pub fn remember_vision(&mut self, mut seq: VisionSeq, generated: &[Token], n_pos: u32) {
        seq.append_generated(generated);
        seq.truncate_to_pos(n_pos);
        self.prefix_len = seq.n_tokens();
        self.vision = Some(seq);
        self.tokens.clear();
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
    /// logits (llama-server `TAG_PROMPT_LOGITS`). `gpu_n_past` is the
    /// restorable cover: live GPU occupancy, or after a watermark
    /// `gpu_n.max(ckpt_n)`. LCP above that cover is clamped so session
    /// identity stays and a shorter host dump can still restore. Returns 0
    /// only when nothing is restorable (`clear_seq`).
    pub fn reuse_for_bind(&mut self, prompt: &[Token], gpu_n_past: u32) -> usize {
        if prompt.is_empty() {
            self.reset();
            return 0;
        }
        let mut n = common_prefix_len(&self.tokens, prompt);
        n = n.min(prompt.len() - 1);
        n = n.min(gpu_n_past as usize);
        if n == 0 {
            self.reset();
            return 0;
        }
        self.prefix_len = n;
        n
    }
}

/// Cross-slot host snapshot map (P8-B).
///
/// Each entry is a `SeqCheckpoint` captured at a known token length
/// (`n_tokens`), keyed by the prompt prefix tokens `[0..n_tokens)`. A bind
/// whose prompt matches an entry restores that dump instead of re-prefilling
/// the shared tool/system head. Lookup is a linear scan — no trie.
///
/// Caps: a single-digit entry limit plus a RAM ceiling. Eviction drops the
/// longest dump first so a 30k+ snapshot cannot push out the short tool head.
#[derive(Clone, Debug)]
pub struct PrefixStore {
    pub entries: HashMap<Vec<Token>, SeqCheckpoint>,
    /// Total bytes of all entry `data` (tracked for the RAM cap).
    pub total_bytes: usize,
    /// Soft RAM cap (bytes). 0 = unlimited.
    pub max_bytes: usize,
    /// Soft entry-count cap. 0 = unlimited. Default is a single digit.
    pub max_entries: usize,
    /// Monotonic "now" tick (advances on every insert/use).
    last_used_tick: u64,
    /// Per-entry tick of last use (keyed by the same prefix slice).
    last_used: HashMap<Vec<Token>, u64>,
}

impl Default for PrefixStore {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            total_bytes: 0,
            max_bytes: 0,
            max_entries: Self::DEFAULT_MAX_ENTRIES,
            last_used_tick: 0,
            last_used: HashMap::new(),
        }
    }
}

impl PrefixStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Recommended RAM cap for the shared tool/system head (~0.8 GiB for
    /// Qwen3.8 FA at 12545 tokens). 2 GiB leaves room for a few system prompts.
    pub const DEFAULT_MAX_BYTES: usize = 2 * 1024 * 1024 * 1024; // 2 GiB

    /// One tool head plus a handful of system prompts — never a ubatch trail.
    pub const DEFAULT_MAX_ENTRIES: usize = 8;

    /// Store-stub window (mirrors `scheduler::STORE_STUB_*`): dumps inside the
    /// shared tool/system head are the shortest tool anchors and are kept last
    /// under the global host-RAM cap (HiCache L2, 4th stage).
    pub const STORE_STUB_MIN: u32 = 8_192;
    pub const STORE_STUB_MAX: u32 = 16_384;

    fn is_store_stub_len(n_tokens: u32) -> bool {
        (Self::STORE_STUB_MIN..=Self::STORE_STUB_MAX).contains(&n_tokens)
    }

    pub fn with_cap(max_bytes: usize) -> Self {
        let mut s = Self::new();
        s.max_bytes = max_bytes;
        s
    }

    pub fn with_limits(max_bytes: usize, max_entries: usize) -> Self {
        let mut s = Self::new();
        s.max_bytes = max_bytes;
        s.max_entries = max_entries;
        s
    }

    /// Insert or refresh a snapshot for prefix `[0..n_tokens)`.
    ///
    /// Skips empty keys and key/`n_tokens` mismatches: those cannot hit
    /// `find_best` (`LCP == n_tokens`) and would only thrash the cap.
    pub fn put(&mut self, prefix: Vec<Token>, ckpt: SeqCheckpoint) {
        if prefix.is_empty() || ckpt.n_tokens == 0 {
            return;
        }
        if prefix.len() != ckpt.n_tokens as usize {
            return;
        }
        self.last_used_tick = self.last_used_tick.saturating_add(1);
        if let Some(old) = self.entries.insert(prefix.clone(), ckpt) {
            self.total_bytes = self.total_bytes.saturating_sub(old.data.len());
        }
        self.total_bytes = self.total_bytes.saturating_add(self.entry_bytes(&prefix));
        self.last_used.insert(prefix, self.last_used_tick);
        self.evict_if_over_cap();
    }

    /// Find the longest usable snapshot for `prompt`.
    ///
    /// An entry is usable when its LCP with `prompt` equals `n_tokens` (the
    /// whole snapshot prefix still matches) and `n_tokens <= reuse_len + 1`
    /// (the GPU can restore it and trim one logits token). Among usable
    /// entries, returns the longest. Linear scan.
    pub fn find_best(&mut self, prompt: &[Token], reuse_len: usize) -> Option<SeqCheckpoint> {
        let mut best: Option<(u32, Vec<Token>)> = None;
        for (prefix, ckpt) in &self.entries {
            if ckpt.n_tokens == 0 || ckpt.n_tokens > reuse_len as u32 + 1 {
                continue;
            }
            if common_prefix_len(prefix, prompt) != ckpt.n_tokens as usize {
                continue;
            }
            match best {
                Some((n, _)) if n >= ckpt.n_tokens => {}
                _ => best = Some((ckpt.n_tokens, prefix.clone())),
            }
        }
        best.map(|(_, prefix)| {
            self.touch(&prefix);
            self.entries[&prefix].clone()
        })
    }

    /// Bind-time lookup. `reuse_len == 0` still searches the new prompt
    /// (`prompt.len() - 1`) so an empty slot can restore the shared head.
    pub fn find_best_for_bind(
        &mut self,
        prompt: &[Token],
        reuse_len: usize,
    ) -> Option<SeqCheckpoint> {
        self.find_best(prompt, host_search_len(prompt.len(), reuse_len))
    }

    /// Mark an entry recently used (advances the tick without changing data).
    fn touch(&mut self, prefix: &[Token]) {
        self.last_used_tick = self.last_used_tick.saturating_add(1);
        if let Some(t) = self.last_used.get_mut(prefix) {
            *t = self.last_used_tick;
        }
    }

    fn entry_bytes(&self, prefix: &[Token]) -> usize {
        self.entries.get(prefix).map(|c| c.data.len()).unwrap_or(0)
    }

    fn over_cap(&self) -> bool {
        (self.max_bytes > 0 && self.total_bytes > self.max_bytes)
            || (self.max_entries > 0 && self.entries.len() > self.max_entries)
    }

    /// Drop the longest dump until the store's own caps hold. Includes 8k–16k
    /// stubs: production `put` only inserts those, so skipping them would
    /// make `max_entries` / `max_bytes` a no-op.
    fn evict_if_over_cap(&mut self) {
        while self.over_cap() && !self.entries.is_empty() {
            let victim = self.pick_longest_victim(false);
            let Some(victim) = victim else {
                return;
            };
            if let Some(ckpt) = self.entries.remove(&victim) {
                self.total_bytes = self.total_bytes.saturating_sub(ckpt.data.len());
            }
            self.last_used.remove(&victim);
        }
    }

    /// Longest dump, then oldest tick at the same length.
    /// `skip_stubs` keeps 8k–16k tool heads (global host-RAM cap only).
    fn pick_longest_victim(&self, skip_stubs: bool) -> Option<Vec<Token>> {
        self.entries
            .iter()
            .filter(|(_, ckpt)| !skip_stubs || !Self::is_store_stub_len(ckpt.n_tokens))
            .max_by(|a, b| {
                a.1.n_tokens.cmp(&b.1.n_tokens).then_with(|| {
                    let ta = self.last_used.get(a.0).copied().unwrap_or(0);
                    let tb = self.last_used.get(b.0).copied().unwrap_or(0);
                    // Same length: older tick is the victim.
                    tb.cmp(&ta)
                })
            })
            .map(|(k, _)| k.clone())
    }

    /// Drop the longest non-stub dump until `total_bytes + chain_bytes <= cap`.
    ///
    /// Stubs stay: the global cap may remain over once only tool heads are
    /// left. The store's own 8-entry / 2 GiB caps still drop stubs via
    /// `evict_if_over_cap`. The caller decides whether slot chains still
    /// need trimming.
    pub fn evict_global_over_cap(&mut self, chain_bytes: usize, cap: usize) {
        while cap > 0 && self.total_bytes + chain_bytes > cap {
            let victim = self.pick_longest_victim(true);
            let Some(victim) = victim else {
                break;
            };
            if let Some(ckpt) = self.entries.remove(&victim) {
                self.total_bytes = self.total_bytes.saturating_sub(ckpt.data.len());
            }
            self.last_used.remove(&victim);
        }
    }

    /// Number of resident entries (for tests / metrics).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
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
    fn reuse_for_bind_clamps_lcp_to_hint() {
        let mut c = SlotPrefixCache::new();
        c.remember(&t(&[1, 2, 3, 4]), &[], 4);
        // Cover (GPU occupancy or ckpt_n) is 2, LCP is 4: reuse the
        // restorable prefix and keep session identity.
        assert_eq!(c.reuse_for_bind(&t(&[1, 2, 3, 4, 5]), 2), 2);
        assert_eq!(c.tokens, t(&[1, 2, 3, 4]));
        assert_eq!(c.prefix_len, 2);
    }

    #[test]
    fn reuse_for_bind_clamps_when_lcp_exceeds_ckpt_hint() {
        let mut c = SlotPrefixCache::new();
        let prompt: Vec<Token> = (0..70).map(|i| i as Token).collect();
        c.remember(&prompt[..60], &prompt[60..], 70);
        let mut next = prompt.clone();
        next.push(99);
        // Watermark: GPU empty, hint_n = ckpt_n = 60, next-turn LCP = 70.
        assert_eq!(c.reuse_for_bind(&next, 60), 60);
        assert_eq!(c.tokens.len(), 70, "session identity kept");
        assert_eq!(c.prefix_len, 60);
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

    fn media(id: &str, n_tokens: u32, n_pos: u32) -> VisionChunk {
        VisionChunk::Media {
            id: id.into(),
            n_tokens,
            n_pos,
        }
    }

    #[test]
    fn vision_lcp_matches_same_image_hash() {
        let a = VisionSeq::from_chunks([
            VisionChunk::Text(t(&[1, 2, 3])),
            media("img-a", 4, 2),
            VisionChunk::Text(t(&[9])),
        ]);
        let b = VisionSeq::from_chunks([
            VisionChunk::Text(t(&[1, 2, 3])),
            media("img-a", 4, 2),
            VisionChunk::Text(t(&[9, 8])),
        ]);
        assert_eq!(a.common_prefix(&b), 8);
        assert_eq!(a.pos_next(3), 3);
        assert_eq!(a.pos_next(7), 5);
        assert_eq!(a.n_pos(), 6);
    }

    #[test]
    fn vision_lcp_stops_at_different_image() {
        let a = VisionSeq::from_chunks([VisionChunk::Text(t(&[1, 2])), media("img-a", 3, 2)]);
        let b = VisionSeq::from_chunks([VisionChunk::Text(t(&[1, 2])), media("img-b", 3, 2)]);
        assert_eq!(a.common_prefix(&b), 2);
    }

    #[test]
    fn vision_lcp_does_not_split_image() {
        let a = VisionSeq::from_chunks([
            VisionChunk::Text(t(&[1])),
            media("img-a", 4, 2),
            VisionChunk::Text(t(&[7])),
        ]);
        let b = VisionSeq::from_chunks([VisionChunk::Text(t(&[1])), media("img-a", 4, 2)]);
        assert_eq!(a.common_prefix(&b), 5);
    }

    #[test]
    fn vision_reuse_leaves_logits_and_skips_mid_image() {
        let prev = VisionSeq::from_chunks([
            VisionChunk::Text(t(&[1, 2, 3])),
            media("img-a", 4, 2),
            VisionChunk::Text(t(&[9, 8])),
        ]);
        // Exact same prompt: clamp off last text token (not mid-image).
        let n = prev.reuse_for_bind(&prev, prev.n_pos());
        assert_eq!(n, 8);
        assert_eq!(prev.pos_next(n), prev.n_pos() - 1);

        // Prompt that ends on an image: snap reuse to image start.
        let img_only =
            VisionSeq::from_chunks([VisionChunk::Text(t(&[1, 2, 3])), media("img-a", 4, 2)]);
        let n = prev.reuse_for_bind(&img_only, prev.n_pos());
        assert_eq!(n, 3, "must not reuse a partial image for logits");
        assert_eq!(img_only.pos_next(n), 3);
    }

    #[test]
    fn vision_reuse_rejects_when_gpu_shorter_than_pos() {
        let prev = VisionSeq::from_chunks([VisionChunk::Text(t(&[1, 2, 3])), media("img-a", 4, 2)]);
        let n = prev.reuse_for_bind(&prev, 2);
        assert_eq!(n, 0);
    }

    #[test]
    fn vision_append_and_truncate_keeps_full_images() {
        let mut seq = VisionSeq::from_chunks([
            VisionChunk::Text(t(&[1, 2])),
            media("img-a", 4, 2),
            VisionChunk::Text(t(&[9])),
        ]);
        // n_pos = 2 + 2 + 1 = 5
        seq.append_generated(&t(&[10, 11, 12]));
        assert_eq!(seq.n_pos(), 8);
        seq.truncate_to_pos(6);
        assert_eq!(seq.n_pos(), 6);
        assert_eq!(seq.tokens, t(&[1, 2, 0, 0, 0, 0, 9, 10]));

        seq.truncate_to_pos(3);
        // pos 3 is inside the image (pos 2..4); drop the image.
        assert_eq!(seq.tokens, t(&[1, 2]));
        assert!(seq.images.is_empty());
    }

    #[test]
    fn remember_vision_clears_text_tokens() {
        let mut c = SlotPrefixCache::new();
        c.remember(&t(&[1, 2, 3]), &[], 3);
        let seq = VisionSeq::from_chunks([VisionChunk::Text(t(&[1, 2, 3])), media("img-a", 2, 1)]);
        c.remember_vision(seq, &t(&[9]), 5);
        assert!(c.tokens.is_empty());
        let vs = c.vision.as_ref().unwrap();
        assert_eq!(vs.n_tokens(), 6);
        assert_eq!(vs.n_pos(), 5);
    }

    #[test]
    fn reset_clears_vision() {
        let mut c = SlotPrefixCache::new();
        c.vision = Some(VisionSeq::from_chunks([VisionChunk::Text(t(&[1]))]));
        c.reset();
        assert!(c.vision.is_none());
    }

    fn ckpt(n_tokens: u32, bytes: u8) -> crate::slot::SeqCheckpoint {
        crate::slot::SeqCheckpoint {
            n_tokens,
            data: vec![bytes; n_tokens as usize],
        }
    }

    #[test]
    fn prefix_store_find_best_hit() {
        let mut store = PrefixStore::new();
        store.put(t(&[1, 2, 3, 4]), ckpt(4, 0xAA));
        // Prompt shares the whole 4-token snapshot; reuse_len=4 fits n_tokens<=5.
        let hit = store.find_best(&t(&[1, 2, 3, 4, 5, 6]), 4);
        assert!(hit.is_some());
        let hit = hit.unwrap();
        assert_eq!(hit.n_tokens, 4);
        assert_eq!(hit.data, vec![0xAA; 4]);
    }

    #[test]
    fn prefix_store_find_best_returns_longest() {
        let mut store = PrefixStore::new();
        store.put(t(&[1, 2]), ckpt(2, 0x11));
        store.put(t(&[1, 2, 3, 4]), ckpt(4, 0x22));
        let hit = store.find_best(&t(&[1, 2, 3, 4, 5, 6]), 5);
        assert!(hit.is_some());
        assert_eq!(
            hit.unwrap().n_tokens,
            4,
            "must return the longest usable snapshot"
        );
    }

    #[test]
    fn prefix_store_find_best_miss() {
        let mut store = PrefixStore::new();
        store.put(t(&[9, 8, 7]), ckpt(3, 0x33));
        // No shared prefix -> None.
        assert!(store.find_best(&t(&[1, 2, 3]), 3).is_none());
    }

    #[test]
    fn prefix_store_find_best_rejects_too_long() {
        let mut store = PrefixStore::new();
        store.put(t(&[1, 2, 3, 4, 5, 6, 7, 8]), ckpt(8, 0x44));
        // n_tokens=8 > reuse_len+1=4 -> unusable even on exact prefix match.
        assert!(store.find_best(&t(&[1, 2, 3, 4, 5, 6, 7, 8]), 3).is_none());
    }

    #[test]
    fn prefix_store_key_collision_overwrites() {
        let mut store = PrefixStore::new();
        store.put(t(&[1, 2, 3]), ckpt(3, 0x55));
        store.put(t(&[1, 2, 3]), ckpt(3, 0x66));
        assert_eq!(store.len(), 1);
        let hit = store.find_best(&t(&[1, 2, 3, 4]), 3).unwrap();
        assert_eq!(hit.data, vec![0x66; 3], "same key must be replaced");
        assert_eq!(store.total_bytes, 3);
    }

    #[test]
    fn prefix_store_evicts_longest_not_short_head() {
        let mut store = PrefixStore::with_cap(10); // tiny cap
        store.put(t(&[1, 2]), ckpt(2, 0x11)); // 2 bytes
        store.put(t(&[1, 2, 3, 4]), ckpt(4, 0x22)); // 4 bytes -> total 6
        store.put(t(&[1, 2, 3, 4, 5, 6]), ckpt(6, 0x33)); // 6 bytes -> total 12 > 10
        assert!(store.total_bytes <= 10, "must evict down to cap");
        // Long dump is the victim; the short prefix must still hit.
        assert!(
            store.find_best(&t(&[1, 2, 9]), 2).is_some(),
            "short prefix must not be pushed out by a longer dump"
        );
        assert!(
            store.entries.values().all(|c| c.n_tokens < 6),
            "the 6-token dump must be the victim"
        );
    }

    #[test]
    fn prefix_store_entry_cap_is_single_digit() {
        let mut store = PrefixStore::with_limits(0, 3);
        for n in 1..=5u32 {
            let prefix: Vec<Token> = (0..n).map(|i| i as Token).collect();
            store.put(prefix, ckpt(n, n as u8));
        }
        assert!(store.len() <= 3);
        // Longest dumps dropped; shortest remain.
        assert!(store.entries.values().all(|c| c.n_tokens <= 3));
        assert!(store.find_best(&t(&[0, 1, 9]), 2).is_some());
    }

    #[test]
    fn snapshot_key_len_eq_n_tokens() {
        let prompt: Vec<Token> = (0..20_000).map(|i| i as Token).collect();
        let key = snapshot_key(&prompt, 12_288).unwrap();
        assert_eq!(key.len(), 12_288);
        assert_eq!(key, prompt[..12_288]);
        assert!(snapshot_key(&prompt, 0).is_none());
        assert!(snapshot_key(&prompt, 20_001).is_none());
        assert_eq!(snapshot_key(&prompt, 20_000).unwrap().len(), 20_000);
    }

    #[test]
    fn empty_slot_store_hit_reuses_head() {
        let mut store = PrefixStore::new();
        let head: Vec<Token> = (0..12_288).map(|i| (i % 7) as Token).collect();
        store.put(head.clone(), ckpt(12_288, 0xAB));
        let mut prompt = head.clone();
        prompt.extend((0..1_000).map(|i| (100 + i % 3) as Token));
        // Empty slot: local reuse_len is 0; lookup still uses the new prompt.
        let hit = store
            .find_best_for_bind(&prompt, 0)
            .expect("store must hit");
        assert_eq!(hit.n_tokens, 12_288, "reused == tool-head ubatch boundary");
        assert_eq!(hit.data, vec![0xAB; 12_288]);
    }

    #[test]
    fn prefix_store_skips_empty_or_mismatched_key() {
        let mut store = PrefixStore::new();
        store.put(Vec::new(), ckpt(4, 0x11));
        store.put(t(&[1, 2]), ckpt(4, 0x22));
        assert!(store.is_empty());
    }

    #[test]
    fn prefix_store_entry_cap_evicts_stubs() {
        // Production put() only inserts 8k–16k stubs. The store's own
        // max_entries must still shrink once only stubs remain.
        let mut store = PrefixStore::new();
        let mut keys = Vec::new();
        for i in 0..10u32 {
            let prefix: Vec<Token> = std::iter::repeat(i as Token).take(12_288).collect();
            keys.push(prefix.clone());
            store.put(prefix, ckpt(12_288, i as u8));
        }
        assert_eq!(
            store.len(),
            PrefixStore::DEFAULT_MAX_ENTRIES,
            "stub-only put must still honour max_entries"
        );
        assert!(
            !store.entries.contains_key(&keys[0]) && !store.entries.contains_key(&keys[1]),
            "oldest stubs must be the victims"
        );
        assert!(store.entries.contains_key(&keys[9]), "newest stub kept");
    }

    #[test]
    fn prefix_store_global_cap_keeps_stubs() {
        let mut store = PrefixStore::new();
        for i in 0..3u32 {
            let prefix: Vec<Token> = std::iter::repeat(i as Token).take(12_288).collect();
            store.put(prefix, ckpt(12_288, i as u8));
        }
        assert_eq!(store.len(), 3);
        store.evict_global_over_cap(0, 1);
        assert_eq!(
            store.len(),
            3,
            "global host-RAM cap must not drop store stubs"
        );
    }
}
