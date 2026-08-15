//! draft-mtp + ngram-mod, matching llama-server `--spec-type draft-mtp,ngram-mod`.
//!
//! Verification is: decode `[sampled, draft…]` on the target, then sample each
//! row until a mismatch. The last sampled token (mismatch or bonus) is not yet
//! in the KV and becomes the next pending token.
//!
//! `spec_draft` may write MTP KV at `n_past…` to score candidates. Those cells
//! must be removed before the next `spec_process` — Qwen35 is M-RoPE and
//! rejects `Y <= X`. llama-server does `seq_rm(ctx_dft, pos_max+1, -1)`
//! immediately after `common_speculative_draft`.

use crate::tokenizer::Token;

/// llama-server `--spec-type` names we implement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpecType {
    DraftMtp,
    NgramMod,
}

impl SpecType {
    pub fn parse_list(s: &str) -> Result<Vec<Self>, String> {
        let mut out = Vec::new();
        for part in s.split(',') {
            let p = part.trim();
            if p.is_empty() {
                continue;
            }
            out.push(Self::from_name(p)?);
        }
        Ok(out)
    }

    pub fn from_name(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "draft-mtp" | "mtp" => Ok(Self::DraftMtp),
            "ngram-mod" => Ok(Self::NgramMod),
            other => Err(format!(
                "unsupported --spec-type {other} (golbang implements draft-mtp,ngram-mod)"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::DraftMtp => "draft-mtp",
            Self::NgramMod => "ngram-mod",
        }
    }
}

#[derive(Clone, Debug)]
pub struct SpecParams {
    pub types: Vec<SpecType>,
    /// MTP `n_max` (llama-server `--spec-draft-n-max`, default 3).
    pub n_max: i32,
    /// MTP min draft probability (`--spec-draft-p-min`).
    pub p_min: f32,
    pub cache_type_k: u32,
    pub cache_type_v: u32,
}

impl Default for SpecParams {
    fn default() -> Self {
        Self {
            types: Vec::new(),
            n_max: 3,
            p_min: 0.90,
            cache_type_k: golbang_sys::GGML_TYPE_Q8_0 as u32,
            cache_type_v: golbang_sys::GGML_TYPE_Q8_0 as u32,
        }
    }
}

impl SpecParams {
    pub fn enabled(&self) -> bool {
        !self.types.is_empty()
    }

    pub fn wants_mtp(&self) -> bool {
        self.types.iter().any(|t| *t == SpecType::DraftMtp)
    }

    pub fn wants_ngram(&self) -> bool {
        self.types.iter().any(|t| *t == SpecType::NgramMod)
    }
}

/// Hash table from llama.cpp `common_ngram_mod` (PR 19164).
pub struct NgramMod {
    n: usize,
    used: usize,
    entries: Vec<Token>,
    /// Per-seq last hist index from which ngrams were added (`draft_one` `i_last`).
    i_last: Vec<usize>,
}

const NGRAM_EMPTY: Token = -1;
const NGRAM_HASH: u64 = 6364136223846793005;
/// llama.cpp `draft_one`: add `hist[i_last .. cur_len-n]` only after this growth.
const NGRAM_CHUNK: usize = 32;

impl NgramMod {
    pub fn new(n: u16, size: usize) -> Self {
        Self {
            n: n.max(1) as usize,
            used: 0,
            entries: vec![NGRAM_EMPTY; size.max(1)],
            i_last: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.entries.fill(NGRAM_EMPTY);
        self.used = 0;
    }

    fn ensure_seq(&mut self, seq_id: i32) -> usize {
        let i = seq_id.max(0) as usize;
        if i >= self.i_last.len() {
            self.i_last.resize(i + 1, 0);
        }
        i
    }

    fn idx(&self, tokens: &[Token]) -> usize {
        let mut res: u64 = 0;
        for &t in tokens.iter().take(self.n) {
            res = res.wrapping_mul(NGRAM_HASH).wrapping_add(t as u64);
        }
        (res % self.entries.len() as u64) as usize
    }

    /// `tokens` is n+1 ids: the n-gram key followed by the value.
    pub fn add(&mut self, tokens: &[Token]) {
        if tokens.len() <= self.n {
            return;
        }
        let i = self.idx(tokens);
        if self.entries[i] == NGRAM_EMPTY {
            self.used += 1;
        }
        self.entries[i] = tokens[self.n];
    }

    pub fn get(&self, tokens: &[Token]) -> Token {
        if tokens.len() < self.n {
            return NGRAM_EMPTY;
        }
        self.entries[self.idx(tokens)]
    }

    pub fn ingest(&mut self, tokens: &[Token]) {
        if tokens.len() <= self.n {
            return;
        }
        for i in 0..tokens.len() - self.n {
            self.add(&tokens[i..]);
        }
        let occ = self.used as f64 / self.entries.len() as f64;
        if occ > 0.25 {
            self.reset();
        }
    }

    /// llama `begin`: seed the table from the prompt and set per-seq `i_last`.
    pub fn begin(&mut self, seq_id: i32, tokens: &[Token]) {
        let sid = self.ensure_seq(seq_id);
        self.i_last[sid] = 0;
        self.ingest(tokens);
        if tokens.len() > self.n {
            self.i_last[sid] = tokens.len() - self.n;
        }
    }

    /// llama `draft_one` chunk add: when `hist` grew ≥32 past `i_last`, add
    /// `hist[i_last .. len-n]` (includes self-generated tokens now in hist).
    pub fn ingest_new(&mut self, seq_id: i32, hist: &[Token]) {
        let sid = self.ensure_seq(seq_id);
        let n = self.n;
        let cur_len = hist.len();
        if cur_len < n {
            return;
        }
        let i_last = self.i_last[sid];
        if i_last + NGRAM_CHUNK >= cur_len {
            return;
        }
        let end = cur_len - n;
        if i_last >= end {
            return;
        }
        for i in i_last..end {
            self.add(&hist[i..]);
        }
        self.i_last[sid] = end;
    }

    /// Draft up to `n_max` tokens after `id_last`. Empty if fewer than `n_min`.
    pub fn draft(
        &self,
        prompt: &[Token],
        id_last: Token,
        n_max: usize,
        n_min: usize,
    ) -> Vec<Token> {
        if prompt.len() < self.n || n_max == 0 {
            return Vec::new();
        }
        let mut window = Vec::with_capacity(self.n + n_max);
        window.extend_from_slice(&prompt[prompt.len() - self.n + 1..]);
        window.push(id_last);
        debug_assert_eq!(window.len(), self.n);

        let mut drafted = Vec::new();
        for _ in 0..n_max {
            let tok = self.get(&window[window.len() - self.n..]);
            if tok == NGRAM_EMPTY {
                break;
            }
            drafted.push(tok);
            window.push(tok);
        }
        if drafted.len() < n_min {
            return Vec::new();
        }
        drafted
    }
}

/// Target logits after decoding `[sampled, draft…]`.
/// Returns accepted tokens: matching drafts plus the first mismatch or bonus.
pub fn accept_drafts(samples: &[Token], drafts: &[Token]) -> Vec<Token> {
    debug_assert!(samples.len() <= drafts.len() + 1);
    let mut accepted = Vec::new();
    for (i, &draft) in drafts.iter().enumerate() {
        let Some(&id) = samples.get(i) else {
            break;
        };
        accepted.push(id);
        if id != draft {
            return accepted;
        }
    }
    if let Some(&bonus) = samples.get(drafts.len()) {
        accepted.push(bonus);
    }
    accepted
}

/// Softmax top-k then pick the mode; returns (token, probability).
///
/// Vocab is 248k on Qwen3.8. Collecting every (index, logit) pair and
/// `select_nth` on that Vec was several milliseconds per MTP draft step.
pub fn topk_mode(logits: &[f32], top_k: usize) -> (Token, f32) {
    if logits.is_empty() {
        return (0, 0.0);
    }
    let k = top_k.max(1).min(logits.len());
    // `best[0]` is the current min of the k-set (logit, index).
    let mut best: Vec<(f32, usize)> = Vec::with_capacity(k);
    for (i, &l) in logits.iter().enumerate() {
        if best.len() < k {
            best.push((l, i));
            if best.len() == k {
                best.select_nth_unstable_by(0, |a, b| a.0.total_cmp(&b.0));
            }
            continue;
        }
        if l > best[0].0 {
            best[0] = (l, i);
            best.select_nth_unstable_by(0, |a, b| a.0.total_cmp(&b.0));
        }
    }
    let max_l = best.iter().map(|p| p.0).fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    let mut best_i = 0usize;
    let mut best_p = 0.0f32;
    for &(l, i) in &best {
        let p = (l - max_l).exp();
        sum += p;
        if p >= best_p {
            best_p = p;
            best_i = i;
        }
    }
    if !sum.is_finite() || sum <= 0.0 {
        return (best_i as Token, 1.0);
    }
    (best_i as Token, best_p / sum)
}

pub fn parse_ggml_type(s: &str) -> Result<u32, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "f32" | "fp32" => Ok(golbang_sys::GGML_TYPE_F32 as u32),
        "f16" | "fp16" => Ok(golbang_sys::GGML_TYPE_F16 as u32),
        "q8_0" => Ok(golbang_sys::GGML_TYPE_Q8_0 as u32),
        "q4_0" => Ok(golbang_sys::GGML_TYPE_Q4_0 as u32),
        other => Err(format!(
            "unsupported cache type {other} (use q8_0|f16|f32|q4_0)"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_type_parses_llama_server_list() {
        let types = SpecType::parse_list("draft-mtp,ngram-mod").unwrap();
        assert_eq!(types, vec![SpecType::DraftMtp, SpecType::NgramMod]);
    }

    #[test]
    fn spec_type_rejects_unknown() {
        let err = SpecType::parse_list("draft-eagle3").unwrap_err();
        assert!(err.contains("draft-eagle3"), "{err}");
    }

    #[test]
    fn ngram_roundtrip_and_draft() {
        let mut ng = NgramMod::new(2, 1024);
        // keys (1,2)->3, (2,3)->4, (3,4)->5
        ng.add(&[1, 2, 3]);
        ng.add(&[2, 3, 4]);
        ng.add(&[3, 4, 5]);
        assert_eq!(ng.get(&[1, 2]), 3);
        assert_eq!(ng.get(&[2, 3]), 4);
        let draft = ng.draft(&[1, 2], 3, 4, 1);
        assert_eq!(draft, vec![4, 5]);
    }

    #[test]
    fn ngram_draft_clears_below_n_min() {
        let mut ng = NgramMod::new(2, 64);
        ng.add(&[1, 2, 3]);
        assert!(ng.draft(&[1, 2], 3, 4, 2).is_empty());
    }

    #[test]
    fn ngram_begin_sets_i_last_and_chunk_add_ingests_new_hist() {
        let mut ng = NgramMod::new(2, 1024);
        let mut hist: Vec<Token> = (0..10).collect();
        ng.begin(0, &hist);
        assert_eq!(ng.get(&[0, 1]), 2);
        assert_eq!(ng.get(&[7, 8]), 9);
        // i_last = 8; 8+32 >= 10 so a short tail is ignored
        hist.extend_from_slice(&[10, 11, 12]);
        ng.ingest_new(0, &hist);
        assert_eq!(ng.get(&[8, 9]), NGRAM_EMPTY);
        // cur_len > i_last+32 → add hist[8 .. len-2]
        hist.extend((13..41).collect::<Vec<Token>>());
        assert!(hist.len() > 8 + NGRAM_CHUNK);
        ng.ingest_new(0, &hist);
        assert_eq!(ng.get(&[8, 9]), 10);
        assert_eq!(
            ng.get(&[hist[hist.len() - 3], hist[hist.len() - 2]]),
            hist[hist.len() - 1]
        );
    }

    #[test]
    fn ngram_chunk_add_is_per_seq() {
        let mut ng = NgramMod::new(2, 1024);
        let hist0: Vec<Token> = (0..41).collect();
        let mut hist1: Vec<Token> = (100..108).collect();
        ng.begin(0, &hist0[..10]);
        ng.begin(1, &hist1);
        hist1.extend_from_slice(&[108, 109, 110]);
        ng.ingest_new(0, &hist0);
        ng.ingest_new(1, &hist1);
        assert_eq!(ng.get(&[8, 9]), 10);
        assert_eq!(
            ng.get(&[106, 107]),
            NGRAM_EMPTY,
            "seq 1 grew <32 so must not ingest 108"
        );
    }

    #[test]
    fn accept_all_drafts_keeps_bonus() {
        let accepted = accept_drafts(&[10, 11, 12, 99], &[10, 11, 12]);
        assert_eq!(accepted, vec![10, 11, 12, 99]);
    }

    #[test]
    fn accept_stops_at_first_mismatch() {
        let accepted = accept_drafts(&[10, 77], &[10, 11, 12]);
        assert_eq!(accepted, vec![10, 77]);
    }

    #[test]
    fn accept_first_token_mismatch() {
        let accepted = accept_drafts(&[7], &[10, 11]);
        assert_eq!(accepted, vec![7]);
    }

    #[test]
    fn topk_mode_picks_largest_logit() {
        let (id, p) = topk_mode(&[0.0, 5.0, 1.0], 10);
        assert_eq!(id, 1);
        assert!(p > 0.5, "p={p}");
    }

    #[test]
    fn topk_mode_scans_without_materializing_vocab() {
        let mut logits = vec![0.0f32; 248];
        logits[17] = 4.0;
        logits[200] = 9.0;
        logits[3] = 3.0;
        let (id, p) = topk_mode(&logits, 2);
        assert_eq!(id, 200);
        assert!(p > 0.9, "p={p}");
    }
}
