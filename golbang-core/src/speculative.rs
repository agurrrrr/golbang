//! draft-mtp + ngram-mod + prompt-lookup, matching llama-server
//! `--spec-type draft-mtp,ngram-mod,prompt-lookup`.
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
    /// Prompt lookup (PLD): copy the tokens that followed an earlier occurrence
    /// of the trailing `pld_n` tokens. halogen `HALOGEN_PLD=3,3`.
    PromptLookup,
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
            "prompt-lookup" | "pld" => Ok(Self::PromptLookup),
            other => Err(format!(
                "unsupported --spec-type {other} (golbang implements draft-mtp,ngram-mod,prompt-lookup)"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::DraftMtp => "draft-mtp",
            Self::NgramMod => "ngram-mod",
            Self::PromptLookup => "prompt-lookup",
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
    /// ngram-mod `n_max` (llama-server default 64). Independent of MTP.
    pub ngram_n_max: i32,
    /// ngram-mod `n_min` (llama-server default 48). Shorter hits are dropped.
    pub ngram_n_min: i32,
    /// Prompt-lookup suffix length (`HALOGEN_PLD` N, default 3).
    pub pld_n: u32,
    /// Prompt-lookup continuation draft length (`HALOGEN_PLD` K, default 3).
    pub pld_k: u32,
    pub cache_type_k: u32,
    pub cache_type_v: u32,
}

impl Default for SpecParams {
    fn default() -> Self {
        Self {
            types: Vec::new(),
            n_max: 3,
            p_min: 0.90,
            ngram_n_max: 64,
            // llama default is 48. On a cold table MTP-fail steps then draft
            // 0 tokens (~10 t/s holes). 1 lets a short ngram fill those.
            ngram_n_min: 1,
            pld_n: 3,
            pld_k: 3,
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

    pub fn wants_pld(&self) -> bool {
        self.types.iter().any(|t| *t == SpecType::PromptLookup)
    }

    /// llama `common_speculative_n_max`: max of each enabled impl's n_max.
    /// Used as the verify budget (`dp.n_max` is remaining ctx/tokens).
    ///
    /// PLD contributes only `pld_k` (default 3), never the ngram 64, so enabling
    /// it does not inflate the CPU expert verify union.
    pub fn verify_n_max(&self) -> i32 {
        let mut n = 0i32;
        for t in &self.types {
            let v = match t {
                SpecType::DraftMtp => self.n_max,
                SpecType::NgramMod => self.ngram_n_max,
                SpecType::PromptLookup => self.pld_k as i32,
            };
            n = n.max(v);
        }
        n.max(0)
    }
}

/// Prompt lookup is greedy-solo only (halogen policy). A non-greedy sampler or
/// a shared batch step makes the copy pattern unreliable and the verify budget
/// contended, so the draft is suppressed.
pub fn pld_allowed(wants_pld: bool, greedy: bool, n_active: u32) -> bool {
    wants_pld && greedy && n_active <= 1
}

/// Recurrent-state rollback depth for a multi-token speculative reject.
///
/// llama-server sets `n_rs_seq = draft-mtp n_max`. PLD extends that chain (and
/// can run alone), so its `pld_k` must be covered too, or the scheduler copies
/// the ~150 MiB PARTIAL_ONLY state every verify step (#247). ngram-mod is
/// excluded here to match llama; it still snapshots.
pub fn spec_rs_need(spec: &SpecParams) -> u32 {
    let mtp = if spec.wants_mtp() { spec.n_max.max(0) } else { 0 };
    let pld = if spec.wants_pld() { spec.pld_k as i32 } else { 0 };
    mtp.max(pld).max(0) as u32
}

/// Prompt-lookup draft. `hist` is the request's token history (prompt +
/// generated, excluding `id_last`); `prior` are drafts an earlier impl (MTP)
/// already produced this step. The needle is the trailing `n` ids of
/// `hist ++ [id_last] ++ prior`, so the lookup chains off MTP proposals.
/// Returns the `k` tokens that followed an earlier occurrence of that needle.
pub fn pld_draft(
    hist: &[Token],
    id_last: Token,
    prior: &[Token],
    n: usize,
    k: usize,
    n_budget: usize,
) -> Vec<Token> {
    if n == 0 || k == 0 || n_budget == 0 {
        return Vec::new();
    }
    let k = k.min(n_budget);
    let mut full = Vec::with_capacity(hist.len() + 1 + prior.len());
    full.extend_from_slice(hist);
    full.push(id_last);
    full.extend_from_slice(prior);
    if full.len() <= n {
        return Vec::new();
    }
    let needle = &full[full.len() - n..];
    // `i == full.len() - n` is the needle itself; search strictly before it.
    for i in (0..=full.len() - n - 1).rev() {
        if &full[i..i + n] == needle {
            let start = i + n;
            let end = (start + k).min(full.len());
            return full[start..end].to_vec();
        }
    }
    Vec::new()
}

/// Hash table from llama.cpp `common_ngram_mod` (PR 19164).
pub struct NgramMod {
    n: usize,
    used: usize,
    entries: Vec<Token>,
    /// Per-seq last hist index from which ngrams were added (`draft_one` `i_last`).
    i_last: Vec<usize>,
    /// Last draft length returned for this seq (`draft_one` `n_draft_last`).
    n_draft_last: Vec<usize>,
    /// Consecutive accept rounds with fraction < 0.25 (`draft_one` `n_low`).
    n_low: Vec<i32>,
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
            n_draft_last: Vec::new(),
            n_low: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.entries.fill(NGRAM_EMPTY);
        self.used = 0;
        self.i_last.fill(0);
        self.n_draft_last.fill(0);
        self.n_low.fill(0);
    }

    fn ensure_seq(&mut self, seq_id: i32) -> usize {
        let i = seq_id.max(0) as usize;
        if i >= self.i_last.len() {
            self.i_last.resize(i + 1, 0);
            self.n_draft_last.resize(i + 1, 0);
            self.n_low.resize(i + 1, 0);
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

    pub fn note_draft(&mut self, seq_id: i32, n_draft: usize) {
        let sid = self.ensure_seq(seq_id);
        self.n_draft_last[sid] = n_draft;
    }

    /// llama `ngram_mod::accept` (`is_other == false`): reset the table after
    /// five consecutive rounds with accept fraction < 0.25.
    pub fn accept(&mut self, seq_id: i32, n_accepted: u16) {
        let sid = self.ensure_seq(seq_id);
        let n_draft = self.n_draft_last[sid];
        self.n_draft_last[sid] = 0;
        if n_draft == 0 {
            return;
        }
        let f_acc = n_accepted as f64 / n_draft as f64;
        if f_acc < 0.25 {
            self.n_low[sid] += 1;
            if self.n_low[sid] >= 5 {
                self.reset();
            }
        } else {
            self.n_low[sid] = 0;
        }
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
    fn spec_type_parses_prompt_lookup_aliases() {
        assert_eq!(
            SpecType::parse_list("prompt-lookup").unwrap(),
            vec![SpecType::PromptLookup]
        );
        assert_eq!(SpecType::parse_list("pld").unwrap(), vec![SpecType::PromptLookup]);
        assert_eq!(
            SpecType::parse_list("draft-mtp,pld").unwrap(),
            vec![SpecType::DraftMtp, SpecType::PromptLookup]
        );
        assert_eq!(SpecType::PromptLookup.as_str(), "prompt-lookup");
    }

    #[test]
    fn verify_n_max_counts_pld_as_k_not_64() {
        let mut p = SpecParams::default();
        p.types = vec![SpecType::PromptLookup];
        assert_eq!(p.verify_n_max(), 3);
        p.pld_k = 5;
        assert_eq!(p.verify_n_max(), 5);
        p.types = vec![SpecType::DraftMtp, SpecType::PromptLookup];
        assert_eq!(p.verify_n_max(), 5);
        p.types = vec![SpecType::NgramMod, SpecType::PromptLookup];
        assert_eq!(p.verify_n_max(), 64, "ngram still dominates");
        p.ngram_n_max = 2;
        assert_eq!(p.verify_n_max(), 5, "pld_k=5 now dominates");
    }

    #[test]
    fn pld_draft_copies_tokens_after_previous_suffix() {
        // history [.. a b c X Y Z .. a b c], needle is the final a b c.
        let hist = [1, 2, 3, 10, 11, 12, 1, 2];
        let draft = pld_draft(&hist, 3, &[], 3, 3, 3);
        assert_eq!(draft, vec![10, 11, 12]);
    }

    #[test]
    fn pld_draft_chains_off_mtp_prior() {
        // needle = [hist.last()=40, id_last=50, prior=60], found at i=0.
        let hist = [40, 50, 60, 99, 20, 30, 40];
        let draft = pld_draft(&hist, 50, &[60], 3, 2, 2);
        assert_eq!(draft, vec![99, 20]);
    }

    #[test]
    fn pld_draft_empty_without_match_or_room() {
        let hist = [1, 2, 3, 4, 5];
        assert!(pld_draft(&hist, 5, &[], 3, 3, 3).is_empty());
        assert!(pld_draft(&[1, 2], 3, &[], 3, 3, 3).is_empty());
        assert!(pld_draft(&hist, 5, &[], 0, 3, 3).is_empty());
        assert!(pld_draft(&hist, 5, &[], 3, 0, 3).is_empty());
        assert!(pld_draft(&hist, 5, &[], 3, 3, 0).is_empty());
    }

    #[test]
    fn pld_draft_respects_budget() {
        let hist = [1, 2, 3, 10, 11, 12, 1, 2];
        assert_eq!(pld_draft(&hist, 3, &[], 3, 3, 2), vec![10, 11]);
    }

    #[test]
    fn pld_gate_is_greedy_solo_only() {
        assert!(pld_allowed(true, true, 1));
        assert!(!pld_allowed(false, true, 1), "type off");
        assert!(!pld_allowed(true, false, 1), "non-greedy");
        assert!(!pld_allowed(true, true, 2), "batched");
    }

    #[test]
    fn spec_rs_need_covers_mtp_and_pld() {
        let mut p = SpecParams::default();
        assert_eq!(spec_rs_need(&p), 0, "nothing enabled");
        p.types = vec![SpecType::NgramMod];
        assert_eq!(spec_rs_need(&p), 0, "ngram excluded (llama snapshots)");
        p.types = vec![SpecType::DraftMtp];
        assert_eq!(spec_rs_need(&p), 3);
        p.n_max = 2;
        assert_eq!(spec_rs_need(&p), 2, "MTP n_max=2 like the Flash-Next unit");
        p.types = vec![SpecType::DraftMtp, SpecType::PromptLookup];
        assert_eq!(spec_rs_need(&p), 3, "PLD extends the reject span");
        p.types = vec![SpecType::PromptLookup];
        assert_eq!(spec_rs_need(&p), 3, "PLD alone");
    }

    #[test]
    fn verify_n_max_is_max_of_enabled_impls() {
        let mut p = SpecParams::default();
        assert_eq!(p.verify_n_max(), 0);
        p.types = vec![SpecType::DraftMtp];
        assert_eq!(p.verify_n_max(), 3);
        p.types = vec![SpecType::DraftMtp, SpecType::NgramMod];
        assert_eq!(p.verify_n_max(), 64);
        p.types = vec![SpecType::NgramMod];
        assert_eq!(p.verify_n_max(), 64);
        p.ngram_n_max = 16;
        assert_eq!(p.verify_n_max(), 16);
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
    fn ngram_low_accept_streak_resets_table() {
        let mut ng = NgramMod::new(2, 1024);
        ng.add(&[1, 2, 3]);
        for _ in 0..4 {
            ng.note_draft(0, 64);
            ng.accept(0, 1);
            assert_eq!(ng.get(&[1, 2]), 3, "need 5 low rounds to reset");
        }
        ng.note_draft(0, 64);
        ng.accept(0, 1);
        assert_eq!(ng.get(&[1, 2]), NGRAM_EMPTY);
        ng.add(&[1, 2, 3]);
        ng.note_draft(0, 64);
        ng.accept(0, 32);
        ng.note_draft(0, 64);
        ng.accept(0, 1);
        assert_eq!(ng.get(&[1, 2]), 3, "good round must clear the streak");
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
