//! One KV sequence. State lives here; the scheduler only moves slots
//! between Empty / Prefilling / Decoding at iteration boundaries.

use std::time::{Duration, Instant};

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::error::Error;
use crate::generate::{FinishReason, GenerateParams, GeneratedToken, Utf8Buf};
use crate::reasoning::ThinkBudget;
use crate::sampler::{Sampler, SamplerParams};
use crate::tokenizer::Token;

/// Per-request stream from a slot back to the HTTP handler.
#[derive(Debug)]
pub enum SlotEvent {
    Token(GeneratedToken),
    /// Prefill still running. HTTP maps this to llama-server `prompt_progress`
    /// (`total`/`cache`/`processed`/`time_ms`) so a long prompt stays observable.
    PromptProgress {
        total: u32,
        cache: u32,
        processed: u32,
        time_ms: u64,
    },
    Finished {
        reason: FinishReason,
        prompt_tokens: u32,
        completion_tokens: u32,
        timings: SlotTimings,
    },
    Failed(Error),
}

/// llama-server-compatible per-request timings (ms / tok/s).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SlotTimings {
    /// Prefix tokens reused from the slot cache (not prefilled this request).
    pub cache_n: u32,
    /// Prompt tokens actually prefilled this request.
    pub prompt_n: u32,
    pub prompt_ms: f64,
    /// Generated tokens (sampled).
    pub predicted_n: u32,
    pub predicted_ms: f64,
    /// Speculative drafts proposed / accepted this request (not the bonus).
    pub draft_n: u32,
    pub draft_n_accepted: u32,
    /// Verify batches that consumed at least one draft (for mean accept len).
    pub draft_verif_steps: u32,
}

impl SlotTimings {
    pub fn prompt_per_token_ms(self) -> f64 {
        rate_ms(self.prompt_ms, self.prompt_n)
    }

    pub fn prompt_per_second(self) -> f64 {
        tokens_per_second(self.prompt_ms, self.prompt_n)
    }

    pub fn predicted_per_token_ms(self) -> f64 {
        rate_ms(self.predicted_ms, self.predicted_n)
    }

    pub fn predicted_per_second(self) -> f64 {
        tokens_per_second(self.predicted_ms, self.predicted_n)
    }

    pub fn total_ms(self) -> f64 {
        self.prompt_ms + self.predicted_ms
    }

    pub fn total_n(self) -> u32 {
        self.prompt_n.saturating_add(self.predicted_n)
    }
}

fn rate_ms(ms: f64, n: u32) -> f64 {
    if n == 0 { 0.0 } else { ms / f64::from(n) }
}

fn tokens_per_second(ms: f64, n: u32) -> f64 {
    if n == 0 || ms <= 0.0 {
        0.0
    } else {
        1e3 / ms * f64::from(n)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SlotId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotPhase {
    Empty,
    Prefilling,
    Decoding,
}

pub(crate) struct ActiveJob {
    pub request_id: u64,
    pub prompt_tokens: Vec<Token>,
    /// Start index for prefill. Tokens `[0..prompt_offset)` were reused from
    /// the slot's prefix cache (P3) and are NOT re-prefilled.
    pub prompt_offset: usize,
    pub prompt_pos: usize,
    pub n_past: u32,
    pub n_generated: u32,
    pub n_prompt: u32,
    pub max_tokens: u32,
    /// Original `max_tokens` request. `max_tokens` is clamped to the current
    /// slot cap and can rise again when a neighbor leaves.
    pub max_tokens_req: u32,
    /// Dynamic KV cap for this job (cells). Recomputed on join/leave.
    pub ctx_cap: u32,
    pub pending: Option<Token>,
    /// Speculative draft tokens to verify on the next decode (after `pending`).
    pub drafts: Vec<Token>,
    /// Sampled token IDs this request (not including EOG). Used to grow the
    /// slot prefix cache so the next turn's LCP covers the assistant reply.
    pub generated: Vec<Token>,
    /// Raw image/audio file bytes for `--mmproj` prefill.
    #[allow(dead_code)]
    pub images: Vec<Vec<u8>>,
    /// `spec_begin` already ran for this job.
    pub spec_begun: bool,
    pub draft_n: u32,
    pub draft_n_accepted: u32,
    pub draft_verif_steps: u32,
    /// Host-side speculative overhead this request (microseconds).
    pub spec_us_snapshot: u64,
    pub spec_us_draft: u64,
    pub spec_ckpt_tgt_bytes: usize,
    pub spec_ckpt_mtp_bytes: usize,
    /// Target/MTP snapshots taken before a speculative verify decode.
    pub spec_ckpt_tgt: Option<Vec<u8>>,
    pub spec_ckpt_mtp: Option<Vec<u8>>,
    pub spec_ckpt_n_past: u32,
    /// llama `spec_is_replay`: last verify restored a checkpoint and left
    /// `drafts` as the accepted prefix. Next decode re-verifies that prefix
    /// instead of drafting again (GDN snapshot ≠ 65-token batch numerics).
    pub spec_is_replay: bool,
    /// This decode is that replay (do not count drafts twice).
    pub spec_replaying: bool,
    pub sampler: Sampler,
    pub stop: Vec<String>,
    pub acc: String,
    pub(crate) utf8: Utf8Buf,
    pub think: ThinkBudget,
    pub reasoning_budget: u32,
    pub start_in_think: bool,
    pub cancel: CancellationToken,
    pub deadline: Option<Instant>,
    pub events: mpsc::UnboundedSender<SlotEvent>,
    pub finish: Option<FinishReason>,
    /// P3 metrics: when the slot was first occupied (TTFT baseline).
    pub started: Instant,
    /// P3 metrics: timestamp of the most recently emitted token (ITL delta).
    pub last_token_at: Instant,
    /// Prefill finished (generation clock). None while still prefilling.
    pub generation_started_at: Option<Instant>,
    /// Last journal progress line (prefill >3s / decode every 3s after 100 toks).
    pub last_progress_at: Instant,
    pub last_progress_n: u32,
}

impl ActiveJob {
    /// llama-server `result_prompt_progress` snapshot for this request.
    pub fn prompt_progress(&self) -> SlotEvent {
        SlotEvent::PromptProgress {
            total: self.n_prompt,
            cache: self.prompt_offset as u32,
            processed: self.prefill_cursor() as u32,
            time_ms: self.started.elapsed().as_millis() as u64,
        }
    }

    pub fn from_parts(
        request_id: u64,
        tokens: Vec<Token>,
        prompt_offset: usize,
        params: GenerateParams,
        cancel: CancellationToken,
        timeout: Option<Duration>,
        events: mpsc::UnboundedSender<SlotEvent>,
        ctx_cap: u32,
        images: Vec<Vec<u8>>,
    ) -> Self {
        let n_prompt = tokens.len() as u32;
        let max_tokens_req = params.max_tokens.max(1);
        let remaining = ctx_cap.saturating_sub(n_prompt).max(1);
        let max_tokens = max_tokens_req.min(remaining);
        let now = Instant::now();
        let deadline = timeout.map(|d| now + d);
        Self {
            request_id,
            prompt_tokens: tokens,
            prompt_offset,
            prompt_pos: 0,
            n_past: prompt_offset as u32,
            n_generated: 0,
            n_prompt,
            max_tokens,
            max_tokens_req,
            ctx_cap,
            pending: None,
            drafts: Vec::new(),
            generated: Vec::new(),
            images,
            spec_begun: false,
            draft_n: 0,
            draft_n_accepted: 0,
            draft_verif_steps: 0,
            spec_us_snapshot: 0,
            spec_us_draft: 0,
            spec_ckpt_tgt_bytes: 0,
            spec_ckpt_mtp_bytes: 0,
            spec_ckpt_tgt: None,
            spec_ckpt_mtp: None,
            spec_ckpt_n_past: 0,
            spec_is_replay: false,
            spec_replaying: false,
            sampler: Sampler::new(SamplerParams {
                temperature: params.temperature,
                top_p: params.top_p,
                top_k: params.top_k,
                seed: params.seed,
            }),
            stop: params.stop,
            acc: String::new(),
            utf8: Utf8Buf::default(),
            think: ThinkBudget::disabled(),
            reasoning_budget: params.reasoning_budget,
            start_in_think: params.start_in_think,
            cancel,
            deadline,
            events,
            finish: None,
            started: now,
            last_token_at: now,
            generation_started_at: None,
            last_progress_at: now,
            last_progress_n: prompt_offset as u32,
        }
    }

    pub fn timings(&self, now: Instant) -> SlotTimings {
        let (prompt_ms, predicted_ms) = match self.generation_started_at {
            Some(decode_t0) => (
                decode_t0
                    .saturating_duration_since(self.started)
                    .as_secs_f64()
                    * 1e3,
                now.saturating_duration_since(decode_t0).as_secs_f64() * 1e3,
            ),
            None => (
                now.saturating_duration_since(self.started).as_secs_f64() * 1e3,
                0.0,
            ),
        };
        SlotTimings {
            cache_n: self.prompt_offset as u32,
            prompt_n: self.n_prompt.saturating_sub(self.prompt_offset as u32),
            prompt_ms,
            predicted_n: self.n_generated,
            predicted_ms,
            draft_n: self.draft_n,
            draft_n_accepted: self.draft_n_accepted,
            draft_verif_steps: self.draft_verif_steps,
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }

    pub fn is_timed_out(&self) -> bool {
        self.deadline.is_some_and(|d| Instant::now() >= d)
    }

    /// Tokens already accounted for: reused prefix plus this request's prefill.
    /// `prompt_pos` is only the suffix this job actually prefills.
    pub fn prefill_cursor(&self) -> usize {
        self.prompt_offset + self.prompt_pos
    }

    pub fn prefill_remaining(&self) -> usize {
        self.prompt_tokens
            .len()
            .saturating_sub(self.prefill_cursor())
    }

    /// True once the full prompt is resident, including a reused prefix.
    /// `prompt_pos >= prompt_tokens.len()` is wrong when `prompt_offset > 0`.
    pub fn prefill_done(&self) -> bool {
        self.prefill_remaining() == 0
    }

    #[cfg(test)]
    pub(crate) fn for_test(prompt_tokens: Vec<Token>) -> Self {
        let (tx, _rx) = mpsc::unbounded_channel();
        Self::from_parts(
            0,
            prompt_tokens,
            0,
            GenerateParams::default(),
            CancellationToken::new(),
            None,
            tx,
            256,
            Vec::new(),
        )
    }
}

pub struct Slot {
    pub id: SlotId,
    pub phase: SlotPhase,
    pub(crate) job: Option<ActiveJob>,
    /// P3 prefix cache for this slot's sequence KV.
    pub prefix_cache: crate::prefix_cache::SlotPrefixCache,
    /// Prefill-end snapshot. DSV4 cannot `seq_rm` a long generated suffix
    /// (`n_rs_seq` is tiny), so the next bind restores this then trims 1 token.
    /// P8-A semantic anchor chain: prefill-end snapshots plus at most one
    /// store stub (last stride in the 6k–16k window), ascending by `n_tokens`.
    /// Same `n_tokens` replaces;
    /// a longer new anchor appends. `settle_prefix_kv` binds from the longest
    /// anchor with `n_tokens <= reuse_len + 1`. Bind then drops
    /// `n_tokens > reuse_len + 1` so affinity_bar cannot report a previous
    /// session. The chain lives only while the slot lives (watermark survival
    /// is #91).
    pub(crate) prefix_ckpts: Vec<SeqCheckpoint>,
    /// When this empty slot last retained prefix KV. `None` if nothing is held.
    pub(crate) retained_at: Option<Instant>,
}

/// Host copy of one sequence at a known length (see [`Slot::prefix_ckpts`]).
/// `pub` so the global `PrefixStore` can hold checkpoint values too.
#[derive(Clone, Debug)]
pub struct SeqCheckpoint {
    pub n_tokens: u32,
    pub data: Vec<u8>,
}

impl Slot {
    pub fn new(id: SlotId) -> Self {
        Self {
            id,
            phase: SlotPhase::Empty,
            job: None,
            prefix_cache: Default::default(),
            prefix_ckpts: Vec::new(),
            retained_at: None,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.phase == SlotPhase::Empty || self.job.is_none()
    }

    pub fn is_active(&self) -> bool {
        !self.is_empty()
    }

    pub(crate) fn occupy(&mut self, job: ActiveJob) {
        self.phase = SlotPhase::Prefilling;
        self.job = Some(job);
        self.retained_at = None;
    }

    pub(crate) fn evict(&mut self) -> Option<ActiveJob> {
        self.phase = SlotPhase::Empty;
        self.job.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate::GenerateParams;

    #[test]
    fn prefill_done_uses_offset_plus_pos() {
        let mut job = ActiveJob::for_test(vec![1, 2, 3, 4, 5, 6, 7, 8]);
        job.prompt_offset = 5;
        job.prompt_pos = 0;
        assert_eq!(job.prefill_cursor(), 5);
        assert_eq!(job.prefill_remaining(), 3);
        assert!(!job.prefill_done());
        assert!(
            job.prompt_pos < job.prompt_tokens.len(),
            "old apply_plan gate stays false after a reused prefix"
        );

        job.prompt_pos = 3;
        assert_eq!(job.prefill_cursor(), 8);
        assert_eq!(job.prefill_remaining(), 0);
        assert!(job.prefill_done());
        assert!(
            job.prompt_pos < job.prompt_tokens.len(),
            "suffix-only pos never reaches prompt_tokens.len() on a reused turn"
        );
    }

    #[test]
    fn prefill_done_first_request_has_no_offset() {
        let mut job = ActiveJob::for_test(vec![1, 2, 3]);
        assert!(!job.prefill_done());
        job.prompt_pos = 3;
        assert!(job.prefill_done());
        assert_eq!(job.prefill_cursor(), 3);
    }

    #[test]
    fn occupy_clears_retained_at() {
        let mut slot = Slot::new(SlotId(0));
        slot.retained_at = Some(Instant::now());
        slot.occupy(ActiveJob::for_test(vec![1]));
        assert!(slot.retained_at.is_none());
    }

    #[test]
    fn from_parts_seeds_n_past_from_reuse_len() {
        // P3: when a prefix is reused (prompt_offset > 0), n_past must be
        // seeded at reuse_len so prefill continues from the suffix.
        let (tx, _rx) = mpsc::unbounded_channel();
        let job = ActiveJob::from_parts(
            7,
            vec![1, 2, 3, 4],
            2, // reused prefix length
            GenerateParams::default(),
            CancellationToken::new(),
            None,
            tx,
            256,
            Vec::new(),
        );
        assert_eq!(job.prompt_offset, 2);
        assert_eq!(job.n_past, 2, "n_past seeded at reuse_len");
        assert_eq!(job.prompt_pos, 0);
        assert_eq!(job.n_prompt, 4);
        match job.prompt_progress() {
            SlotEvent::PromptProgress {
                total,
                cache,
                processed,
                time_ms: _,
            } => {
                assert_eq!(total, 4);
                assert_eq!(cache, 2);
                assert_eq!(processed, 2);
            }
            other => panic!("expected PromptProgress, got {other:?}"),
        }
    }

    #[test]
    fn from_parts_no_reuse_starts_at_zero() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let job = ActiveJob::from_parts(
            8,
            vec![9],
            0,
            GenerateParams::default(),
            CancellationToken::new(),
            None,
            tx,
            256,
            Vec::new(),
        );
        assert_eq!(job.prompt_offset, 0);
        assert_eq!(job.n_past, 0);
    }

    #[test]
    fn timings_rates_match_llama_server_formula() {
        let t = SlotTimings {
            cache_n: 2,
            prompt_n: 100,
            prompt_ms: 2000.0,
            predicted_n: 50,
            predicted_ms: 5000.0,
            ..SlotTimings::default()
        };
        assert!((t.prompt_per_token_ms() - 20.0).abs() < 1e-9);
        assert!((t.prompt_per_second() - 50.0).abs() < 1e-9);
        assert!((t.predicted_per_token_ms() - 100.0).abs() < 1e-9);
        assert!((t.predicted_per_second() - 10.0).abs() < 1e-9);
        assert!((t.total_ms() - 7000.0).abs() < 1e-9);
        assert_eq!(t.total_n(), 150);
    }

    #[test]
    fn timings_zero_tokens_or_ms_are_zero_not_inf() {
        let empty = SlotTimings::default();
        assert_eq!(empty.prompt_per_second(), 0.0);
        assert_eq!(empty.predicted_per_second(), 0.0);
        assert_eq!(empty.prompt_per_token_ms(), 0.0);
        let no_time = SlotTimings {
            predicted_n: 8,
            predicted_ms: 0.0,
            ..SlotTimings::default()
        };
        assert_eq!(no_time.predicted_per_second(), 0.0);
    }

    #[test]
    fn timings_splits_prefill_and_decode_clocks() {
        let mut job = ActiveJob::for_test(vec![1, 2, 3, 4]);
        job.prompt_offset = 1;
        job.n_prompt = 4;
        job.n_generated = 8;
        job.generation_started_at = Some(job.started + Duration::from_millis(200));
        let now = job.started + Duration::from_millis(700);
        let t = job.timings(now);
        assert_eq!(t.cache_n, 1);
        assert_eq!(t.prompt_n, 3);
        assert_eq!(t.predicted_n, 8);
        assert!(
            (t.prompt_ms - 200.0).abs() < 1.0,
            "prompt_ms={}",
            t.prompt_ms
        );
        assert!(
            (t.predicted_ms - 500.0).abs() < 1.0,
            "predicted_ms={}",
            t.predicted_ms
        );
    }
}
