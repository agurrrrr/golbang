//! One KV sequence. State lives here; the scheduler only moves slots
//! between Empty / Prefilling / Decoding at iteration boundaries.

use std::time::{Duration, Instant};

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::error::Error;
use crate::generate::{FinishReason, GenerateParams, GeneratedToken, Utf8Buf};
use crate::sampler::{Sampler, SamplerParams};
use crate::tokenizer::Token;

/// Per-request stream from a slot back to the HTTP handler.
#[derive(Debug)]
pub enum SlotEvent {
    Token(GeneratedToken),
    Finished {
        reason: FinishReason,
        prompt_tokens: u32,
        completion_tokens: u32,
    },
    Failed(Error),
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
    pub prompt_pos: usize,
    pub n_past: u32,
    pub n_generated: u32,
    pub n_prompt: u32,
    pub max_tokens: u32,
    pub pending: Option<Token>,
    pub sampler: Sampler,
    pub stop: Vec<String>,
    pub acc: String,
    pub(crate) utf8: Utf8Buf,
    pub cancel: CancellationToken,
    pub deadline: Option<Instant>,
    pub events: mpsc::UnboundedSender<SlotEvent>,
    pub finish: Option<FinishReason>,
}

impl ActiveJob {
    pub fn from_parts(
        request_id: u64,
        tokens: Vec<Token>,
        params: GenerateParams,
        cancel: CancellationToken,
        timeout: Option<Duration>,
        events: mpsc::UnboundedSender<SlotEvent>,
        n_ctx_seq: u32,
    ) -> Self {
        let n_prompt = tokens.len() as u32;
        let remaining = n_ctx_seq.saturating_sub(n_prompt).max(1);
        let max_tokens = params.max_tokens.max(1).min(remaining);
        let deadline = timeout.map(|d| Instant::now() + d);
        Self {
            request_id,
            prompt_tokens: tokens,
            prompt_pos: 0,
            n_past: 0,
            n_generated: 0,
            n_prompt,
            max_tokens,
            pending: None,
            sampler: Sampler::new(SamplerParams {
                temperature: params.temperature,
                top_p: params.top_p,
                top_k: params.top_k,
                seed: params.seed,
            }),
            stop: params.stop,
            acc: String::new(),
            utf8: Utf8Buf::default(),
            cancel,
            deadline,
            events,
            finish: None,
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }

    pub fn is_timed_out(&self) -> bool {
        self.deadline.is_some_and(|d| Instant::now() >= d)
    }

    #[cfg(test)]
    pub(crate) fn for_test(prompt_tokens: Vec<Token>) -> Self {
        let (tx, _rx) = mpsc::unbounded_channel();
        Self::from_parts(
            0,
            prompt_tokens,
            GenerateParams::default(),
            CancellationToken::new(),
            None,
            tx,
            256,
        )
    }
}

pub struct Slot {
    pub id: SlotId,
    pub phase: SlotPhase,
    pub(crate) job: Option<ActiveJob>,
}

impl Slot {
    pub fn new(id: SlotId) -> Self {
        Self {
            id,
            phase: SlotPhase::Empty,
            job: None,
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
    }

    pub(crate) fn evict(&mut self) -> Option<ActiveJob> {
        self.phase = SlotPhase::Empty;
        self.job.take()
    }
}
