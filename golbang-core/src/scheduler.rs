//! Async iteration loop. Join / evict / cancel run only at decode boundaries.
//! GPU work is `spawn_blocking`; HTTP `try_submit` never waits on decode.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::batch::BatchBuilder;
use crate::engine::Engine;
use crate::error::Error;
use crate::generate::{FinishReason, GenerateParams, GeneratedToken};
use crate::policy::{IterationBudget, SchedulePolicy, SlotView, WaitingJobView};
use crate::slot::{ActiveJob, Slot, SlotEvent, SlotId, SlotPhase, SlotTimings};

#[derive(Debug)]
pub enum SubmitError {
    Full,
    Closed,
}

impl std::fmt::Display for SubmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Full => write!(f, "scheduler queue is full"),
            Self::Closed => write!(f, "scheduler is closed"),
        }
    }
}

impl std::error::Error for SubmitError {}

#[derive(Clone, Debug)]
pub struct SchedulerConfig {
    pub n_parallel: u32,
    pub queue_capacity: usize,
    pub default_timeout: Option<Duration>,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            n_parallel: 2,
            queue_capacity: 2,
            default_timeout: None,
        }
    }
}

pub struct Job {
    pub request_id: u64,
    pub prompt: String,
    pub params: GenerateParams,
    pub cancel: CancellationToken,
    pub timeout: Option<Duration>,
    pub events: mpsc::UnboundedSender<SlotEvent>,
}

impl Job {
    pub fn new(
        prompt: String,
        params: GenerateParams,
        cancel: CancellationToken,
        events: mpsc::UnboundedSender<SlotEvent>,
    ) -> Self {
        static IDS: AtomicU64 = AtomicU64::new(1);
        Self {
            request_id: IDS.fetch_add(1, Ordering::Relaxed),
            prompt,
            params,
            cancel,
            timeout: None,
            events,
        }
    }
}

#[derive(Default)]
pub struct SchedulerMetrics {
    pub iterations: AtomicU64,
    pub joins: AtomicU64,
    pub evicts: AtomicU64,
    pub decodes: AtomicU64,
    // P3 §4.3 observability — histogram buckets are coarse (ms).
    // TTFT (time-to-first-token) and ITL (inter-token latency) buckets.
    pub ttft_bucket_ms: [AtomicU64; 8],
    pub itl_bucket_ms: [AtomicU64; 8],
    /// Slot occupancy: sampled count of active slots per iteration.
    pub slots_active_total: AtomicU64,
    pub slots_active_samples: AtomicU64,
    /// Queue depth: sampled waiting length per iteration.
    pub queue_depth_total: AtomicU64,
    pub queue_depth_samples: AtomicU64,
    /// 503 responses served at the HTTP edge (incremented by the server).
    pub service_unavailable_total: AtomicU64,
    /// Prompt tokens actually prefilled (excludes prefix-cache reuse).
    pub prompt_tokens_total: AtomicU64,
    /// Completion tokens sampled and emitted.
    pub tokens_generated_total: AtomicU64,
    /// Accumulated prefill / decode wall time (microseconds).
    pub prompt_us_total: AtomicU64,
    pub predicted_us_total: AtomicU64,
}

#[derive(Clone)]
pub struct SchedulerHandle {
    tx: mpsc::Sender<Job>,
    pub metrics: Arc<SchedulerMetrics>,
}

impl SchedulerHandle {
    /// Non-blocking. Full queue → immediate 503 at the HTTP edge.
    pub fn try_submit(&self, job: Job) -> Result<(), SubmitError> {
        match self.tx.try_send(job) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Full(_)) => Err(SubmitError::Full),
            Err(mpsc::error::TrySendError::Closed(_)) => Err(SubmitError::Closed),
        }
    }
}

pub struct SpawnedScheduler {
    pub handle: SchedulerHandle,
    pub worker: JoinHandle<()>,
}

pub fn spawn_scheduler(
    engine: Arc<Engine>,
    policy: Box<dyn SchedulePolicy>,
    config: SchedulerConfig,
) -> SpawnedScheduler {
    let cap = config.queue_capacity.max(1);
    let (tx, rx) = mpsc::channel(cap);
    let metrics = Arc::new(SchedulerMetrics::default());
    let handle = SchedulerHandle {
        tx,
        metrics: metrics.clone(),
    };
    tracing::info!(
        policy = policy.name(),
        n_parallel = config.n_parallel,
        queue_capacity = cap,
        "scheduler start"
    );
    let worker = tokio::spawn(run_loop(engine, policy, config, rx, metrics));
    SpawnedScheduler { handle, worker }
}

async fn run_loop(
    engine: Arc<Engine>,
    mut policy: Box<dyn SchedulePolicy>,
    config: SchedulerConfig,
    mut rx: mpsc::Receiver<Job>,
    metrics: Arc<SchedulerMetrics>,
) {
    let n = config.n_parallel.max(1) as usize;
    let n_seq_max = engine.n_seq_max() as usize;
    if n > n_seq_max {
        tracing::error!(
            n_parallel = n,
            n_seq_max,
            "n_parallel exceeds context n_seq_max"
        );
        return;
    }

    let mut slots: Vec<Slot> = (0..n).map(|i| Slot::new(SlotId(i as u32))).collect();
    let n_batch = engine.n_batch().max(1) as usize;
    let n_ubatch = engine.n_ubatch().max(1) as usize;
    let builder = BatchBuilder::new(n_batch);
    let budget = resolve_budget(policy.budget(), n_batch, n_ubatch, n);
    tracing::info!(
        n_batch,
        n_ubatch,
        prefill_max = budget.prefill_max,
        decode_max = budget.decode_max,
        "scheduler budget"
    );
    let n_ctx_seq = engine.n_ctx_seq();
    let mut waiting: VecDeque<Job> = VecDeque::new();
    let mut iter = 0u64;

    loop {
        iter += 1;
        metrics.iterations.store(iter, Ordering::Relaxed);

        // P3 §4.3: sample occupancy and queue depth each iteration.
        let active = slots.iter().filter(|s| s.is_active()).count() as u64;
        metrics
            .slots_active_total
            .fetch_add(active, Ordering::Relaxed);
        metrics.slots_active_samples.fetch_add(1, Ordering::Relaxed);
        let q = waiting.len() as u64;
        metrics.queue_depth_total.fetch_add(q, Ordering::Relaxed);
        metrics.queue_depth_samples.fetch_add(1, Ordering::Relaxed);

        while let Ok(job) = rx.try_recv() {
            waiting.push_back(job);
        }

        evict_cancelled(&mut slots, &engine, iter, &metrics);

        let extra = policy.evict(&slot_views(&slots));
        for id in extra {
            if let Some(slot) = slots.iter_mut().find(|s| s.id == id) {
                finish_slot(slot, &engine, FinishReason::Cancelled, iter, &metrics);
            }
        }

        join_waiting(
            &mut slots,
            &mut waiting,
            &mut *policy,
            &engine,
            n_ctx_seq,
            iter,
            &metrics,
        );

        let mut order = policy.rank(&slot_views(&slots));
        if order.is_empty() {
            order = slots
                .iter()
                .filter(|s| s.is_active())
                .map(|s| s.id)
                .collect();
        }
        let plan = builder.plan(&slots, &order, budget);

        if plan.is_empty() {
            if !has_active(&slots) && waiting.is_empty() {
                match rx.recv().await {
                    Some(job) => waiting.push_back(job),
                    None => {
                        tracing::info!(iter, "scheduler channel closed");
                        return;
                    }
                }
            } else {
                tokio::select! {
                    job = rx.recv() => {
                        match job {
                            Some(j) => waiting.push_back(j),
                            None => return,
                        }
                    }
                    _ = tokio::time::sleep(Duration::from_millis(5)) => {}
                }
            }
            continue;
        }

        let engine_d = engine.clone();
        let tokens = plan.tokens.clone();
        tracing::debug!(
            iter,
            n_tokens = tokens.len(),
            n_logits = plan.logit_slots.len(),
            "decode begin"
        );
        let decode_res =
            tokio::task::spawn_blocking(move || engine_d.decode_and_logits(&tokens)).await;
        metrics.decodes.fetch_add(1, Ordering::Relaxed);
        tracing::debug!(iter, "decode returned; join window open");

        let logits = match decode_res {
            Ok(Ok(rows)) => rows,
            Ok(Err(e)) => {
                tracing::error!(iter, error = %e, "decode failed");
                fail_all_active(&mut slots, &engine, e);
                continue;
            }
            Err(e) => {
                tracing::error!(iter, error = %e, "decode worker join failed");
                fail_all_active(&mut slots, &engine, Error::Decode(-1));
                continue;
            }
        };

        apply_plan(&mut slots, &plan);
        maybe_log_prefill_progress(&mut slots);
        sample_and_emit(&mut slots, &plan, &logits, &engine, n_ctx_seq, &metrics);
        evict_finished(&mut slots, &engine, iter, &metrics);
    }
}

/// The P3 32/16 default is a unit-test placeholder. A live context should
/// fill `n_batch` (single slot) or `n_ubatch` (shared GPU), matching llama-server.
fn resolve_budget(
    requested: IterationBudget,
    n_batch: usize,
    n_ubatch: usize,
    n_parallel: usize,
) -> IterationBudget {
    if requested == IterationBudget::default() {
        return IterationBudget::for_context(n_batch, n_ubatch, n_parallel);
    }
    IterationBudget {
        prefill_max: requested.prefill_max.min(n_batch).max(1),
        decode_max: requested.decode_max.max(1),
    }
}

fn slot_views(slots: &[Slot]) -> Vec<SlotView> {
    slots
        .iter()
        .filter(|s| s.is_active())
        .map(|s| SlotView {
            id: s.id,
            phase: s.phase,
            n_past: s.job.as_ref().map(|j| j.n_past).unwrap_or(0),
            n_generated: s.job.as_ref().map(|j| j.n_generated).unwrap_or(0),
        })
        .collect()
}

fn has_active(slots: &[Slot]) -> bool {
    slots.iter().any(|s| s.is_active())
}

fn join_waiting(
    slots: &mut [Slot],
    waiting: &mut VecDeque<Job>,
    policy: &mut dyn SchedulePolicy,
    engine: &Engine,
    n_ctx_seq: u32,
    iter: u64,
    metrics: &SchedulerMetrics,
) {
    let empty: Vec<SlotId> = slots
        .iter()
        .filter(|s| s.is_empty())
        .map(|s| s.id)
        .collect();
    if empty.is_empty() || waiting.is_empty() {
        return;
    }
    let views: Vec<WaitingJobView> = waiting
        .iter()
        .map(|j| WaitingJobView {
            request_id: j.request_id,
            n_prompt: 0,
        })
        .collect();
    let matches = policy.join(&empty, &views);

    let mut taken = vec![false; waiting.len()];
    let mut picks: Vec<(SlotId, usize)> = Vec::new();
    for (slot_id, wait_idx) in matches {
        if wait_idx < waiting.len()
            && !taken[wait_idx]
            && empty.contains(&slot_id)
            && slots.iter().any(|s| s.id == slot_id && s.is_empty())
        {
            taken[wait_idx] = true;
            picks.push((slot_id, wait_idx));
        }
    }
    picks.sort_by_key(|(_, i)| *i);

    let mut extracted: Vec<(SlotId, Job)> = Vec::new();
    let mut remain = VecDeque::new();
    for (i, job) in waiting.drain(..).enumerate() {
        if let Some(&(slot_id, _)) = picks.iter().find(|(_, wi)| *wi == i) {
            extracted.push((slot_id, job));
        } else {
            remain.push_back(job);
        }
    }
    *waiting = remain;

    for (slot_id, job) in extracted {
        let Some(slot) = slots.iter_mut().find(|s| s.id == slot_id) else {
            waiting.push_front(job);
            continue;
        };
        if bind_slot(slot, job, engine, n_ctx_seq) {
            metrics.joins.fetch_add(1, Ordering::Relaxed);
            tracing::info!(iter, slot = slot_id.0, "join after decode boundary");
        }
    }
}

fn bind_slot(slot: &mut Slot, job: Job, engine: &Engine, n_ctx_seq: u32) -> bool {
    let tokens = match engine.encode(&job.prompt) {
        Ok(t) if !t.is_empty() => t,
        Ok(_) => {
            let _ = job.events.send(SlotEvent::Failed(Error::EmptyPrompt));
            return false;
        }
        Err(e) => {
            let _ = job.events.send(SlotEvent::Failed(e));
            return false;
        }
    };
    let n_prompt = tokens.len() as u32;
    if n_prompt >= n_ctx_seq {
        let _ = job.events.send(SlotEvent::Failed(Error::ContextFull {
            prompt: n_prompt,
            n_ctx: n_ctx_seq,
        }));
        return false;
    }
    // P3 prefix reuse: the slot's cached prefix KV stays resident in seq.
    // Only the suffix `[reuse_len..)` needs prefill.
    let reuse_len = slot.prefix_cache.reuse(&tokens);
    if reuse_len == 0 {
        // No shared prefix → clear the slot's seq so positions start at 0.
        engine.clear_seq(slot.id.0 as i32);
    }
    let req = job.request_id;
    slot.occupy(ActiveJob::from_parts(
        job.request_id,
        tokens,
        reuse_len,
        job.params,
        job.cancel,
        job.timeout,
        job.events,
        n_ctx_seq,
    ));
    tracing::debug!(
        slot = slot.id.0,
        request_id = req,
        n_prompt,
        reused = reuse_len,
        "slot bound"
    );
    true
}

fn apply_plan(slots: &mut [Slot], plan: &crate::batch::BatchPlan) {
    for (id, take) in &plan.prefill_consumed {
        if let Some(slot) = slots.iter_mut().find(|s| s.id == *id) {
            if let Some(job) = slot.job.as_mut() {
                job.prompt_pos += *take as usize;
                job.n_past += *take;
                if job.prompt_pos >= job.prompt_tokens.len() {
                    slot.phase = SlotPhase::Decoding;
                    if job.generation_started_at.is_none() {
                        job.generation_started_at = Some(Instant::now());
                    }
                }
            }
        }
    }
    for tok in &plan.tokens {
        if !plan
            .prefill_consumed
            .iter()
            .any(|(id, _)| id.0 as i32 == tok.seq_id)
        {
            if let Some(slot) = slots.iter_mut().find(|s| s.id.0 as i32 == tok.seq_id) {
                if let Some(job) = slot.job.as_mut() {
                    job.n_past = (tok.pos as u32).saturating_add(1);
                }
            }
        }
    }
}

/// P3 §4.3: bucket a millisecond latency into a coarse histogram.
/// Bucket bounds (ms): 1, 5, 10, 25, 50, 100, 500 (index 7 = overflow).
const BUCKET_BOUNDS_MS: [u64; 8] = [1, 5, 10, 25, 50, 100, 500, u64::MAX];

fn record_bucket(bucket: &[AtomicU64; 8], ms: u64) {
    let idx = BUCKET_BOUNDS_MS.iter().position(|b| ms <= *b).unwrap_or(7);
    bucket[idx].fetch_add(1, Ordering::Relaxed);
}

fn sample_and_emit(
    slots: &mut [Slot],
    plan: &crate::batch::BatchPlan,
    logits: &[Vec<f32>],
    engine: &Engine,
    n_ctx_seq: u32,
    metrics: &SchedulerMetrics,
) {
    for (row_i, slot_id) in plan.logit_slots.iter().enumerate() {
        let Some(row) = logits.get(row_i) else {
            continue;
        };
        let Some(slot) = slots.iter_mut().find(|s| s.id == *slot_id) else {
            continue;
        };
        let Some(job) = slot.job.as_mut() else {
            continue;
        };
        if job.finish.is_some() {
            continue;
        }

        // P3 §4.3: TTFT is the first emitted token's latency from join.
        // ITL is the per-token inter-arrival time for subsequent tokens.
        let now = Instant::now();
        if job.n_generated == 0 {
            let ms = now.duration_since(job.started).as_millis() as u64;
            record_bucket(&metrics.ttft_bucket_ms, ms);
            if job.generation_started_at.is_none() {
                job.generation_started_at = Some(now);
            }
        } else {
            let ms = now.duration_since(job.last_token_at).as_millis() as u64;
            record_bucket(&metrics.itl_bucket_ms, ms);
        }
        job.last_token_at = now;

        let token = job.sampler.sample(row);
        if engine.is_eog(token) {
            let _ = job.utf8.flush();
            job.finish = Some(FinishReason::Stop);
            continue;
        }

        let bytes = match engine.token_to_piece(token) {
            Ok(b) => b,
            Err(e) => {
                let _ = job.events.send(SlotEvent::Failed(e));
                job.finish = Some(FinishReason::Stop);
                continue;
            }
        };
        let mut piece = job.utf8.push(&bytes);
        job.n_generated += 1;
        job.pending = Some(token);
        slot.phase = SlotPhase::Decoding;
        maybe_log_decode_progress(slot.id.0, job);

        if job.n_generated >= job.max_tokens {
            piece.push_str(&job.utf8.flush());
            job.finish = Some(FinishReason::Length);
        } else if !job.stop.is_empty() {
            job.acc.push_str(&piece);
            if job
                .stop
                .iter()
                .any(|s| !s.is_empty() && job.acc.contains(s))
            {
                job.finish = Some(FinishReason::Stop);
            }
        }

        if job.n_past + 1 > n_ctx_seq && job.finish.is_none() {
            job.finish = Some(FinishReason::Length);
        }

        let _ = job
            .events
            .send(SlotEvent::Token(GeneratedToken { token, piece }));
    }
}

fn evict_cancelled(slots: &mut [Slot], engine: &Engine, iter: u64, metrics: &SchedulerMetrics) {
    for slot in slots.iter_mut() {
        let Some(job) = slot.job.as_ref() else {
            continue;
        };
        if job.is_cancelled() {
            finish_slot(slot, engine, FinishReason::Cancelled, iter, metrics);
        } else if job.is_timed_out() {
            finish_slot(slot, engine, FinishReason::Timeout, iter, metrics);
        }
    }
}

fn evict_finished(slots: &mut [Slot], engine: &Engine, iter: u64, metrics: &SchedulerMetrics) {
    let ids: Vec<SlotId> = slots
        .iter()
        .filter(|s| s.job.as_ref().is_some_and(|j| j.finish.is_some()))
        .map(|s| s.id)
        .collect();
    for id in ids {
        if let Some(slot) = slots.iter_mut().find(|s| s.id == id) {
            let reason = slot
                .job
                .as_ref()
                .and_then(|j| j.finish)
                .unwrap_or(FinishReason::Stop);
            finish_slot(slot, engine, reason, iter, metrics);
        }
    }
}

fn finish_slot(
    slot: &mut Slot,
    engine: &Engine,
    reason: FinishReason,
    iter: u64,
    metrics: &SchedulerMetrics,
) {
    let id = slot.id;
    if let Some(job) = slot.evict() {
        engine.clear_seq(id.0 as i32);
        let timings = job.timings(Instant::now());
        record_request_totals(metrics, &timings);
        log_slot_timings(id.0, job.request_id, reason.as_str(), &timings);
        let ev = match reason {
            FinishReason::Cancelled => SlotEvent::Failed(Error::Cancelled),
            FinishReason::Timeout => SlotEvent::Failed(Error::Timeout),
            other => SlotEvent::Finished {
                reason: other,
                prompt_tokens: job.n_prompt,
                completion_tokens: job.n_generated,
                timings,
            },
        };
        let _ = job.events.send(ev);
        metrics.evicts.fetch_add(1, Ordering::Relaxed);
        tracing::info!(
            iter,
            slot = id.0,
            request_id = job.request_id,
            reason = reason.as_str(),
            "evict"
        );
    }
}

fn record_request_totals(metrics: &SchedulerMetrics, t: &SlotTimings) {
    metrics
        .prompt_tokens_total
        .fetch_add(u64::from(t.prompt_n), Ordering::Relaxed);
    metrics
        .tokens_generated_total
        .fetch_add(u64::from(t.predicted_n), Ordering::Relaxed);
    metrics
        .prompt_us_total
        .fetch_add((t.prompt_ms * 1e3) as u64, Ordering::Relaxed);
    metrics
        .predicted_us_total
        .fetch_add((t.predicted_ms * 1e3) as u64, Ordering::Relaxed);
}

fn log_slot_timings(slot: u32, request_id: u64, reason: &str, t: &SlotTimings) {
    tracing::info!(
        "slot print_timing: id {slot} | task {request_id} | reason {reason} | cache_n {}",
        t.cache_n
    );
    tracing::info!(
        "prompt eval time = {:10.2} ms / {:5} tokens ({:8.2} ms per token, {:8.2} tokens per second)",
        t.prompt_ms,
        t.prompt_n,
        t.prompt_per_token_ms(),
        t.prompt_per_second(),
    );
    tracing::info!(
        "       eval time = {:10.2} ms / {:5} tokens ({:8.2} ms per token, {:8.2} tokens per second)",
        t.predicted_ms,
        t.predicted_n,
        t.predicted_per_token_ms(),
        t.predicted_per_second(),
    );
    tracing::info!(
        "      total time = {:10.2} ms / {:5} tokens",
        t.total_ms(),
        t.total_n(),
    );
}

/// llama-server `print_timings_pp`: long prefills emit a progress line every 3s.
fn maybe_log_prefill_progress(slots: &mut [Slot]) {
    const MIN_MS: u128 = 3000;
    for slot in slots.iter_mut() {
        if slot.phase != SlotPhase::Prefilling {
            continue;
        }
        let Some(job) = slot.job.as_mut() else {
            continue;
        };
        let elapsed = job.started.elapsed();
        if elapsed.as_millis() < MIN_MS {
            continue;
        }
        if job.last_progress_at.elapsed().as_millis() < MIN_MS {
            continue;
        }
        let processed = job.prompt_pos as u32;
        let remaining = job
            .prompt_tokens
            .len()
            .saturating_sub(job.prompt_offset + job.prompt_pos) as u32;
        let total = processed.saturating_add(remaining);
        let progress = if total == 0 {
            1.0
        } else {
            f64::from(processed) / f64::from(total)
        };
        let secs = elapsed.as_secs_f64();
        let tps = if secs > 0.0 {
            f64::from(processed) / secs
        } else {
            0.0
        };
        tracing::info!(
            slot = slot.id.0,
            request_id = job.request_id,
            n_tokens = processed,
            progress,
            tps,
            "prompt processing, n_tokens = {processed}, progress = {progress:.2}, t = {secs:.2} s / {tps:.2} tokens per second"
        );
        job.last_progress_at = Instant::now();
        job.last_progress_n = processed;
    }
}

/// llama-server `print_timings_tg`: running decode speed after 100 tokens, every 3s.
fn maybe_log_decode_progress(slot: u32, job: &mut ActiveJob) {
    const MIN_DECODED: u32 = 100;
    const MIN_MS: u128 = 3000;
    if job.n_generated < MIN_DECODED {
        return;
    }
    if job.last_progress_at.elapsed().as_millis() < MIN_MS {
        return;
    }
    let Some(started_gen) = job.generation_started_at else {
        return;
    };
    let gen_secs = started_gen.elapsed().as_secs_f64();
    let tg = if gen_secs > 0.0 {
        f64::from(job.n_generated) / gen_secs
    } else {
        0.0
    };
    let win_n = job.n_generated.saturating_sub(job.last_progress_n);
    let win_secs = job.last_progress_at.elapsed().as_secs_f64();
    let tg_3s = if win_secs > 0.0 {
        f64::from(win_n) / win_secs
    } else {
        0.0
    };
    tracing::info!(
        slot,
        request_id = job.request_id,
        n_decoded = job.n_generated,
        tg,
        tg_3s,
        "n_decoded = {:6}, tg = {:6.2} t/s, tg_3s = {:6.2} t/s",
        job.n_generated,
        tg,
        tg_3s,
    );
    job.last_progress_at = Instant::now();
    job.last_progress_n = job.n_generated;
}

fn fail_all_active(slots: &mut [Slot], engine: &Engine, err: Error) {
    for slot in slots.iter_mut() {
        let id = slot.id;
        if let Some(job) = slot.evict() {
            engine.clear_seq(id.0 as i32);
            let _ = job.events.send(SlotEvent::Failed(err.clone()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_job() -> Job {
        let (tx, _rx) = mpsc::unbounded_channel();
        Job::new(
            "hi".into(),
            GenerateParams::default(),
            CancellationToken::new(),
            tx,
        )
    }

    #[test]
    fn default_budget_follows_single_slot_n_batch() {
        let b = resolve_budget(IterationBudget::default(), 5800, 1024, 1);
        assert_eq!(b.prefill_max, 5800);
        assert_eq!(b.decode_max, 1);
    }

    #[test]
    fn explicit_budget_is_clamped_to_n_batch() {
        let b = resolve_budget(
            IterationBudget {
                prefill_max: 99999,
                decode_max: 4,
            },
            5800,
            1024,
            1,
        );
        assert_eq!(b.prefill_max, 5800);
        assert_eq!(b.decode_max, 4);
    }

    #[tokio::test]
    async fn try_submit_rejects_when_full() {
        let (tx, _rx) = mpsc::channel(1);
        let handle = SchedulerHandle {
            tx,
            metrics: Arc::new(SchedulerMetrics::default()),
        };
        assert!(handle.try_submit(dummy_job()).is_ok());
        assert!(matches!(
            handle.try_submit(dummy_job()),
            Err(SubmitError::Full)
        ));
    }
}

#[cfg(all(test, unix))]
mod gpu_tests {
    use super::*;
    use crate::engine::Engine;
    use crate::model::{LoadParams, Model};
    use crate::policy::FifoPolicy;
    use std::fs::OpenOptions;
    use std::os::unix::io::AsRawFd;
    use std::path::Path;
    use std::time::{Duration, Instant};

    fn lock_gpu() -> std::fs::File {
        let f = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open("/tmp/golbang-gpu-test.lock")
            .expect("gpu lock");
        let rc = unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX) };
        assert_eq!(rc, 0, "flock");
        f
    }

    fn test_model() -> Option<String> {
        match std::env::var("GOLBANG_TEST_MODEL") {
            Ok(p) if !p.is_empty() && Path::new(&p).is_file() => Some(p),
            _ => {
                eprintln!("skip: set GOLBANG_TEST_MODEL");
                None
            }
        }
    }

    async fn collect_job(
        handle: &SchedulerHandle,
        prompt: &str,
        max_tokens: u32,
    ) -> (Duration, u32, Option<FinishReason>) {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let t0 = Instant::now();
        let job = Job::new(
            prompt.to_string(),
            GenerateParams {
                max_tokens,
                temperature: 0.0,
                ..GenerateParams::default()
            },
            CancellationToken::new(),
            tx,
        );
        handle.try_submit(job).expect("submit");
        let mut ttft = None;
        let mut n = 0u32;
        let mut reason = None;
        while let Some(ev) = rx.recv().await {
            match ev {
                SlotEvent::Token(_) => {
                    n += 1;
                    if ttft.is_none() {
                        ttft = Some(t0.elapsed());
                    }
                }
                SlotEvent::Finished {
                    reason: r,
                    completion_tokens,
                    ..
                } => {
                    reason = Some(r);
                    if n == 0 {
                        n = completion_tokens;
                    }
                    break;
                }
                SlotEvent::Failed(e) => panic!("job failed: {e}"),
            }
        }
        (ttft.unwrap_or_else(|| t0.elapsed()), n, reason)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn two_slots_both_complete_and_late_ttft_beats_serial() {
        let Some(path) = test_model() else {
            return;
        };
        let _lock = lock_gpu();

        let model = tokio::task::spawn_blocking(move || {
            Model::load(
                path,
                LoadParams {
                    n_ctx: 256,
                    n_gpu_layers: 99,
                    n_seq_max: 2,
                    ..Default::default()
                },
            )
        })
        .await
        .expect("join")
        .expect("load");
        let engine = Arc::new(Engine::new(model));

        let serial = spawn_scheduler(
            engine.clone(),
            Box::new(FifoPolicy::default()),
            SchedulerConfig {
                n_parallel: 1,
                queue_capacity: 4,
                default_timeout: None,
            },
        );
        let prompt = "<|im_start|>user\n안녕<|im_end|>\n<|im_start|>assistant\n";
        let t_serial = Instant::now();
        let s1 = collect_job(&serial.handle, prompt, 8);
        let s2 = collect_job(&serial.handle, prompt, 8);
        let (r1, r2) = tokio::join!(s1, s2);
        let serial_wall = t_serial.elapsed();
        let late_serial = r1.0.max(r2.0);
        drop(serial.handle);
        let _ = serial.worker.await;

        let parallel = spawn_scheduler(
            engine,
            Box::new(FifoPolicy::default()),
            SchedulerConfig {
                n_parallel: 2,
                queue_capacity: 2,
                default_timeout: None,
            },
        );
        assert!(parallel.handle.metrics.joins.load(Ordering::Relaxed) == 0);
        let t_par = Instant::now();
        let p1 = collect_job(&parallel.handle, prompt, 8);
        let p2 = collect_job(&parallel.handle, prompt, 8);
        let (q1, q2) = tokio::join!(p1, p2);
        let par_wall = t_par.elapsed();
        let late_par = q1.0.max(q2.0);
        let joins = parallel.handle.metrics.joins.load(Ordering::Relaxed);
        let decodes = parallel.handle.metrics.decodes.load(Ordering::Relaxed);
        drop(parallel.handle);
        let _ = parallel.worker.await;

        assert!(r1.1 >= 1 && r2.1 >= 1, "serial produced tokens");
        assert!(q1.1 >= 1 && q2.1 >= 1, "parallel produced tokens");
        assert!(joins >= 2, "both jobs should join, joins={joins}");
        assert!(decodes >= 1, "at least one decode");
        eprintln!(
            "P2 TTFT late serial={late_serial:?} parallel={late_par:?} \
             wall serial={serial_wall:?} parallel={par_wall:?} joins={joins} decodes={decodes}"
        );
        assert!(
            late_par < late_serial,
            "late TTFT should improve vs n_parallel=1 sequential wait: par={late_par:?} serial={late_serial:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn cancel_is_accepted_immediately_and_reclaimed_after_decode() {
        let Some(path) = test_model() else {
            return;
        };
        let _lock = lock_gpu();

        let model = tokio::task::spawn_blocking(move || {
            Model::load(
                path,
                LoadParams {
                    n_ctx: 256,
                    n_gpu_layers: 99,
                    n_seq_max: 1,
                    ..Default::default()
                },
            )
        })
        .await
        .expect("join")
        .expect("load");
        let engine = Arc::new(Engine::new(model));
        let spawned = spawn_scheduler(
            engine,
            Box::new(FifoPolicy::default()),
            SchedulerConfig {
                n_parallel: 1,
                queue_capacity: 1,
                default_timeout: None,
            },
        );

        let (tx, mut rx) = mpsc::unbounded_channel();
        let cancel = CancellationToken::new();
        let job = Job::new(
            "<|im_start|>user\n안녕<|im_end|>\n<|im_start|>assistant\n".into(),
            GenerateParams {
                max_tokens: 64,
                temperature: 0.0,
                ..GenerateParams::default()
            },
            cancel.clone(),
            tx,
        );
        spawned.handle.try_submit(job).expect("submit");
        let first = tokio::time::timeout(Duration::from_secs(30), rx.recv())
            .await
            .expect("first event")
            .expect("event");
        assert!(matches!(first, SlotEvent::Token(_)), "got {first:?}");
        cancel.cancel();
        let end = tokio::time::timeout(Duration::from_secs(30), async {
            while let Some(ev) = rx.recv().await {
                if matches!(
                    ev,
                    SlotEvent::Failed(Error::Cancelled) | SlotEvent::Finished { .. }
                ) {
                    return ev;
                }
            }
            panic!("channel closed without cancel/finish");
        })
        .await
        .expect("reclaim at decode boundary");
        assert!(
            matches!(end, SlotEvent::Failed(Error::Cancelled)),
            "slot reclaim should surface cancel, got {end:?}"
        );
        drop(spawned.handle);
        let _ = spawned.worker.await;
    }
}
