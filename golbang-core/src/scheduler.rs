//! Async iteration loop. Join / evict / cancel run only at decode boundaries.
//! GPU work is `spawn_blocking`; HTTP `try_submit` never waits on decode.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::batch::{
    BatchBuilder, BatchToken, MixMode, decide_mix_mode, has_pending_decode, max_prefill_remaining,
};
use crate::ctx_cap::{cap_for_join, effective_cap, resolve_single_max, slot_cap};
use crate::engine::Engine;
use crate::error::Error;
use crate::generate::{FinishReason, GenerateParams, GeneratedToken};
use crate::policy::{EmptySlotView, IterationBudget, SchedulePolicy, SlotView, WaitingJobView};
use crate::prefix_cache::{PrefixStore, common_prefix_len, host_search_len, snapshot_key};
use crate::slot::{ActiveJob, SeqCheckpoint, Slot, SlotEvent, SlotId, SlotPhase, SlotTimings};
use crate::speculative::accept_drafts;
use crate::tokenizer::Token;

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
    /// Solo-slot KV cap (`--single-max-ctx`). `0` = full pool (`n_ctx * n_seq_max`).
    pub single_max_ctx: u32,
    /// Speculative verify budget (`SpecParams::verify_n_max()`). `0` = spec off.
    /// Caps reserve `(spec_n_max + 1) * n_active` draft cells from the pool so
    /// a verify step never overruns into another sequence's cells (#8565).
    pub spec_n_max: u32,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            n_parallel: 2,
            queue_capacity: 2,
            default_timeout: None,
            single_max_ctx: 0,
            spec_n_max: 0,
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
    /// Decoded image bytes for `--mmproj` (one per `<__media__>` marker).
    pub images: Vec<Vec<u8>>,
    /// Filled at join so prefix-aware pick and bind share one tokenize.
    pub prompt_tokens: Option<Vec<Token>>,
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
            images: Vec::new(),
            prompt_tokens: None,
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
    /// Speculative draft tokens proposed / accepted (not counting the bonus).
    pub draft_tokens_total: AtomicU64,
    pub draft_accepted_total: AtomicU64,
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
        single_max_ctx = config.single_max_ctx,
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
    let prefix_store = Arc::new(std::sync::Mutex::new(PrefixStore::with_cap(
        PrefixStore::DEFAULT_MAX_BYTES,
    )));
    let n_batch = engine.n_batch().max(1) as usize;
    let n_ubatch = engine.n_ubatch().max(1) as usize;
    let builder = BatchBuilder::new(n_batch);
    let budget = resolve_budget(policy.budget(), n_batch, n_ubatch, n);
    tracing::info!(
        n_batch,
        n_ubatch,
        prefill_max = budget.prefill_max,
        decode_max = budget.decode_max,
        mixed_prefill_max = budget.mixed_prefill_max,
        prefill_yield_every = budget.prefill_yield_every,
        prefill_yield_max = budget.prefill_yield_max,
        "scheduler budget"
    );
    let pool = engine.n_ctx();
    let single_max = resolve_single_max(config.single_max_ctx, pool);
    let spec_n_max = config.spec_n_max;
    tracing::info!(
        pool,
        single_max,
        spec_n_max,
        n_ctx_seq = engine.n_ctx_seq(),
        "scheduler ctx pool"
    );
    let mut waiting: VecDeque<Job> = VecDeque::new();
    let mut iter = 0u64;
    let mut decode_iters_since_prefill_yield = 0u32;

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

        // Free idle prefix before join so a waiting job sees the extra room
        // in this iteration instead of failing ContextFull and retrying.
        release_retained_under_pressure(&mut slots, &*engine, pool, &prefix_store);

        join_waiting(
            &mut slots,
            &mut waiting,
            &mut *policy,
            &engine,
            pool,
            single_max,
            spec_n_max,
            iter,
            &metrics,
            &prefix_store,
        );
        recompute_slot_caps(&mut slots, pool, single_max, spec_n_max);

        let mut order = policy.rank(&slot_views(&slots));
        if order.is_empty() {
            order = slots
                .iter()
                .filter(|s| s.is_active())
                .map(|s| s.id)
                .collect();
        }
        let mix = decide_mix_mode(&slots, budget, decode_iters_since_prefill_yield);
        let remaining_prefill = max_prefill_remaining(&slots);
        let plan = if mix == MixMode::PrefillOnly {
            let mut yield_budget = budget;
            yield_budget.prefill_max = budget.yield_chunk(remaining_prefill);
            decode_iters_since_prefill_yield = 0;
            tracing::info!(
                remaining = remaining_prefill,
                take = yield_budget.prefill_max,
                "prefill yield; decode paused this iteration"
            );
            builder.plan_prefill_only(&slots, &order, yield_budget)
        } else {
            if remaining_prefill > 0 && has_pending_decode(&slots) {
                decode_iters_since_prefill_yield =
                    decode_iters_since_prefill_yield.saturating_add(1);
            } else {
                decode_iters_since_prefill_yield = 0;
            }
            builder.plan(&slots, &order, budget)
        };

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
        let metrics_d = metrics.clone();
        let prefix_store_d = prefix_store.clone();
        let n_slots = slots.len();
        tracing::debug!(
            iter,
            n_tokens = plan.tokens.len(),
            n_logits = plan.logit_slots.len(),
            "decode begin"
        );
        // Target decode, in-place sample, seq_rm/replay, and MTP draft are
        // all HIP. One blocking worker keeps them on the same thread.
        let gpu = tokio::task::spawn_blocking(move || {
            snapshot_spec_slots(&mut slots, &engine_d, &plan);
            let samples = match engine_d.decode_and_sample(&plan.tokens, &mut |i, row| {
                let sid = plan.logit_slots.get(i).copied();
                sid.and_then(|id| {
                    slots
                        .iter_mut()
                        .find(|s| s.id == id)
                        .and_then(|s| s.job.as_mut())
                        .map(|j| {
                            j.think
                                .forced_token()
                                .unwrap_or_else(|| j.sampler.sample(row))
                        })
                })
                .unwrap_or(0)
            }) {
                Ok(s) => s,
                Err(e) => return (slots, Err(e)),
            };
            apply_plan(&mut slots, &plan, engine_d.as_ref(), &prefix_store_d);
            maybe_log_prefill_progress(&mut slots);
            sample_and_emit(&mut slots, &plan, &samples, &engine_d, &metrics_d);
            fill_drafts_sync(&mut slots, &engine_d);
            (slots, Ok(()))
        })
        .await;
        metrics.decodes.fetch_add(1, Ordering::Relaxed);
        tracing::debug!(iter, "decode returned; join window open");

        match gpu {
            Ok((s, Ok(()))) => slots = s,
            Ok((s, Err(e))) => {
                slots = s;
                tracing::error!(iter, error = %e, "decode failed");
                fail_all_active(&mut slots, &engine, e);
                continue;
            }
            Err(e) => {
                tracing::error!(iter, error = %e, "decode worker join failed");
                slots = (0..n_slots).map(|i| Slot::new(SlotId(i as u32))).collect();
                continue;
            }
        }
        evict_finished(&mut slots, &engine, iter, &metrics);
        release_retained_under_pressure(&mut slots, &*engine, pool, &prefix_store);
        recompute_slot_caps(&mut slots, pool, single_max, spec_n_max);
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
        mixed_prefill_max: requested.mixed_prefill_max.min(n_batch),
        prefill_yield_every: requested.prefill_yield_every,
        prefill_yield_max: requested.prefill_yield_max.min(n_batch),
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
    pool: u32,
    single_max: u32,
    spec_n_max: u32,
    iter: u64,
    metrics: &SchedulerMetrics,
    prefix_store: &Arc<std::sync::Mutex<PrefixStore>>,
) {
    let empty: Vec<SlotId> = slots
        .iter()
        .filter(|s| s.is_empty())
        .map(|s| s.id)
        .collect();
    if empty.is_empty() || waiting.is_empty() {
        return;
    }
    let n_cand = empty.len().min(waiting.len());
    for job in waiting.iter_mut().take(n_cand) {
        if job.images.is_empty() && job.prompt_tokens.is_none() {
            if let Ok(t) = engine.encode(&job.prompt) {
                job.prompt_tokens = Some(t);
            }
        }
    }
    let matches = {
        let empty_views: Vec<EmptySlotView<'_>> = empty
            .iter()
            .filter_map(|id| slots.iter().find(|s| s.id == *id))
            .map(|s| EmptySlotView {
                id: s.id,
                prefix: s.prefix_cache.tokens.as_slice(),
                prefix_len: s.prefix_cache.prefix_len as u32,
                ckpt_n: latest_ckpt_n(s),
            })
            .collect();
        let views: Vec<WaitingJobView<'_>> = waiting
            .iter()
            .map(|j| {
                let tokens = j.prompt_tokens.as_deref().unwrap_or(&[]);
                WaitingJobView {
                    request_id: j.request_id,
                    n_prompt: tokens.len() as u32,
                    tokens,
                }
            })
            .collect();
        for job in &views {
            for slot in &empty_views {
                let lcp = common_prefix_len(slot.prefix, job.tokens);
                tracing::info!(
                    slot = slot.id.0,
                    request_id = job.request_id,
                    lcp,
                    prefix_n = slot.prefix.len(),
                    ckpt_n = slot.ckpt_n,
                    aff_bar = slot.affinity_bar(),
                    affinity = crate::policy::is_prefix_affinity(lcp, slot.affinity_bar()),
                    "join candidate"
                );
            }
        }
        policy.join(&empty_views, &views)
    };

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
        if !slots.iter().any(|s| s.id == slot_id) {
            waiting.push_front(job);
            continue;
        }
        let n_active_after = slots.iter().filter(|s| s.is_active()).count() as u32 + 1;
        let others_used: u32 = slots
            .iter()
            .filter(|s| s.id != slot_id)
            .map(slot_kv_used_for_cap)
            .sum();
        let cap = cap_for_join(others_used, n_active_after, pool, single_max, spec_n_max);
        let (prefix_n, ckpt_n, lcp) = slots
            .iter()
            .find(|s| s.id == slot_id)
            .map(|s| {
                let prefix_n = s.prefix_cache.tokens.len();
                let ckpt_n = latest_ckpt_n(s);
                let lcp = job
                    .prompt_tokens
                    .as_deref()
                    .map(|t| common_prefix_len(&s.prefix_cache.tokens, t))
                    .unwrap_or(0);
                (prefix_n, ckpt_n, lcp)
            })
            .unwrap_or((0, 0, 0));
        let req = job.request_id;
        let bound = {
            let Some(slot) = slots.iter_mut().find(|s| s.id == slot_id) else {
                waiting.push_front(job);
                continue;
            };
            bind_slot(slot, job, engine, cap, prefix_store)
        };
        if !bound {
            tracing::warn!(
                iter,
                slot = slot_id.0,
                request_id = req,
                cap,
                others_used,
                n_active_after,
                lcp,
                prefix_n,
                ckpt_n,
                "join bind rejected"
            );
        }
        if bound {
            metrics.joins.fetch_add(1, Ordering::Relaxed);
            tracing::info!(
                iter,
                slot = slot_id.0,
                request_id = req,
                cap,
                others_used,
                n_active_after,
                lcp,
                prefix_n,
                ckpt_n,
                "join after decode boundary"
            );
            recompute_slot_caps(slots, pool, single_max, spec_n_max);
        }
    }
    // HiCache L2 (4th stage): a vision bind may have captured a new chain
    // dump (`capture_prefix_checkpoint`); enforce the global host-RAM cap.
    enforce_host_ram_cap(slots, prefix_store, HOST_RAM_CAP);
}

/// Sequence KV ops that bind / restore / promote need.
///
/// [`Engine`] is the production impl. Tests inject an in-memory fake so
/// `bind_slot` → `settle_prefix_kv` → `restore_host_snapshot` (and
/// `apply_plan` promotion) run without loading a GGUF.
trait SeqKv {
    fn spec_reset_seq(&self, seq: i32);
    fn n_past_seq(&self, seq: i32) -> u32;
    fn clear_seq(&self, seq: i32);
    fn seq_state_set(&self, seq: i32, data: &[u8]) -> bool;
    fn rm_seq_from(&self, seq: i32, p0: i32) -> bool;
    fn seq_state_get(&self, seq: i32) -> Option<Vec<u8>>;
    fn n_ubatch(&self) -> u32;
    fn encode(&self, text: &str) -> crate::error::Result<Vec<Token>>;
    fn tokenize_special(&self, text: &str) -> crate::error::Result<Vec<Token>>;
    fn bind_vision(&self, slot: &mut Slot, job: Job, ctx_cap: u32) -> bool {
        let _ = (slot, ctx_cap);
        let _ = job.events.send(SlotEvent::Failed(Error::VisionDisabled));
        false
    }
}

impl SeqKv for Engine {
    fn spec_reset_seq(&self, seq: i32) {
        Engine::spec_reset_seq(self, seq);
    }
    fn n_past_seq(&self, seq: i32) -> u32 {
        Engine::n_past_seq(self, seq)
    }
    fn clear_seq(&self, seq: i32) {
        Engine::clear_seq(self, seq);
    }
    fn seq_state_set(&self, seq: i32, data: &[u8]) -> bool {
        Engine::seq_state_set(self, seq, data)
    }
    fn rm_seq_from(&self, seq: i32, p0: i32) -> bool {
        Engine::rm_seq_from(self, seq, p0)
    }
    fn seq_state_get(&self, seq: i32) -> Option<Vec<u8>> {
        Engine::seq_state_get(self, seq)
    }
    fn n_ubatch(&self) -> u32 {
        Engine::n_ubatch(self)
    }
    fn encode(&self, text: &str) -> crate::error::Result<Vec<Token>> {
        Engine::encode(self, text)
    }
    fn tokenize_special(&self, text: &str) -> crate::error::Result<Vec<Token>> {
        Engine::tokenize_special(self, text)
    }
    fn bind_vision(&self, slot: &mut Slot, job: Job, ctx_cap: u32) -> bool {
        bind_vision_slot(slot, job, self, ctx_cap)
    }
}

fn bind_slot(
    slot: &mut Slot,
    mut job: Job,
    engine: &impl SeqKv,
    ctx_cap: u32,
    prefix_store: &Arc<std::sync::Mutex<PrefixStore>>,
) -> bool {
    let seq = slot.id.0 as i32;
    engine.spec_reset_seq(seq);

    if !job.images.is_empty() {
        return engine.bind_vision(slot, job, ctx_cap);
    }

    let tokens = match job.prompt_tokens.take() {
        Some(t) if !t.is_empty() => t,
        Some(_) => {
            let _ = job.events.send(SlotEvent::Failed(Error::EmptyPrompt));
            return false;
        }
        None => match engine.encode(&job.prompt) {
            Ok(t) if !t.is_empty() => t,
            Ok(_) => {
                let _ = job.events.send(SlotEvent::Failed(Error::EmptyPrompt));
                return false;
            }
            Err(e) => {
                let _ = job.events.send(SlotEvent::Failed(e));
                return false;
            }
        },
    };
    let n_prompt = tokens.len() as u32;
    if n_prompt >= ctx_cap {
        tracing::warn!(
            slot = slot.id.0,
            request_id = job.request_id,
            n_prompt,
            ctx_cap,
            "prompt exceeds context cap"
        );
        let _ = job.events.send(SlotEvent::Failed(Error::ContextFull {
            prompt: n_prompt,
            n_ctx: ctx_cap,
        }));
        return false;
    }
    // P5: keep resident KV for the LCP. DSV4 cannot seq_rm a long generated
    // suffix (n_rs_seq is 1), so we restore the prefill checkpoint then trim.
    let gpu_n = engine.n_past_seq(seq);
    let ckpt_n = latest_ckpt_n(slot);
    let hint_n = gpu_n.max(ckpt_n);
    let mut reuse_len = slot.prefix_cache.reuse_for_bind(&tokens, hint_n);
    reuse_len = settle_prefix_kv(slot, engine, reuse_len, gpu_n, prefix_store, &tokens);
    let req = job.request_id;
    slot.occupy(ActiveJob::from_parts(
        job.request_id,
        tokens,
        reuse_len,
        job.params,
        job.cancel,
        job.timeout,
        job.events,
        ctx_cap,
        job.images,
    ));
    arm_think_budget(slot, engine);
    if let Some(active) = slot.job.as_ref() {
        emit_prompt_progress(active);
    }
    tracing::info!(
        slot = slot.id.0,
        request_id = req,
        n_prompt,
        reused = reuse_len,
        gpu_n,
        ctx_cap,
        "slot bound"
    );
    true
}

fn bind_vision_slot(slot: &mut Slot, job: Job, engine: &Engine, ctx_cap: u32) -> bool {
    if !engine.vision_enabled() {
        let _ = job.events.send(SlotEvent::Failed(Error::VisionDisabled));
        return false;
    }
    let seq = slot.id.0 as i32;
    let tok = match engine.vision_tokenize(&job.prompt, &job.images) {
        Ok(t) if !t.seq.tokens.is_empty() => t,
        Ok(_) => {
            let _ = job.events.send(SlotEvent::Failed(Error::EmptyPrompt));
            return false;
        }
        Err(e) => {
            let _ = job.events.send(SlotEvent::Failed(e));
            return false;
        }
    };
    let n_pos = tok.seq.n_pos();
    if n_pos >= ctx_cap {
        let _ = job.events.send(SlotEvent::Failed(Error::ContextFull {
            prompt: n_pos,
            n_ctx: ctx_cap,
        }));
        return false;
    }

    let gpu_n = engine.n_past_seq(seq);
    let ckpt_n = latest_ckpt_n(slot);
    let hint_n = gpu_n.max(ckpt_n);
    let reuse_tok = slot
        .prefix_cache
        .vision
        .as_ref()
        .map(|prev| prev.reuse_for_bind(&tok.seq, hint_n))
        .unwrap_or(0);
    // Vision: no host-store prefix hit (README: image turns have none).
    // `tok.seq.tokens` embeds image markers, so a PrefixStore lookup would be
    // wrong. Only local vision prefix reuse; otherwise start from a clean seq.
    if reuse_tok == 0 {
        engine.clear_seq(seq);
        slot.prefix_cache.reset();
        clear_prefix_chain(slot);
    }
    let reuse_pos = tok.seq.pos_next(reuse_tok);

    let eval_t0 = Instant::now();
    let n_past = match engine.vision_eval_from(seq, &tok, reuse_tok, reuse_pos) {
        Ok(n) => n,
        Err(e) => {
            engine.clear_seq(seq);
            slot.prefix_cache.reset();
            clear_prefix_chain(slot);
            let _ = job.events.send(SlotEvent::Failed(e));
            return false;
        }
    };
    if n_past >= ctx_cap {
        let _ = job.events.send(SlotEvent::Failed(Error::ContextFull {
            prompt: n_past,
            n_ctx: ctx_cap,
        }));
        return false;
    }

    let req = job.request_id;
    let n_img = job.images.len();
    let vision_seq = tok.seq.clone();
    drop(tok);
    slot.occupy(ActiveJob::from_parts(
        job.request_id,
        vision_seq.tokens.clone(),
        reuse_pos as usize,
        job.params,
        job.cancel,
        job.timeout,
        job.events,
        ctx_cap,
        job.images,
    ));
    arm_think_budget(slot, engine);
    if let Some(active) = slot.job.as_mut() {
        active.started = eval_t0;
        active.generation_started_at = Some(Instant::now());
        active.n_past = n_past;
        active.n_prompt = n_past;
        active.prompt_offset = reuse_pos as usize;
        active.prompt_pos = 0;
        slot.phase = SlotPhase::Decoding;
        emit_prompt_progress(active);
    }
    slot.prefix_cache.tokens.clear();
    slot.prefix_cache.prefix_len = reuse_tok;
    slot.prefix_cache.vision = Some(vision_seq);
    capture_prefix_checkpoint(slot, engine);
    if let Err(e) = sample_from_existing_logits(slot, engine) {
        if let Some(job) = slot.job.take() {
            let _ = job.events.send(SlotEvent::Failed(e));
        }
        slot.phase = SlotPhase::Empty;
        return false;
    }
    maybe_fill_drafts(slot, engine);
    tracing::info!(
        slot = slot.id.0,
        request_id = req,
        n_past,
        n_images = n_img,
        reuse_tok,
        reuse_pos,
        gpu_n,
        ctx_cap,
        "vision slot bound"
    );
    true
}

fn arm_think_budget(slot: &mut Slot, engine: &impl SeqKv) {
    let Some(job) = slot.job.as_mut() else {
        return;
    };
    let limit = job.reasoning_budget;
    if limit == 0 {
        return;
    }
    match engine.tokenize_special("</think>\n\n") {
        Ok(toks) if !toks.is_empty() => {
            job.think = crate::reasoning::ThinkBudget::new(limit, job.start_in_think, toks);
            tracing::info!(
                slot = slot.id.0,
                request_id = job.request_id,
                start_in_think = job.start_in_think,
                budget = limit,
                close_n = job.think.close_len(),
                "reasoning budget armed"
            );
        }
        Ok(_) => tracing::warn!(
            slot = slot.id.0,
            "reasoning-budget: empty </think> tokenization; disabled"
        ),
        Err(e) => tracing::warn!(
            slot = slot.id.0,
            error = %e,
            "reasoning-budget: tokenize </think> failed"
        ),
    }
}

fn sample_from_existing_logits(slot: &mut Slot, engine: &Engine) -> crate::error::Result<()> {
    let row = engine.last_logits()?;
    let token = match slot.job.as_mut() {
        Some(job) => job
            .think
            .forced_token()
            .unwrap_or_else(|| job.sampler.sample(&row)),
        None => return Ok(()),
    };
    emit_sampled(slot, engine, &[token], None);
    Ok(())
}

fn apply_plan(
    slots: &mut [Slot],
    plan: &crate::batch::BatchPlan,
    engine: &impl SeqKv,
    prefix_store: &Arc<std::sync::Mutex<PrefixStore>>,
) {
    for (id, take) in &plan.prefill_consumed {
        if let Some(slot) = slots.iter_mut().find(|s| s.id == *id) {
            let mut need_chain = false;
            let mut promote_key: Option<Vec<Token>> = None;
            let n_ubatch = engine.n_ubatch().max(1) as u32;
            if let Some(job) = slot.job.as_mut() {
                job.prompt_pos += *take as usize;
                job.n_past += *take;
                if *take > 0 {
                    // llama-server sends `prompt_progress` after every ubatch.
                    emit_prompt_progress(job);
                    job.last_progress_n = job.prefill_cursor() as u32;
                }
                let n_past = job.n_past;
                let stride_stub = is_host_stub_boundary(n_past, n_ubatch);
                if job.prefill_done() {
                    slot.phase = SlotPhase::Decoding;
                    job.last_progress_n = 0;
                    job.last_progress_at = Instant::now();
                    if job.generation_started_at.is_none() {
                        job.generation_started_at = Some(Instant::now());
                        need_chain = true;
                    }
                    if stride_stub {
                        promote_key = snapshot_key(&job.prompt_tokens, n_past);
                    }
                } else if stride_stub {
                    // One stub in the chain: the last 8k–16k stride boundary.
                    // Other stride dumps go to the host store only.
                    need_chain = is_last_host_stub(n_past);
                    promote_key = snapshot_key(&job.prompt_tokens, n_past);
                }
            }
            if need_chain || promote_key.is_some() {
                let n_tokens = slot.job.as_ref().map(|j| j.n_past).unwrap_or(0);
                match engine.seq_state_get(slot.id.0 as i32) {
                    Some(data) if !data.is_empty() => match (need_chain, promote_key) {
                        (true, Some(key)) => {
                            if is_store_stub_len(n_tokens) {
                                slot.prefix_ckpts.retain(|c| {
                                    !is_store_stub_len(c.n_tokens) || c.n_tokens == n_tokens
                                });
                            }
                            push_prefix_ckpt(slot, n_tokens, data.clone());
                            let bytes = data.len();
                            prefix_store
                                .lock()
                                .unwrap()
                                .put(key, SeqCheckpoint { n_tokens, data });
                            tracing::info!(
                                slot = slot.id.0,
                                n_tokens,
                                bytes,
                                "host prefix snapshot promoted"
                            );
                        }
                        (true, None) => {
                            push_prefix_ckpt(slot, n_tokens, data);
                        }
                        (false, Some(key)) => {
                            let bytes = data.len();
                            prefix_store
                                .lock()
                                .unwrap()
                                .put(key, SeqCheckpoint { n_tokens, data });
                            tracing::info!(
                                slot = slot.id.0,
                                n_tokens,
                                bytes,
                                "host prefix snapshot promoted"
                            );
                        }
                        (false, None) => {}
                    },
                    _ => tracing::debug!(slot = slot.id.0, "prefix checkpoint skipped"),
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
    // HiCache L2 (4th stage): keep host RAM (PrefixStore + slot chains) under
    // the global cap after new dumps are promoted.
    enforce_host_ram_cap(slots, prefix_store, HOST_RAM_CAP);
}

/// P3 §4.3: bucket a millisecond latency into a coarse histogram.
/// Bucket bounds (ms): 1, 5, 10, 25, 50, 100, 500 (index 7 = overflow).
const BUCKET_BOUNDS_MS: [u64; 8] = [1, 5, 10, 25, 50, 100, 500, u64::MAX];

fn record_bucket(bucket: &[AtomicU64; 8], ms: u64) {
    let idx = BUCKET_BOUNDS_MS.iter().position(|b| ms <= *b).unwrap_or(7);
    bucket[idx].fetch_add(1, Ordering::Relaxed);
}

fn snapshot_spec_slots(slots: &mut [Slot], engine: &Engine, plan: &crate::batch::BatchPlan) {
    for slot in slots.iter_mut() {
        let Some(job) = slot.job.as_mut() else {
            continue;
        };
        if job.drafts.is_empty()
            || job.pending.is_none()
            || !plan.logit_slots.iter().any(|id| *id == slot.id)
        {
            job.spec_ckpt_tgt = None;
            job.spec_ckpt_mtp = None;
            continue;
        }
        let seq = slot.id.0 as i32;
        job.spec_ckpt_n_past = job.n_past;
        // llama-server: n_rs_seq = need_n_rs_seq() == MTP n_max (3), and
        // skips PARTIAL_ONLY when draft.len() <= n_rs_seq. ngram-mod drafts
        // of 48–64 therefore snapshot (same as llama).
        if (job.drafts.len() as u32) <= engine.n_rs_seq() {
            job.spec_ckpt_tgt = None;
            job.spec_ckpt_mtp = None;
            continue;
        }
        let t0 = Instant::now();
        job.spec_ckpt_tgt = engine.seq_state_get(seq);
        job.spec_ckpt_mtp = None;
        job.spec_us_snapshot += t0.elapsed().as_micros() as u64;
        job.spec_ckpt_tgt_bytes = job.spec_ckpt_tgt.as_ref().map(|b| b.len()).unwrap_or(0);
        job.spec_ckpt_mtp_bytes = 0;
    }
}

fn restore_spec_checkpoint(slot: &mut Slot, engine: &Engine) {
    let seq = slot.id.0 as i32;
    let Some(job) = slot.job.as_ref() else {
        return;
    };
    let n0 = job.spec_ckpt_n_past;
    if let Some(data) = job.spec_ckpt_tgt.as_deref() {
        if engine.seq_state_set(seq, data) {
            let _ = engine.rm_seq_from(seq, n0 as i32);
        }
    }
    if let Some(data) = job.spec_ckpt_mtp.as_deref() {
        if engine.spec_state_set(seq, data) {
            let _ = engine.spec_rm_from(seq, n0 as i32);
        }
    }
}

fn replay_spec_prefix(slot: &mut Slot, engine: &Engine, drafts: &[Token], n_matched: usize) {
    let seq = slot.id.0 as i32;
    let Some(job) = slot.job.as_ref() else {
        return;
    };
    let Some(pending) = job.pending else {
        return;
    };
    let n0 = job.spec_ckpt_n_past;
    restore_spec_checkpoint(slot, engine);
    let mut items = Vec::with_capacity(1 + n_matched);
    items.push(BatchToken {
        token: pending,
        pos: n0 as i32,
        seq_id: seq,
        logits: n_matched == 0,
    });
    for (k, &d) in drafts.iter().take(n_matched).enumerate() {
        items.push(BatchToken {
            token: d,
            pos: n0 as i32 + 1 + k as i32,
            seq_id: seq,
            logits: k + 1 == n_matched,
        });
    }
    if let Err(e) = engine.decode_and_logits(&items) {
        tracing::error!(slot = slot.id.0, error = %e, "spec replay decode failed");
    } else {
        tracing::debug!(
            slot = slot.id.0,
            n_matched,
            n0,
            "spec replay after failed seq_rm"
        );
    }
}

fn sample_and_emit(
    slots: &mut [Slot],
    plan: &crate::batch::BatchPlan,
    samples: &[Token],
    engine: &Engine,
    metrics: &SchedulerMetrics,
) {
    let mut i = 0;
    while i < plan.logit_slots.len() {
        let slot_id = plan.logit_slots[i];
        let mut j = i + 1;
        while j < plan.logit_slots.len() && plan.logit_slots[j] == slot_id {
            j += 1;
        }
        let toks = &samples[i.min(samples.len())..j.min(samples.len())];
        i = j;
        if toks.is_empty() {
            continue;
        }
        let Some(slot) = slots.iter_mut().find(|s| s.id == slot_id) else {
            continue;
        };
        if slot.job.as_ref().is_none_or(|j| j.finish.is_some()) {
            continue;
        }
        emit_sampled(slot, engine, toks, Some(metrics));
    }
}

fn emit_sampled(
    slot: &mut Slot,
    engine: &Engine,
    samples: &[Token],
    metrics: Option<&SchedulerMetrics>,
) {
    let n_verify = samples.len().saturating_sub(1);
    let drafts: Vec<Token> = slot
        .job
        .as_ref()
        .map(|j| j.drafts.iter().copied().take(n_verify).collect())
        .unwrap_or_default();

    if n_verify > 0 && !drafts.is_empty() {
        verify_and_emit(slot, engine, samples, &drafts, metrics);
        return;
    }

    let Some(&token) = samples.first() else {
        return;
    };
    let Some(job) = slot.job.as_mut() else {
        return;
    };
    if job.finish.is_some() {
        return;
    }
    record_token_latency(job, metrics);
    if !push_token(slot, engine, token) {
        return;
    }
}

fn verify_and_emit(
    slot: &mut Slot,
    engine: &Engine,
    samples: &[Token],
    drafts: &[Token],
    metrics: Option<&SchedulerMetrics>,
) {
    let seq = slot.id.0 as i32;
    let n_verify = drafts.len();
    let (accepted, n_past_after) = {
        let Some(job) = slot.job.as_mut() else {
            return;
        };
        if job.finish.is_some() {
            return;
        }
        if job.spec_is_replay {
            job.spec_is_replay = false;
            job.spec_replaying = true;
        }
        let accepted = accept_drafts(samples, drafts);
        if accepted.is_empty() {
            job.drafts.clear();
            job.spec_replaying = false;
            return;
        }
        (accepted, job.n_past)
    };
    let n_matched = accepted
        .iter()
        .zip(drafts.iter())
        .take_while(|(a, d)| *a == *d)
        .count();
    let keep_pos = n_past_after.saturating_sub((n_verify - n_matched) as u32);
    let used_ckpt = slot.job.as_ref().is_some_and(|j| j.spec_ckpt_tgt.is_some());

    if n_matched < n_verify {
        if used_ckpt {
            // llama: restore, keep accepted as the next draft, re-verify next
            // iteration. Replaying in this step on a 65-token GDN batch leaves
            // a state that does not match a short sequential decode.
            restore_spec_checkpoint(slot, engine);
            engine.spec_accept(seq, n_matched as u16);
            if let Some(m) = metrics {
                m.draft_tokens_total
                    .fetch_add(n_verify as u64, Ordering::Relaxed);
            }
            if let Some(job) = slot.job.as_mut() {
                job.n_past = job.spec_ckpt_n_past;
                job.drafts = accepted;
                job.spec_is_replay = true;
                job.spec_ckpt_tgt = None;
                job.spec_ckpt_mtp = None;
                job.draft_n = job.draft_n.saturating_add(n_verify as u32);
                job.draft_verif_steps = job.draft_verif_steps.saturating_add(1);
            }
            return;
        }
        let tgt_ok = engine.rm_seq_from(seq, keep_pos as i32);
        let mtp_ok = engine.spec_rm_from(seq, keep_pos as i32);
        if !tgt_ok || !mtp_ok {
            replay_spec_prefix(slot, engine, drafts, n_matched);
        }
    }

    engine.spec_accept(seq, n_matched as u16);
    let replaying = slot.job.as_ref().is_some_and(|j| j.spec_replaying);
    if let Some(m) = metrics {
        if !replaying {
            m.draft_tokens_total
                .fetch_add(n_verify as u64, Ordering::Relaxed);
        }
        m.draft_accepted_total
            .fetch_add(n_matched as u64, Ordering::Relaxed);
    }
    tracing::debug!(
        slot = slot.id.0,
        n_draft = n_verify,
        n_matched,
        n_emit = accepted.len(),
        replaying,
        "spec accept"
    );

    if let Some(job) = slot.job.as_mut() {
        if !job.spec_replaying {
            job.draft_n = job.draft_n.saturating_add(n_verify as u32);
        }
        job.spec_replaying = false;
        job.draft_n_accepted = job.draft_n_accepted.saturating_add(n_matched as u32);
        job.draft_verif_steps = job.draft_verif_steps.saturating_add(1);
        job.n_past = keep_pos;
        job.drafts.clear();
        job.spec_ckpt_tgt = None;
        job.spec_ckpt_mtp = None;
    }

    for tok in accepted {
        let Some(job) = slot.job.as_mut() else {
            break;
        };
        if job.finish.is_some() {
            break;
        }
        record_token_latency(job, metrics);
        if !push_token(slot, engine, tok) {
            break;
        }
    }
}

fn record_token_latency(job: &mut ActiveJob, metrics: Option<&SchedulerMetrics>) {
    let now = Instant::now();
    if let Some(metrics) = metrics {
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
    } else if job.n_generated == 0 && job.generation_started_at.is_none() {
        job.generation_started_at = Some(now);
    }
    job.last_token_at = now;
}

/// Emit one sampled token. `pending` becomes this token (not yet in KV if it
/// is a bonus / correction). Returns false if the job finished without emit.
fn push_token(slot: &mut Slot, engine: &Engine, token: Token) -> bool {
    let Some(job) = slot.job.as_mut() else {
        return false;
    };
    if engine.is_eog(token) {
        let _ = job.utf8.flush();
        job.finish = Some(FinishReason::Stop);
        job.pending = None;
        job.drafts.clear();
        return false;
    }
    let bytes = match engine.token_to_piece(token) {
        Ok(b) => b,
        Err(e) => {
            let _ = job.events.send(SlotEvent::Failed(e));
            job.finish = Some(FinishReason::Stop);
            return false;
        }
    };
    let mut piece = job.utf8.push(&bytes);
    job.n_generated += 1;
    job.generated.push(token);
    job.pending = Some(token);
    slot.phase = SlotPhase::Decoding;
    let was_forcing = job.think.is_forcing();
    job.think.on_emit(&piece);
    if !was_forcing && job.think.is_forcing() {
        tracing::info!(
            slot = slot.id.0,
            request_id = job.request_id,
            think_n = job.think.think_tokens(),
            "reasoning budget exhausted; forcing </think>"
        );
    }
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
    if job.n_past + 1 > job.ctx_cap && job.finish.is_none() {
        job.finish = Some(FinishReason::Length);
    }
    let _ = job
        .events
        .send(SlotEvent::Token(GeneratedToken { token, piece }));
    true
}

struct DraftJob {
    slot_id: SlotId,
    seq: i32,
    hist: Vec<Token>,
    id_last: Token,
    n_past: i32,
    n_max: i32,
}

fn take_draft_jobs(slots: &mut [Slot], engine: &Engine) -> Vec<DraftJob> {
    let spec_on = engine.spec_enabled();
    let spec_n_max = if spec_on { engine.spec_n_max() } else { 0 };
    let mut jobs = Vec::new();
    for slot in slots.iter_mut() {
        let seq = slot.id.0 as i32;
        let Some(job) = slot.job.as_mut() else {
            continue;
        };
        if job.spec_is_replay && !job.drafts.is_empty() {
            // Keep the accepted prefix for the next decode. Do not consume
            // the flag here — verify_and_emit needs it after that decode.
            continue;
        }
        job.spec_replaying = false;
        job.drafts.clear();
        if !spec_on || job.finish.is_some() || job.think.is_forcing() {
            continue;
        }
        let Some(id_last) = job.pending else {
            continue;
        };
        if !job.spec_begun {
            engine.spec_begin(seq, &job.prompt_tokens);
            job.spec_begun = true;
        }
        let mut hist = job.prompt_tokens.clone();
        if job.generated.len() > 1 {
            hist.extend_from_slice(&job.generated[..job.generated.len() - 1]);
        }
        // llama `get_n_draft_max`: remaining ctx and remaining gen tokens,
        // not MTP n_max. ngram-mod then drafts up to its own n_max (64).
        let remain_ctx = job.ctx_cap.saturating_sub(job.n_past.saturating_add(1)) as i32;
        let remain_gen = job.max_tokens.saturating_sub(job.n_generated) as i32;
        let n_max = spec_n_max
            .min(remain_ctx.saturating_sub(1))
            .min(remain_gen.saturating_sub(1))
            .max(0);
        if n_max <= 0 {
            continue;
        }
        jobs.push(DraftJob {
            slot_id: slot.id,
            seq,
            hist,
            id_last,
            n_past: job.n_past as i32,
            n_max,
        });
    }
    jobs
}

fn fill_drafts_sync(slots: &mut [Slot], engine: &Engine) {
    let reqs = take_draft_jobs(slots, engine);
    for r in reqs {
        let t0 = Instant::now();
        let drafts = engine.spec_draft(r.seq, &r.hist, r.id_last, r.n_past, r.n_max);
        if let Some(slot) = slots.iter_mut().find(|s| s.id == r.slot_id) {
            if let Some(job) = slot.job.as_mut() {
                if job.finish.is_some() {
                    continue;
                }
                job.drafts = drafts;
                job.spec_us_draft += t0.elapsed().as_micros() as u64;
            }
        }
    }
}

/// Sync path for vision bind (one shot, not the decode loop).
fn maybe_fill_drafts(slot: &mut Slot, engine: &Engine) {
    let reqs = take_draft_jobs(std::slice::from_mut(slot), engine);
    let Some(r) = reqs.into_iter().next() else {
        return;
    };
    let t0 = Instant::now();
    let drafts = engine.spec_draft(r.seq, &r.hist, r.id_last, r.n_past, r.n_max);
    if let Some(job) = slot.job.as_mut() {
        job.drafts = drafts;
        job.spec_us_draft += t0.elapsed().as_micros() as u64;
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
        if keeps_prefix_kv(reason) {
            let gpu_n = engine.n_past_seq(id.0 as i32);
            if gpu_n != job.n_past {
                tracing::debug!(
                    slot = id.0,
                    gpu_n,
                    n_past = job.n_past,
                    "n_past vs seq_pos_max+1"
                );
            }
            // Cancel/Timeout keep whatever was actually decoded (full prompt
            // or a mid-prefill prefix). llama-server does the same: the next
            // bind's LCP continues instead of throwing 10+ minutes of work.
            if let Some(vs) = slot.prefix_cache.vision.take() {
                slot.prefix_cache
                    .remember_vision(vs, &job.generated, job.n_past);
            } else {
                slot.prefix_cache
                    .remember(&job.prompt_tokens, &job.generated, job.n_past);
            }
            // The MTP context mirrors the target 1:1 in its OWN pool of the
            // same size, but a rebind always resets it (`bind_slot` →
            // `spec_reset_seq`). Keeping the retained session's MTP cells
            // only blocks that pool: a 70k retained session left ~70k stale
            // MTP cells behind and starved the next 70k decode into
            // `failed to find a memory slot` while main decode stayed healthy
            // (#8565 recurrence, 2026-08-23 23:37). Drop them now.
            engine.spec_reset_seq(id.0 as i32);
            tracing::info!(
                slot = id.0,
                request_id = job.request_id,
                reason = reason.as_str(),
                n_past = job.n_past,
                ckpt_n = latest_ckpt_n(slot),
                "prefix kv retained"
            );
            slot.retained_at = Some(Instant::now());
        } else {
            engine.clear_seq(id.0 as i32);
            slot.prefix_cache.reset();
            clear_prefix_chain(slot);
            slot.retained_at = None;
        }
        let timings = job.timings(Instant::now());
        record_request_totals(metrics, &timings);
        log_slot_timings(id.0, job.request_id, reason.as_str(), &timings);
        log_draft_acceptance(&timings);
        if job.spec_us_snapshot > 0 || job.spec_us_draft > 0 {
            tracing::info!(
                slot = id.0,
                request_id = job.request_id,
                snapshot_ms = job.spec_us_snapshot as f64 / 1e3,
                draft_ms = job.spec_us_draft as f64 / 1e3,
                ckpt_tgt_kib = job.spec_ckpt_tgt_bytes / 1024,
                ckpt_mtp_kib = job.spec_ckpt_mtp_bytes / 1024,
                "spec host overhead"
            );
        }
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

fn log_draft_acceptance(t: &SlotTimings) {
    if t.draft_n == 0 {
        return;
    }
    let ratio = f64::from(t.draft_n_accepted) / f64::from(t.draft_n);
    let mean = if t.draft_verif_steps > 0 {
        1.0 + f64::from(t.draft_n_accepted) / f64::from(t.draft_verif_steps)
    } else {
        1.0
    };
    tracing::info!(
        "draft acceptance = {ratio:0.5} ({:5} accepted / {:5} generated), mean len = {mean:5.2}",
        t.draft_n_accepted,
        t.draft_n,
    );
}

fn emit_prompt_progress(job: &ActiveJob) {
    let _ = job.events.send(job.prompt_progress());
}

/// llama-server `print_timings_pp`: journal every 3s. SSE already got a
/// `prompt_progress` event per ubatch in `apply_plan`; this still pings when
/// the cursor is stuck behind a neighbor decode.
fn maybe_log_prefill_progress(slots: &mut [Slot]) {
    const MIN_MS: u128 = 3000;
    let decode_busy = has_pending_decode(slots);
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
        let processed = job.prefill_cursor() as u32;
        let total = job.prompt_tokens.len() as u32;
        let remaining = job.prefill_remaining() as u32;
        let progress = if total == 0 {
            1.0
        } else {
            f64::from(processed) / f64::from(total)
        };
        let pct = progress * 100.0;
        let secs = elapsed.as_secs_f64();
        // tok/s is only the work this request actually prefills (not cache_n).
        let prefilled = job.prompt_pos as u32;
        let tps = if secs > 0.0 {
            f64::from(prefilled) / secs
        } else {
            0.0
        };
        let behind_decode = remaining > 0 && processed == job.last_progress_n && decode_busy;
        let behind_msg = if behind_decode {
            " (waiting behind decode)"
        } else {
            ""
        };
        tracing::info!(
            slot = slot.id.0,
            request_id = job.request_id,
            n_tokens = processed,
            n_prompt = total,
            remaining,
            progress_pct = pct,
            tps,
            behind_decode,
            "prompt processing, n_tokens = {processed}/{total} ({pct:.1}%), remaining = {remaining}, t = {secs:.2} s / {tps:.2} tokens per second{behind_msg}"
        );
        emit_prompt_progress(job);
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

/// KV cells this slot actually occupies (active job or retained prefix).
///
/// Used for pressure accounting (`release_retained_under_pressure`) where the
/// real occupancy matters. Cap assignment must NOT include retained slots —
/// see [`slot_kv_used_for_cap`].
///
/// P8-C: an empty slot counts only GPU-resident KV — `prefix_len` (+vision
/// `n_pos`). The host `prefix_ckpts` chain is RAM, not GPU cells, so it must
/// not lift `slot_kv_used` after a watermark release or the pressure loop
/// would wrongly re-evict other idle GPU slots (#8565 defence).
fn slot_kv_used(slot: &Slot) -> u32 {
    if let Some(job) = slot.job.as_ref() {
        return job.n_past.max(job.n_prompt);
    }
    match slot.prefix_cache.vision.as_ref() {
        Some(vs) => vs.n_pos(),
        None => slot.prefix_cache.prefix_len as u32,
    }
}

/// KV cells a slot counts against the shared pool when assigning ctx caps.
///
/// A retained slot (`job.is_none()`) still holds prefix KV, but that is
/// released under pressure and must not shrink a live job's cap — otherwise a
/// solo slot could be starved below its full `S` even though nothing else is
/// actively running (#8565). So retained slots count as 0 for cap purposes.
fn slot_kv_used_for_cap(slot: &Slot) -> u32 {
    if slot.job.is_some() {
        slot_kv_used(slot)
    } else {
        0
    }
}

/// Recompute per-job `ctx_cap` from occupancy. Call on join/leave.
fn recompute_slot_caps(slots: &mut [Slot], pool: u32, single_max: u32, spec_n_max: u32) {
    let n_active = slots.iter().filter(|s| s.is_active()).count() as u32;
    if n_active == 0 {
        return;
    }
    let used: Vec<u32> = slots.iter().map(slot_kv_used_for_cap).collect();
    let total_used: u32 = used.iter().copied().sum();
    for (i, slot) in slots.iter_mut().enumerate() {
        let Some(job) = slot.job.as_mut() else {
            continue;
        };
        let policy = slot_cap(single_max, used[i], pool, n_active, spec_n_max);
        let others = total_used.saturating_sub(used[i]);
        let cap = effective_cap(policy, used[i], pool, others, spec_n_max, n_active);
        if job.ctx_cap != cap {
            tracing::info!(
                slot = slot.id.0,
                from = job.ctx_cap,
                to = cap,
                used = used[i],
                n_active,
                others_used = others,
                "slot ctx cap"
            );
        }
        job.ctx_cap = cap;
        let remaining = cap.saturating_sub(job.n_prompt).max(1);
        job.max_tokens = job.max_tokens_req.min(remaining);
    }
}

fn fail_all_active(slots: &mut [Slot], engine: &Engine, err: Error) {
    for slot in slots.iter_mut() {
        let id = slot.id;
        if let Some(job) = slot.evict() {
            engine.clear_seq(id.0 as i32);
            slot.prefix_cache.reset();
            clear_prefix_chain(slot);
            slot.retained_at = None;
            let _ = job.events.send(SlotEvent::Failed(err.clone()));
        }
    }
}

/// Retained prefix KV (#8554 affinity) is released only when total pool
/// occupancy crosses `RETAINED_WATERMARK`. A wall-clock TTL used to wipe an
/// idle session after 30s, which is shorter than a typical agent tool round,
/// so the next turn of a 30k–70k prompt full-prefilled for minutes (#8785/86).
/// The watermark still covers #8565 (two ~70k sessions filling the pool).
const RETAINED_WATERMARK: f64 = 0.85;

fn retained_release_due(total_used: u32, pool: u32) -> bool {
    pool > 0 && f64::from(total_used) > RETAINED_WATERMARK * f64::from(pool)
}

fn release_retained_under_pressure(
    slots: &mut [Slot],
    engine: &impl SeqKv,
    pool: u32,
    prefix_store: &Arc<std::sync::Mutex<PrefixStore>>,
) {
    let total_used: u32 = slots.iter().map(slot_kv_used).sum();
    if !retained_release_due(total_used, pool) {
        return;
    }
    for slot in slots.iter_mut() {
        if slot.retained_at.is_none() {
            continue; // active job or nothing retained
        }
        let id = slot.id;
        let n = slot_kv_used(slot);
        // P8-C (HiCache L2): clear only the GPU KV (L1). The host anchors
        // (`prefix_ckpts` chain + global `PrefixStore`) and the same-session
        // `prefix_cache.tokens` stay so the next bind restores instead of
        // full-prefilling. `prefix_len` is zeroed so the host dump no longer
        // counts as GPU occupancy in `slot_kv_used` (#8565 defence intact).
        engine.clear_seq(id.0 as i32);
        slot.prefix_cache.prefix_len = 0;
        if slot.prefix_cache.vision.is_some() {
            slot.prefix_cache.vision = None;
        }
        slot.retained_at = None;
        tracing::info!(
            slot = id.0,
            retained_cells = n,
            total_used,
            pool,
            chain_n = slot.prefix_ckpts.len(),
            bytes = slot
                .prefix_ckpts
                .iter()
                .map(|c| c.data.len())
                .sum::<usize>(),
            prefix_n = slot.prefix_cache.tokens.len(),
            "retained prefix kv released; host anchors kept"
        );
    }
    // HiCache L2 (4th stage): releasing GPU KV never drops host anchors, but
    // the global host-RAM cap is still enforced on the retained chains.
    enforce_host_ram_cap(slots, prefix_store, HOST_RAM_CAP);
}

fn keeps_prefix_kv(reason: FinishReason) -> bool {
    // Decode errors still clear: KV may be inconsistent. Cancel/Timeout are
    // clean stops — the resident prefix is valid and the next identical
    // (or continuation) prompt must be able to reuse it.
    matches!(
        reason,
        FinishReason::Stop | FinishReason::Length | FinishReason::Cancelled | FinishReason::Timeout
    )
}

/// P8-A chain bounds: per-slot anchor count and total host bytes.
/// DSV4 PARTIAL ≈ 18 MiB/anchor, so 2 GiB rarely binds; the cap guards Qwen FA
/// length-proportional dumps. On overflow keep the newest prefill-end and any
/// 8k–16k store stub (tool-head candidate). Do not drop those just to meet
/// the byte cap.
const CHAIN_MAX_ANCHORS: usize = 8;
const CHAIN_MAX_BYTES: usize = 2 * 1024 * 1024 * 1024; // 2 GiB

/// Global host-RAM cap (HiCache L2, 4th stage): `PrefixStore.total_bytes`
/// plus all slot `prefix_ckpts` dumps must stay under this many bytes.
/// Kept at 2 GiB to match `CHAIN_MAX_BYTES` / `PrefixStore::DEFAULT_MAX_BYTES`.
const HOST_RAM_CAP: usize = 2 * 1024 * 1024 * 1024; // 2 GiB

/// Store-stub window: last ubatch-aligned dumps still likely inside the
/// shared tool/system head (production ~12545, ubatch boundary 12288).
const STORE_STUB_MIN: u32 = 8_192;
const STORE_STUB_MAX: u32 = 16_384;
const STORE_STUB_STRIDE: u32 = 2_048;

fn is_store_stub_len(n_tokens: u32) -> bool {
    (STORE_STUB_MIN..=STORE_STUB_MAX).contains(&n_tokens)
}

fn is_host_stub_boundary(n_tokens: u32, n_ubatch: u32) -> bool {
    let n_ubatch = n_ubatch.max(1);
    n_tokens > 0
        && is_store_stub_len(n_tokens)
        && n_tokens.is_multiple_of(n_ubatch)
        && n_tokens.is_multiple_of(STORE_STUB_STRIDE)
}

fn is_last_host_stub(n_tokens: u32) -> bool {
    is_store_stub_len(n_tokens) && n_tokens + STORE_STUB_STRIDE > STORE_STUB_MAX
}

/// Push a fresh snapshot onto the slot's anchor chain (ascending by `n_tokens`).
/// Same `n_tokens` replaces; a longer one appends. Enforces chain bounds.
fn push_prefix_ckpt(slot: &mut Slot, n_tokens: u32, data: Vec<u8>) {
    let bytes = data.len();
    let ckpt = crate::slot::SeqCheckpoint { n_tokens, data };
    // Same `n_tokens` → replace in place (newer dump wins).
    if let Some(existing) = slot
        .prefix_ckpts
        .iter_mut()
        .find(|c| c.n_tokens == n_tokens)
    {
        *existing = ckpt;
    } else {
        slot.prefix_ckpts.push(ckpt);
        slot.prefix_ckpts.sort_by_key(|c| c.n_tokens);
    }
    trim_prefix_chain(slot, CHAIN_MAX_ANCHORS, CHAIN_MAX_BYTES);
    tracing::info!(
        slot = slot.id.0,
        n_tokens,
        bytes,
        chain_n = slot.prefix_ckpts.len(),
        "prefix checkpoint saved"
    );
}

/// Drop unprotected middles. Newest and 8k–16k stubs stay even if the byte
/// cap is still exceeded (a single Qwen FA 70k dump is ~4 GiB).
///
/// Eviction direction is intentional: the chain keeps the newest + store-stub
/// anchors and drops the shortest unprotected middle, whereas `PrefixStore`
/// drops the *longest* dump (`pick_longest_victim`) so a short tool-head
/// prefix is never evicted while a longer dump exists.
fn trim_prefix_chain(slot: &mut Slot, max_anchors: usize, max_bytes: usize) {
    loop {
        let n = slot.prefix_ckpts.len();
        if n == 0 {
            return;
        }
        let bytes: usize = slot.prefix_ckpts.iter().map(|c| c.data.len()).sum();
        if n <= max_anchors && bytes <= max_bytes {
            return;
        }
        let newest_n = slot
            .prefix_ckpts
            .iter()
            .map(|c| c.n_tokens)
            .max()
            .unwrap_or(0);
        let drop = slot
            .prefix_ckpts
            .iter()
            .enumerate()
            .find(|(_, c)| c.n_tokens != newest_n && !is_store_stub_len(c.n_tokens))
            .map(|(i, _)| i);
        let Some(drop) = drop else {
            return;
        };
        let removed = slot.prefix_ckpts.remove(drop);
        tracing::info!(
            slot = slot.id.0,
            dropped_n = removed.n_tokens,
            chain_n = slot.prefix_ckpts.len(),
            bytes = slot
                .prefix_ckpts
                .iter()
                .map(|c| c.data.len())
                .sum::<usize>(),
            "prefix chain trimmed"
        );
    }
}

/// Sum of all slot `prefix_ckpts` dump bytes (host RAM owned by the chains).
fn host_chain_bytes(slots: &[Slot]) -> usize {
    slots
        .iter()
        .map(|s| s.prefix_ckpts.iter().map(|c| c.data.len()).sum::<usize>())
        .sum()
}

/// Enforce the global host-RAM cap (HiCache L2, 4th stage).
///
/// `PrefixStore.total_bytes` + all slot `prefix_ckpts` bytes must stay under
/// `HOST_RAM_CAP`. Drop the longest `PrefixStore` dump first
/// (`evict_global_over_cap` / `pick_longest_victim`), then the shortest
/// unprotected middle of each slot chain (`trim_prefix_chain`). The shortest
/// tool head (8k–16k store stub) and each slot's newest anchor are kept last.
fn enforce_host_ram_cap(
    slots: &mut [Slot],
    prefix_store: &Arc<std::sync::Mutex<PrefixStore>>,
    cap: usize,
) {
    let mut store = prefix_store.lock().unwrap();
    loop {
        let chain_bytes = host_chain_bytes(slots);
        if store.total_bytes + chain_bytes <= cap {
            break;
        }
        // Stage 1: drop the longest PrefixStore dump.
        let before = store.total_bytes;
        store.evict_global_over_cap(chain_bytes, cap);
        if store.total_bytes < before {
            continue;
        }
        // Stage 2: trim each slot chain's unprotected middle (keeps newest +
        // store stubs). `max_bytes = 0` forces the middle-only eviction rule.
        let mut dropped = false;
        for slot in slots.iter_mut() {
            let before_n = slot.prefix_ckpts.len();
            trim_prefix_chain(slot, CHAIN_MAX_ANCHORS, 0);
            dropped |= slot.prefix_ckpts.len() < before_n;
        }
        if !dropped {
            // Only newest + store stubs remain everywhere; nothing left to
            // evict without dropping a protected anchor.
            break;
        }
    }
}

/// After bind, drop anchors the new prompt cannot restore. Stops
/// `latest_ckpt_n` / affinity_bar from reporting a previous session's length.
fn prune_chain_beyond_reuse(slot: &mut Slot, reuse_len: usize) {
    let max_n = reuse_len as u32 + 1;
    let before = slot.prefix_ckpts.len();
    slot.prefix_ckpts.retain(|c| c.n_tokens <= max_n);
    if slot.prefix_ckpts.len() != before {
        tracing::info!(
            slot = slot.id.0,
            reuse_len,
            chain_n = slot.prefix_ckpts.len(),
            latest_ckpt_n = latest_ckpt_n(slot),
            "prefix chain pruned past reuse_len"
        );
    }
}

fn capture_prefix_checkpoint(slot: &mut Slot, engine: &Engine) {
    let n_tokens = slot.job.as_ref().map(|j| j.n_past).unwrap_or(0);
    if n_tokens == 0 {
        return;
    }
    match engine.seq_state_get(slot.id.0 as i32) {
        Some(data) if !data.is_empty() => {
            push_prefix_ckpt(slot, n_tokens, data);
        }
        _ => tracing::debug!(slot = slot.id.0, "prefix checkpoint skipped"),
    }
}

/// Latest anchor's `n_tokens` (0 if none) — used for same-session affinity.
/// Affinity bar keeps using the newest anchor so a same-session continuation
/// still binds; only `settle_prefix_kv` chooses from the chain.
fn latest_ckpt_n(slot: &Slot) -> u32 {
    slot.prefix_ckpts
        .iter()
        .map(|c| c.n_tokens)
        .max()
        .unwrap_or(0)
}

/// Longest anchor with `n_tokens <= reuse_len + 1` (one-token slack covers
/// ` thinking` vs ` response`). Returns `(index, n_tokens)` in the chain,
/// or `None` when nothing fits.
fn best_chain_anchor(slot: &Slot, reuse_len: usize) -> Option<(usize, u32)> {
    slot.prefix_ckpts
        .iter()
        .enumerate()
        .filter(|(_, c)| c.n_tokens != 0 && c.n_tokens <= reuse_len as u32 + 1)
        .max_by_key(|(_, c)| c.n_tokens)
        .map(|(i, c)| (i, c.n_tokens))
}

/// Clear the chain (used on full prefill / clear paths).
fn clear_prefix_chain(slot: &mut Slot) {
    slot.prefix_ckpts.clear();
}

/// Trim or restore so GPU KV covers exactly `reuse_len` cells (or 0 = clear).
///
/// P8-A: bind from the longest anchor in the slot's chain with
/// `n_tokens <= reuse_len + 1`, so a prompt trimmed in the middle (think body
/// removed, tool result removed, message trimmed) resumes at the closest
/// earlier semantic boundary instead of full-prefilling. Never discard a
/// shorter anchor just because the newest one is too long.
///
/// P8-B: when local reuse is 0 (empty slot) or no chain anchor fits, look up
/// the global host `PrefixStore` with the **new** prompt. `reuse_len == 0`
/// still searches `prompt.len() - 1` so a 12k tool head can restore.
///
/// After bind, drop `n_tokens > reuse_len + 1` so `latest_ckpt_n` cannot lift
/// affinity_bar to a previous session's length.
fn settle_prefix_kv(
    slot: &mut Slot,
    engine: &impl SeqKv,
    reuse_len: usize,
    gpu_n: u32,
    prefix_store: &Arc<std::sync::Mutex<PrefixStore>>,
    prompt: &[Token],
) -> usize {
    let seq = slot.id.0 as i32;
    if reuse_len == 0 {
        engine.clear_seq(seq);
        if let Some(kept) =
            restore_host_snapshot(slot, engine, reuse_len, gpu_n, prefix_store, prompt)
        {
            return kept;
        }
        prune_chain_beyond_reuse(slot, 0);
        return 0;
    }
    if trim_seq_to(engine, seq, reuse_len) {
        prune_chain_beyond_reuse(slot, reuse_len);
        return reuse_len;
    }
    let gpu_after = engine.n_past_seq(seq);

    // P8-A: pick the longest anchor that still fits the LCP.
    if let Some((_, ckpt_n)) = best_chain_anchor(slot, reuse_len) {
        let ckpt = slot
            .prefix_ckpts
            .iter()
            .find(|c| c.n_tokens == ckpt_n)
            .unwrap();
        if !engine.seq_state_set(seq, &ckpt.data) {
            tracing::warn!(slot = slot.id.0, ckpt_n, "prefix chain restore failed");
            engine.clear_seq(seq);
            slot.prefix_cache.reset();
            prune_chain_beyond_reuse(slot, 0);
            return 0;
        }
        // PARTIAL_ONLY skips ISWA non-SWA base. seq_rm(ckpt_n, -1) is past SWA
        // pos_max so DSV4 drops leftover base without n_rs_seq; prefix stays.
        if !engine.rm_seq_from(seq, ckpt_n as i32) {
            tracing::warn!(
                slot = slot.id.0,
                ckpt_n,
                "prefix restore leftover sweep failed"
            );
        }
        let kept = reuse_len.min(ckpt_n as usize);
        if trim_seq_to(engine, seq, kept) {
            tracing::info!(
                slot = slot.id.0,
                reuse_len = kept,
                ckpt_n,
                gpu_n,
                gpu_after = engine.n_past_seq(seq),
                chain_n = slot.prefix_ckpts.len(),
                bytes = slot
                    .prefix_ckpts
                    .iter()
                    .map(|c| c.data.len())
                    .sum::<usize>(),
                "prefix restored from checkpoint"
            );
            prune_chain_beyond_reuse(slot, kept);
            return kept;
        }
        tracing::warn!(
            slot = slot.id.0,
            reuse_len,
            ckpt_n,
            gpu_after = engine.n_past_seq(seq),
            "prefix chain restore still longer than LCP; full prefill"
        );
        engine.clear_seq(seq);
        slot.prefix_cache.reset();
        prune_chain_beyond_reuse(slot, 0);
        return 0;
    }

    if let Some(kept) = restore_host_snapshot(slot, engine, reuse_len, gpu_n, prefix_store, prompt)
    {
        return kept;
    }
    tracing::warn!(
        slot = slot.id.0,
        reuse_len,
        latest_ckpt_n = latest_ckpt_n(slot),
        gpu_n,
        gpu_after,
        "prefix chain not usable; full prefill"
    );
    engine.clear_seq(seq);
    slot.prefix_cache.reset();
    prune_chain_beyond_reuse(slot, 0);
    0
}

/// Restore the longest `PrefixStore` dump that matches `prompt`. Uses the new
/// job tokens, not `slot.prefix_cache`. Returns `None` on miss or GPU failure.
fn restore_host_snapshot(
    slot: &mut Slot,
    engine: &impl SeqKv,
    reuse_len: usize,
    gpu_n: u32,
    prefix_store: &Arc<std::sync::Mutex<PrefixStore>>,
    prompt: &[Token],
) -> Option<usize> {
    let seq = slot.id.0 as i32;
    let search_len = host_search_len(prompt.len(), reuse_len);
    if search_len == 0 {
        return None;
    }
    let host = prefix_store
        .lock()
        .unwrap()
        .find_best_for_bind(prompt, reuse_len);
    let host_ckpt = host?;
    tracing::info!(
        slot = slot.id.0,
        reuse_len,
        search_len,
        host_n = host_ckpt.n_tokens,
        gpu_n,
        gpu_after = engine.n_past_seq(seq),
        "host snapshot restore"
    );
    engine.clear_seq(seq);
    if !engine.seq_state_set(seq, &host_ckpt.data) {
        tracing::warn!(
            slot = slot.id.0,
            reuse_len,
            host_n = host_ckpt.n_tokens,
            "host snapshot restore failed; full prefill"
        );
        engine.clear_seq(seq);
        prune_chain_beyond_reuse(slot, 0);
        return None;
    }
    if !engine.rm_seq_from(seq, host_ckpt.n_tokens as i32) {
        tracing::warn!(
            slot = slot.id.0,
            host_n = host_ckpt.n_tokens,
            "host snapshot leftover sweep failed"
        );
    }
    let kept = search_len.min(host_ckpt.n_tokens as usize);
    if trim_seq_to(engine, seq, kept) {
        tracing::info!(
            slot = slot.id.0,
            reuse_len = kept,
            host_n = host_ckpt.n_tokens,
            gpu_after = engine.n_past_seq(seq),
            "host snapshot restored"
        );
        prune_chain_beyond_reuse(slot, kept);
        return Some(kept);
    }
    tracing::warn!(
        slot = slot.id.0,
        reuse_len,
        host_n = host_ckpt.n_tokens,
        "host snapshot restore still longer than LCP; full prefill"
    );
    engine.clear_seq(seq);
    prune_chain_beyond_reuse(slot, 0);
    None
}

fn trim_seq_to(engine: &impl SeqKv, seq: i32, n: usize) -> bool {
    let gpu = engine.n_past_seq(seq);
    if gpu < n as u32 {
        return false;
    }
    // gpu == n is not "already done". After a PARTIAL restore, SWA pos_max+1
    // equals n while orphaned non-SWA base cells at pos >= n still occupy
    // slots. seq_rm(n, -1) is a no-op when nothing is there.
    engine.rm_seq_from(seq, n as i32) && engine.n_past_seq(seq) == n as u32
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
        assert_eq!(b.mixed_prefill_max, 0);
    }

    #[test]
    fn explicit_budget_is_clamped_to_n_batch() {
        let b = resolve_budget(
            IterationBudget {
                prefill_max: 99999,
                decode_max: 4,
                mixed_prefill_max: 99999,
                prefill_yield_every: 16,
                prefill_yield_max: 99999,
            },
            5800,
            1024,
            1,
        );
        assert_eq!(b.prefill_max, 5800);
        assert_eq!(b.decode_max, 4);
        assert_eq!(b.mixed_prefill_max, 5800);
        assert_eq!(b.prefill_yield_every, 16);
        assert_eq!(b.prefill_yield_max, 5800);
    }

    #[test]
    fn stop_length_cancel_and_timeout_keep_kv() {
        assert!(keeps_prefix_kv(FinishReason::Stop));
        assert!(keeps_prefix_kv(FinishReason::Length));
        assert!(keeps_prefix_kv(FinishReason::Cancelled));
        assert!(keeps_prefix_kv(FinishReason::Timeout));
    }

    #[test]
    fn slot_kv_used_is_max_of_n_past_and_n_prompt() {
        let mut slot = Slot::new(SlotId(0));
        let mut job = ActiveJob::for_test(vec![1, 2, 3, 4, 5]);
        job.n_past = 2;
        slot.occupy(job);
        assert_eq!(slot_kv_used(&slot), 5);
        if let Some(j) = slot.job.as_mut() {
            j.n_past = 9;
        }
        assert_eq!(slot_kv_used(&slot), 9);
    }

    #[test]
    fn empty_slot_kv_used_counts_gpu_prefix_only() {
        let mut slot = Slot::new(SlotId(0));
        slot.prefix_cache.prefix_len = 12_000;
        assert_eq!(slot_kv_used(&slot), 12_000);
        // P8-C: host `prefix_ckpts` is RAM, not GPU cells — a 20k dump must
        // not lift empty-slot occupancy (else pressure wrongly re-evicts).
        slot.prefix_ckpts = vec![crate::slot::SeqCheckpoint {
            n_tokens: 20_000,
            data: vec![0xAA; 20_000],
        }];
        assert_eq!(slot_kv_used(&slot), 12_000, "host dump is not GPU occupancy");
    }

    /// #8565 regression: a retained slot (job None) must NOT shrink a live
    /// slot's cap. `slot_kv_used` still reports the retained prefix for
    /// pressure accounting, but cap assignment uses `slot_kv_used_for_cap`.
    #[test]
    fn retained_slot_counts_zero_for_cap() {
        let mut retained = Slot::new(SlotId(0));
        retained.prefix_cache.prefix_len = 69_638;
        // pressure accounting still sees the real occupancy
        assert_eq!(slot_kv_used(&retained), 69_638);
        // cap assignment ignores it
        assert_eq!(slot_kv_used_for_cap(&retained), 0);
    }

    #[test]
    fn retained_slot_does_not_shrink_solo_cap() {
        const T: u32 = 80_000;
        const S: u32 = 60_000;
        let mut slots = vec![Slot::new(SlotId(0)), Slot::new(SlotId(1))];

        // slot1 retains a huge prefix (job finished), slot2 is the live solo job.
        slots[0].prefix_cache.prefix_len = 60_000;
        let mut live = ActiveJob::for_test(vec![1]);
        live.n_prompt = 0;
        live.n_past = 0;
        live.max_tokens_req = 100_000;
        slots[1].occupy(live);
        recompute_slot_caps(&mut slots, T, S, 0);

        // Without the fix, others_used=60000 would cap slot2 at 20000.
        let cap = slots[1].job.as_ref().unwrap().ctx_cap;
        assert_eq!(cap, S, "retained prefix must not shrink the live solo cap");
    }

    #[test]
    fn recompute_caps_matches_work_order_examples() {
        const T: u32 = 80_000;
        const S: u32 = 60_000;
        let mut slots = vec![Slot::new(SlotId(0)), Slot::new(SlotId(1))];

        let mut solo = ActiveJob::for_test(vec![1]);
        solo.n_prompt = 0;
        solo.n_past = 0;
        solo.max_tokens_req = 100_000;
        slots[0].occupy(solo);
        recompute_slot_caps(&mut slots, T, S, 0);
        assert_eq!(slots[0].job.as_ref().unwrap().ctx_cap, 60_000);

        slots[0].job.as_mut().unwrap().n_past = 35_000;
        slots[0].job.as_mut().unwrap().n_prompt = 1_000;
        let mut second = ActiveJob::for_test(vec![1]);
        second.n_prompt = 0;
        second.n_past = 0;
        second.max_tokens_req = 100_000;
        slots[1].occupy(second);
        recompute_slot_caps(&mut slots, T, S, 0);
        assert_eq!(slots[0].job.as_ref().unwrap().ctx_cap, 40_000);
        assert_eq!(slots[1].job.as_ref().unwrap().ctx_cap, 40_000);

        slots[0].job.as_mut().unwrap().n_past = 60_000;
        slots[0].job.as_mut().unwrap().n_prompt = 1_000;
        recompute_slot_caps(&mut slots, T, S, 0);
        assert_eq!(slots[0].job.as_ref().unwrap().ctx_cap, 60_000);
        assert_eq!(slots[1].job.as_ref().unwrap().ctx_cap, 20_000);

        let _ = slots[1].evict();
        recompute_slot_caps(&mut slots, T, S, 0);
        assert_eq!(slots[0].job.as_ref().unwrap().ctx_cap, 60_000);
    }

    #[test]
    fn retained_release_due_on_watermark_only() {
        let pool = 140_032;
        // Two ~44k sessions (typical agent pair) stay under 85% so each
        // keeps prefix KV across tool rounds that last minutes, not 30s.
        assert!(!retained_release_due(88_000, pool));
        assert!(!retained_release_due(60_000, pool));
        // Past the 85% watermark: release (two ~70k sessions, #8565).
        assert!(retained_release_due(125_000, pool));
        // Exactly at the watermark boundary (not strictly above): keep.
        let at_mark = (RETAINED_WATERMARK * f64::from(pool)) as u32;
        assert!(!retained_release_due(at_mark, pool));
        assert!(retained_release_due(at_mark + 1, pool));
        assert!(!retained_release_due(0, 0));
    }

    // ── P8-A semantic anchor chain ────────────────────────────────────────
    fn chain_slot(anchors: &[u32]) -> Slot {
        let mut slot = Slot::new(SlotId(0));
        for &n in anchors {
            push_prefix_ckpt(&mut slot, n, vec![0u8; n as usize]);
        }
        slot
    }

    #[test]
    fn best_chain_anchor_returns_longest_fitting_anchor() {
        let slot = chain_slot(&[100, 200, 300]);

        // reuse_len=200 → anchor 200 fits (200 <= 201) and is the longest.
        assert_eq!(best_chain_anchor(&slot, 200).map(|(_, n)| n), Some(200));
        // reuse_len=150 → 200/300 too long; only 100 fits.
        assert_eq!(best_chain_anchor(&slot, 150).map(|(_, n)| n), Some(100));
        // reuse_len=300 → 300 fits (300 <= 301).
        assert_eq!(best_chain_anchor(&slot, 300).map(|(_, n)| n), Some(300));
        // reuse_len below (shortest - 1) → nothing fits.
        // (shortest=100, +1 slack → need reuse_len <= 98 for 100 to not fit)
        assert_eq!(best_chain_anchor(&slot, 98).map(|(_, n)| n), None);
        assert_eq!(best_chain_anchor(&slot, 99).map(|(_, n)| n), Some(100));
    }

    #[test]
    fn best_chain_anchor_one_token_slack() {
        let slot = chain_slot(&[100, 200]);
        // n_tokens=200 <= reuse_len+1=201 → still fits at reuse_len=200.
        assert_eq!(best_chain_anchor(&slot, 200).map(|(_, n)| n), Some(200));
        // reuse_len=199 → 200 (<=200) still fits due to +1 slack.
        assert_eq!(best_chain_anchor(&slot, 199).map(|(_, n)| n), Some(200));
        // reuse_len=198 → 200 (>199) does not fit; 100 does.
        assert_eq!(best_chain_anchor(&slot, 198).map(|(_, n)| n), Some(100));
    }

    #[test]
    fn latest_ckpt_n_is_newest_anchor() {
        let slot = chain_slot(&[100, 200, 300]);
        assert_eq!(latest_ckpt_n(&slot), 300);
        assert_eq!(latest_ckpt_n(&Slot::new(SlotId(0))), 0);
    }

    #[test]
    fn push_prefix_ckpt_replaces_and_keeps_ascending() {
        let mut slot = Slot::new(SlotId(0));
        push_prefix_ckpt(&mut slot, 100, vec![1u8; 10]);
        push_prefix_ckpt(&mut slot, 300, vec![2u8; 20]);
        // Same n_tokens replaces in place (newer dump wins), no duplicate.
        push_prefix_ckpt(&mut slot, 100, vec![3u8; 30]);
        let ns: Vec<u32> = slot.prefix_ckpts.iter().map(|c| c.n_tokens).collect();
        assert_eq!(ns, vec![100, 300]);
        // The replaced dump is the newer one.
        assert_eq!(slot.prefix_ckpts[0].data.len(), 30);
    }

    #[test]
    fn chain_overflow_keeps_12288_and_prefill_end() {
        let mut slot = Slot::new(SlotId(0));
        // Simulate a 37k prefill that used to snapshot every 2048-token ubatch.
        for n in (2048..=36_864).step_by(2048) {
            push_prefix_ckpt(&mut slot, n as u32, vec![0u8; 8]);
        }
        push_prefix_ckpt(&mut slot, 37_233, vec![0u8; 8]);
        assert!(slot.prefix_ckpts.len() <= CHAIN_MAX_ANCHORS);
        assert!(
            slot.prefix_ckpts.iter().any(|c| c.n_tokens == 12_288),
            "tool-head ubatch 12288 must survive overflow: {:?}",
            slot.prefix_ckpts
                .iter()
                .map(|c| c.n_tokens)
                .collect::<Vec<_>>()
        );
        assert!(
            slot.prefix_ckpts.iter().any(|c| c.n_tokens == 37_233),
            "prefill-end must survive overflow"
        );
        let ns: Vec<u32> = slot.prefix_ckpts.iter().map(|c| c.n_tokens).collect();
        assert!(ns.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(latest_ckpt_n(&slot), 37_233);
    }

    #[test]
    fn chain_byte_cap_keeps_protected_when_over() {
        let mut slot = Slot::new(SlotId(0));
        slot.prefix_ckpts = vec![
            crate::slot::SeqCheckpoint {
                n_tokens: 12_288,
                data: vec![0u8; 50],
            },
            crate::slot::SeqCheckpoint {
                n_tokens: 40_000,
                data: vec![0u8; 50],
            },
        ];
        // Only the stub and the newest remain, and together they exceed the
        // tiny cap. They must not be dropped.
        trim_prefix_chain(&mut slot, 8, 10);
        assert_eq!(slot.prefix_ckpts.len(), 2);
        assert!(slot.prefix_ckpts.iter().any(|c| c.n_tokens == 12_288));
        assert!(slot.prefix_ckpts.iter().any(|c| c.n_tokens == 40_000));
        assert_eq!(latest_ckpt_n(&slot), 40_000);
    }

    #[test]
    fn prune_chain_drops_too_long_after_bind() {
        let mut slot = chain_slot(&[12_288, 16_384, 37_000]);
        prune_chain_beyond_reuse(&mut slot, 12_288);
        let ns: Vec<u32> = slot.prefix_ckpts.iter().map(|c| c.n_tokens).collect();
        assert_eq!(ns, vec![12_288]);
        assert_eq!(latest_ckpt_n(&slot), 12_288);
    }

    #[test]
    fn host_stub_boundary_is_stride_in_window() {
        assert!(is_host_stub_boundary(12_288, 2048));
        assert!(is_host_stub_boundary(8_192, 2048));
        assert!(is_host_stub_boundary(16_384, 2048));
        assert!(!is_host_stub_boundary(2_048, 2048));
        assert!(!is_host_stub_boundary(18_432, 2048));
        assert!(!is_host_stub_boundary(12_545, 2048));
        assert!(is_last_host_stub(16_384));
        assert!(!is_last_host_stub(12_288));
        // CUDA n_ubatch=512 still hits the 2048 stride.
        assert!(is_host_stub_boundary(12_288, 512));
        assert!(!is_host_stub_boundary(8_704, 512));
    }

    #[test]
    fn snapshot_key_from_job_prompt_matches_n_tokens() {
        let prompt: Vec<Token> = (0..20_000).map(|i| i as Token).collect();
        let key = snapshot_key(&prompt, 12_288).unwrap();
        assert_eq!(key.len(), 12_288);
        assert_eq!(&key[..], &prompt[..12_288]);
    }

    // ── P8 bind-path harness (issue #94): no GGUF / Engine ────────────────
    //
    // `empty_slot_store_hit_reuses_head` only calls `find_best_for_bind`.
    // The original #93 bug (lookup/promote via empty `prefix_cache.tokens`)
    // would still pass that unit test. These go through bind_slot →
    // settle_prefix_kv → restore_host_snapshot and apply_plan.

    struct FakeSeqKv {
        inner: std::sync::Mutex<FakeKvState>,
    }

    struct FakeKvState {
        n_past: std::collections::HashMap<i32, u32>,
        state: std::collections::HashMap<i32, Vec<u8>>,
        n_ubatch: u32,
    }

    impl FakeSeqKv {
        fn new() -> Self {
            Self {
                inner: std::sync::Mutex::new(FakeKvState {
                    n_past: std::collections::HashMap::new(),
                    state: std::collections::HashMap::new(),
                    n_ubatch: 2048,
                }),
            }
        }

        fn with_n_ubatch(n_ubatch: u32) -> Self {
            let kv = Self::new();
            kv.inner.lock().unwrap().n_ubatch = n_ubatch;
            kv
        }

        fn seed_state(&self, seq: i32, data: Vec<u8>) {
            let n = data.len() as u32;
            let mut g = self.inner.lock().unwrap();
            g.n_past.insert(seq, n);
            g.state.insert(seq, data);
        }
    }

    impl SeqKv for FakeSeqKv {
        fn spec_reset_seq(&self, _seq: i32) {}

        fn n_past_seq(&self, seq: i32) -> u32 {
            self.inner
                .lock()
                .unwrap()
                .n_past
                .get(&seq)
                .copied()
                .unwrap_or(0)
        }

        fn clear_seq(&self, seq: i32) {
            let mut g = self.inner.lock().unwrap();
            g.n_past.insert(seq, 0);
            g.state.remove(&seq);
        }

        fn seq_state_set(&self, seq: i32, data: &[u8]) -> bool {
            if data.is_empty() {
                return false;
            }
            let mut g = self.inner.lock().unwrap();
            g.state.insert(seq, data.to_vec());
            // Tests size dumps as `n_tokens` bytes so n_past matches the
            // restored checkpoint (production dumps are opaque).
            g.n_past.insert(seq, data.len() as u32);
            true
        }

        fn rm_seq_from(&self, seq: i32, p0: i32) -> bool {
            let mut g = self.inner.lock().unwrap();
            let cur = g.n_past.get(&seq).copied().unwrap_or(0);
            if p0 < 0 {
                g.n_past.insert(seq, 0);
            } else if (p0 as u32) <= cur {
                g.n_past.insert(seq, p0 as u32);
            }
            true
        }

        fn seq_state_get(&self, seq: i32) -> Option<Vec<u8>> {
            self.inner.lock().unwrap().state.get(&seq).cloned()
        }

        fn n_ubatch(&self) -> u32 {
            self.inner.lock().unwrap().n_ubatch
        }

        fn encode(&self, _text: &str) -> crate::error::Result<Vec<Token>> {
            Err(Error::Tokenize(
                "FakeSeqKv: tests must set job.prompt_tokens".into(),
            ))
        }

        fn tokenize_special(&self, _text: &str) -> crate::error::Result<Vec<Token>> {
            Ok(Vec::new())
        }
    }

    fn tool_head_and_prompt() -> (Vec<Token>, Vec<Token>) {
        let head: Vec<Token> = (0..12_288).map(|i| (i % 7) as Token).collect();
        let mut prompt = head.clone();
        prompt.extend((0..1_000).map(|i| (100 + i % 3) as Token));
        (head, prompt)
    }

    fn ckpt_n(n: u32, fill: u8) -> SeqCheckpoint {
        SeqCheckpoint {
            n_tokens: n,
            data: vec![fill; n as usize],
        }
    }

    fn job_with_tokens(tokens: Vec<Token>) -> Job {
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut job = Job::new(
            "unused".into(),
            GenerateParams::default(),
            CancellationToken::new(),
            tx,
        );
        job.prompt_tokens = Some(tokens);
        job
    }

    #[test]
    fn empty_slot_bind_store_hit_reuses_head() {
        // Arrange: empty slot, empty prefix_cache.tokens (the #93 lookup
        // used this, so a store hit was impossible). Store has the 12288
        // tool-head dump; the new job prompt starts with that head.
        let mut slot = Slot::new(SlotId(0));
        assert!(slot.prefix_cache.tokens.is_empty());
        let (head, prompt) = tool_head_and_prompt();
        let store = Arc::new(std::sync::Mutex::new(PrefixStore::new()));
        store.lock().unwrap().put(head, ckpt_n(12_288, 0xAB));
        let kv = FakeSeqKv::new();
        let n_prompt = prompt.len() as u32;

        // Act: full text bind path (not find_best_for_bind alone).
        let bound = bind_slot(&mut slot, job_with_tokens(prompt), &kv, 40_000, &store);

        // Assert: restore_host_snapshot kept the tool-head length.
        assert!(bound, "empty-slot bind with store hit must succeed");
        let job = slot.job.as_ref().expect("slot occupied");
        assert_eq!(
            job.prompt_offset, 12_288,
            "reused == 12288 (tool-head ubatch boundary)"
        );
        assert_eq!(job.n_past, 12_288);
        assert_eq!(job.n_prompt, n_prompt);
        assert_eq!(kv.n_past_seq(0), 12_288);
        // Bind must not have written the prompt into prefix_cache; occupy
        // leaves it empty until remember() at finish.
        assert!(
            slot.prefix_cache.tokens.is_empty(),
            "prefix_cache.tokens stays empty during the job; store lookup used the new prompt"
        );
    }

    // ── P8-C (issue #91): watermark clears GPU KV, keeps host anchors ──────

    #[test]
    fn watermark_release_clears_gpu_keeps_host_anchors() {
        // P8-C: under pressure, release only GPU KV (L1). Host anchors
        // (`prefix_ckpts` chain) and same-session `prefix_cache.tokens` stay
        // so the next bind restores instead of full-prefilling.
        let mut slot = Slot::new(SlotId(0));
        slot.prefix_cache.tokens = vec![1, 2, 3, 4, 5]; // session identity
        slot.prefix_cache.prefix_len = 60_000;
        slot.prefix_ckpts = vec![ckpt_n(60_000, 0xAA), ckpt_n(12_288, 0xBB)];
        slot.retained_at = Some(std::time::Instant::now());
        let kv = FakeSeqKv::new();
        kv.seed_state(0, vec![0xCD; 60_000]);

        let store = Arc::new(std::sync::Mutex::new(PrefixStore::new()));
        let mut slots = vec![slot];
        // tiny pool: 60k (via prefix_len) crosses the watermark
        release_retained_under_pressure(&mut slots, &kv, 60_000, &store);

        let s = &slots[0];
        assert_eq!(kv.n_past_seq(0), 0, "GPU KV cleared");
        assert_eq!(s.prefix_ckpts.len(), 2, "host chain kept");
        assert_eq!(
            s.prefix_ckpts.iter().map(|c| c.data.len()).sum::<usize>(),
            60_000 + 12_288,
            "chain bytes kept"
        );
        assert_eq!(s.prefix_cache.tokens.len(), 5, "session tokens kept");
        assert_eq!(s.prefix_cache.prefix_len, 0, "prefix_len zeroed");
        assert!(s.retained_at.is_none());
        assert_eq!(slot_kv_used(s), 0, "host dump is not GPU occupancy");
    }

    #[test]
    fn same_session_bind_restores_chain_after_watermark() {
        // Watermark release, then the same session returns → LCP>0 via
        // `tokens`, GPU empty → best_chain_anchor + seq_state_set restores.
        let shared: Vec<Token> = (0..60_000).map(|i| (i % 7) as Token).collect();
        let mut slot = Slot::new(SlotId(0));
        slot.prefix_cache.tokens = shared.clone();
        slot.prefix_cache.prefix_len = 60_000;
        slot.prefix_ckpts = vec![ckpt_n(60_000, 0xAA)];
        slot.retained_at = Some(std::time::Instant::now());
        let kv = FakeSeqKv::new();
        kv.seed_state(0, vec![0xCD; 60_000]);
        let store = Arc::new(std::sync::Mutex::new(PrefixStore::new()));

        let mut slots = vec![slot];
        release_retained_under_pressure(&mut slots, &kv, 60_000, &store);
        assert_eq!(kv.n_past_seq(0), 0, "watermark cleared GPU");

        // same session: shared 60k + small suffix
        let mut prompt = shared.clone();
        prompt.extend((0..1_000).map(|i| (100 + i % 3) as Token));
        let bound = bind_slot(&mut slots[0], job_with_tokens(prompt), &kv, 100_000, &store);
        assert!(bound, "same-session bind after watermark must succeed");
        let job = slots[0].job.as_ref().expect("slot occupied");
        assert!(
            job.prompt_offset > 0 && job.prompt_offset == 60_000,
            "reused from chain anchor, got {}",
            job.prompt_offset
        );
        assert_eq!(kv.n_past_seq(0), 60_000, "GPU restored from chain anchor");
        assert_eq!(slots[0].prefix_ckpts.len(), 1, "60k anchor still resident");
    }

    #[test]
    fn other_session_bind_keeps_shared_head_prunes_long_anchor() {
        // Watermark release leaves a 60k host chain. A different session
        // binds → only the shared 12288 head LCP; the 60k anchor is pruned.
        let shared: Vec<Token> = (0..60_000).map(|i| (i % 7) as Token).collect();
        let mut slot = Slot::new(SlotId(0));
        slot.prefix_cache.tokens = shared.clone();
        slot.prefix_cache.prefix_len = 60_000;
        slot.prefix_ckpts = vec![ckpt_n(60_000, 0xAA)];
        slot.retained_at = Some(std::time::Instant::now());
        let kv = FakeSeqKv::new();
        kv.seed_state(0, vec![0xCD; 60_000]);
        let store = Arc::new(std::sync::Mutex::new(PrefixStore::new()));
        let mut slots = vec![slot];
        release_retained_under_pressure(&mut slots, &kv, 60_000, &store);

        // different session: shared tool head (12288) + different suffix
        let (head, _) = tool_head_and_prompt();
        let mut other_prompt = head.clone();
        other_prompt.extend((0..500).map(|i| (200 + i % 5) as Token));
        store.lock().unwrap().put(head.clone(), ckpt_n(12_288, 0xAB));

        let bound = bind_slot(&mut slots[0], job_with_tokens(other_prompt), &kv, 40_000, &store);
        assert!(bound, "other-session bind must succeed");
        let job = slots[0].job.as_ref().expect("slot occupied");
        assert_eq!(job.prompt_offset, 12_288, "shared head reused");
        assert!(
            slots[0].prefix_ckpts.iter().all(|c| c.n_tokens <= 12_288 + 1),
            "60k anchor pruned past reuse_len"
        );
    }

    #[test]
    fn pressure_release_clears_gpu_keeps_host_for_both_sessions() {
        // #8565 regression: two long idle sessions cross the pool watermark.
        // GPU must still be cleared (slot shortage avoided); host anchors and
        // session tokens stay for both.
        let mut slots = vec![Slot::new(SlotId(0)), Slot::new(SlotId(1))];
        for (i, slot) in slots.iter_mut().enumerate() {
            let shared: Vec<Token> = (0..70_000).map(|j| (j % 7 + i as Token) as Token).collect();
            slot.prefix_cache.tokens = shared;
            slot.prefix_cache.prefix_len = 70_000;
            slot.prefix_ckpts = vec![ckpt_n(70_000, 0xAA)];
            slot.retained_at = Some(std::time::Instant::now());
        }
        let kv = FakeSeqKv::new();
        kv.seed_state(0, vec![0xCD; 70_000]);
        kv.seed_state(1, vec![0xCE; 70_000]);

        let pool = 140_032;
        let store = Arc::new(std::sync::Mutex::new(PrefixStore::new()));
        release_retained_under_pressure(&mut slots, &kv, pool, &store);

        assert_eq!(kv.n_past_seq(0), 0, "GPU cleared to avoid #8565");
        assert_eq!(kv.n_past_seq(1), 0);
        for s in &slots {
            assert_eq!(s.prefix_ckpts.len(), 1, "host anchor kept");
            assert_eq!(s.prefix_cache.tokens.len(), 70_000, "session tokens kept");
            assert_eq!(slot_kv_used(s), 0, "host dump not GPU occupancy");
        }
    }

    // ── P8-C 4th stage: global host-RAM LRU ───────────────────────────────

    /// Global host-RAM cap test: when `PrefixStore.total_bytes` + all slot
    /// chain bytes cross the cap, drop the longest `PrefixStore` dump first,
    /// then each slot chain's unprotected middle. The shortest tool head
    /// (store stub, 8k–16k) and each slot's newest anchor survive.
    #[test]
    fn host_ram_cap_evicts_long_store_middle_keeps_shortest_head_and_newest() {
        // Slot 0 chain: newest 60k + mid 30k + store stub 12288.
        let mut slot0 = Slot::new(SlotId(0));
        push_prefix_ckpt(&mut slot0, 60_000, vec![0u8; 60_000]);
        push_prefix_ckpt(&mut slot0, 30_000, vec![0u8; 30_000]);
        push_prefix_ckpt(&mut slot0, 12_288, vec![0u8; 12_288]);
        // Slot 1 chain: newest 40k + mid 20k.
        let mut slot1 = Slot::new(SlotId(1));
        push_prefix_ckpt(&mut slot1, 40_000, vec![0u8; 40_000]);
        push_prefix_ckpt(&mut slot1, 20_000, vec![0u8; 20_000]);

        // Global store: a long 50k dump + a short 12288 tool-head stub.
        let store = Arc::new(std::sync::Mutex::new(PrefixStore::new()));
        {
            let mut g = store.lock().unwrap();
            g.put(vec![1i32; 50_000], ckpt_n(50_000, 0xAB));
            g.put(vec![2i32; 12_288], ckpt_n(12_288, 0xAC));
        }

        // Total = store(50k+12288) + chain0(60k+30k+12288) + chain1(40k+20k)
        //        = 62_288 + 102_288 + 60_000 = 224_576. Cap at 100k forces
        //        both stages.
        let mut slots = vec![slot0, slot1];
        enforce_host_ram_cap(&mut slots, &store, 100_000);

        // Stage 1: longest PrefixStore dump (50k) evicted; short 12288 stub
        // stays because a longer dump existed.
        let g = store.lock().unwrap();
        assert_eq!(g.len(), 1, "long store dump evicted");
        let (key, ckpt) = g.entries.iter().next().expect("stub kept");
        assert_eq!(key.len(), 12_288, "short tool-head stub kept");
        assert_eq!(ckpt.n_tokens, 12_288);

        // Stage 2: each slot chain drops mid anchors, keeps newest + stub.
        let s0 = &slots[0];
        let s0_kept: Vec<u32> = s0.prefix_ckpts.iter().map(|c| c.n_tokens).collect();
        assert!(s0_kept.contains(&60_000), "newest 60k kept");
        assert!(s0_kept.contains(&12_288), "store stub 12288 kept");
        assert!(!s0_kept.contains(&30_000), "mid 30k dropped");

        let s1 = &slots[1];
        let s1_kept: Vec<u32> = s1.prefix_ckpts.iter().map(|c| c.n_tokens).collect();
        assert!(s1_kept.contains(&40_000), "newest 40k kept");
        assert!(!s1_kept.contains(&20_000), "mid 20k dropped");
    }

    #[test]
    fn empty_slot_bind_store_miss_reuses_zero() {
        let mut slot = Slot::new(SlotId(0));
        let (_head, prompt) = tool_head_and_prompt();
        let store = Arc::new(std::sync::Mutex::new(PrefixStore::new()));
        let kv = FakeSeqKv::new();
        let bound = bind_slot(&mut slot, job_with_tokens(prompt), &kv, 40_000, &store);
        assert!(bound);
        let job = slot.job.as_ref().expect("slot occupied");
        assert_eq!(job.prompt_offset, 0, "no store hit → full prefill");
        assert_eq!(job.n_past, 0);
    }

    #[test]
    fn apply_plan_promote_key_matches_job_prompt_n_tokens() {
        // First prefill: occupy does not copy the prompt into
        // prefix_cache.tokens. Promotion must key by job.prompt_tokens[..n].
        let mut slot = Slot::new(SlotId(0));
        let prompt: Vec<Token> = (0..20_000).map(|i| (i % 7) as Token).collect();
        slot.occupy(ActiveJob::for_test(prompt.clone()));
        assert!(
            slot.prefix_cache.tokens.is_empty(),
            "first prefill: prefix_cache.tokens is empty (the #93 promote bug)"
        );

        let kv = FakeSeqKv::with_n_ubatch(2048);
        kv.seed_state(0, vec![0xCD; 12_288]);
        let store = Arc::new(std::sync::Mutex::new(PrefixStore::new()));
        let plan = crate::batch::BatchPlan {
            prefill_consumed: vec![(SlotId(0), 12_288)],
            ..Default::default()
        };

        apply_plan(std::slice::from_mut(&mut slot), &plan, &kv, &store);

        let g = store.lock().unwrap();
        assert_eq!(g.len(), 1, "12288 stub must be promoted");
        let (key, ckpt) = g.entries.iter().next().expect("promoted entry");
        assert_eq!(key.len(), 12_288, "promote key length == n_tokens");
        assert_eq!(
            key.as_slice(),
            &prompt[..12_288],
            "promote key is job.prompt_tokens[..n_tokens], not prefix_cache.tokens"
        );
        assert_eq!(ckpt.n_tokens, 12_288);
        assert_eq!(ckpt.data, vec![0xCD; 12_288]);
        // If apply_plan keyed by prefix_cache.tokens (empty), snapshot_key
        // would return None and the store would stay empty.
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
                SlotEvent::PromptProgress { .. } => {}
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
                ..Default::default()
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
                ..Default::default()
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
                ..Default::default()
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
        let first = tokio::time::timeout(Duration::from_secs(30), async {
            while let Some(ev) = rx.recv().await {
                if !matches!(ev, SlotEvent::PromptProgress { .. }) {
                    return ev;
                }
            }
            panic!("channel closed before first token");
        })
        .await
        .expect("first event");
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

    async fn collect_timings(
        handle: &SchedulerHandle,
        prompt: &str,
        max_tokens: u32,
    ) -> SlotTimings {
        let (tx, mut rx) = mpsc::unbounded_channel();
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
        while let Some(ev) = rx.recv().await {
            match ev {
                SlotEvent::Finished { timings, .. } => return timings,
                SlotEvent::Failed(e) => panic!("job failed: {e}"),
                SlotEvent::Token(_) | SlotEvent::PromptProgress { .. } => {}
            }
        }
        panic!("channel closed without finish");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn second_turn_reuses_slot_prefix_kv() {
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
                ..Default::default()
            },
        );

        let turn1 = "<|im_start|>user\n안녕<|im_end|>\n<|im_start|>assistant\n";
        let t1 = collect_timings(&spawned.handle, turn1, 8).await;
        assert_eq!(t1.cache_n, 0, "first turn has no prefix");

        let turn2 = "<|im_start|>user\n안녕<|im_end|>\n<|im_start|>assistant\n응답<|im_end|>\n<|im_start|>user\n다음<|im_end|>\n<|im_start|>assistant\n";
        let t2 = collect_timings(&spawned.handle, turn2, 8).await;
        assert!(
            t2.cache_n > 0,
            "second turn must reuse slot KV, cache_n={}",
            t2.cache_n
        );
        assert!(
            t2.prompt_n < t1.prompt_n + t2.cache_n,
            "suffix prefill should be shorter than a full second prompt: t1.prompt_n={} t2.prompt_n={} t2.cache_n={}",
            t1.prompt_n,
            t2.prompt_n,
            t2.cache_n
        );

        drop(spawned.handle);
        let _ = spawned.worker.await;
    }
}
