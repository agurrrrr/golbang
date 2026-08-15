//! Swappable schedule policy. The scheduler only calls this trait;
//! FIFO is the P2 default. P3 chunked-prefill budget hangs off `rank`.

use crate::slot::{SlotId, SlotPhase};

#[derive(Clone, Copy, Debug)]
pub struct SlotView {
    pub id: SlotId,
    pub phase: SlotPhase,
    pub n_past: u32,
    pub n_generated: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct WaitingJobView {
    pub request_id: u64,
    pub n_prompt: u32,
}

/// Per-iteration token budget for the unified batch (P3 §4.2 chunked prefill).
///
/// `decode_max` bounds how many **decoding slots** may run this iteration so a
/// burst of prefill cannot starve already-running generation. Each admitted
/// slot includes its pending token plus speculative drafts (those extra tokens
/// must not count against `decode_max`, or `n_parallel=1` never verifies).
/// `prefill_max` bounds how many prefill tokens one slot may consume per
/// iteration, so a very long prompt is split across several iterations.
///
/// [`Default`] is the P3 unit-test placeholder (32 / 16). Production must call
/// [`IterationBudget::for_context`] so the chunk size matches llama.cpp
/// `n_batch` / `n_ubatch` — a 32-token cap on a 1024-ubatch context is the
/// DSV4 long-prompt stall (≈15 tok/s, ~100 decode launches per 3k tokens).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IterationBudget {
    pub prefill_max: usize,
    pub decode_max: usize,
}

impl Default for IterationBudget {
    fn default() -> Self {
        Self {
            prefill_max: 32,
            decode_max: 16,
        }
    }
}

impl IterationBudget {
    /// Align the planner with the llama context it will submit to.
    ///
    /// - `n_parallel == 1`: fill the logical `n_batch`. `llama_decode` splits
    ///   internally by `n_ubatch`, same as llama-server on a single slot.
    /// - `n_parallel > 1`: cap one slot at `n_ubatch` so other slots' decode
    ///   can run between physical ubatches (P3 fairness).
    pub fn for_context(n_batch: usize, n_ubatch: usize, n_parallel: usize) -> Self {
        let n_ubatch = n_ubatch.max(1);
        let n_batch = n_batch.max(n_ubatch);
        let n_parallel = n_parallel.max(1);
        Self {
            prefill_max: if n_parallel == 1 { n_batch } else { n_ubatch },
            decode_max: n_parallel,
        }
    }
}

pub trait SchedulePolicy: Send + 'static {
    fn name(&self) -> &'static str;

    /// Pair empty slots with waiting jobs. Return `(slot, waiting_index)`.
    /// Indices refer to the current `waiting` slice; duplicates are ignored.
    fn join(&mut self, empty: &[SlotId], waiting: &[WaitingJobView]) -> Vec<(SlotId, usize)>;

    /// Extra evictions this iteration. Cancel / EOS / max_tokens are
    /// handled by the scheduler and must not be reimplemented here.
    fn evict(&mut self, active: &[SlotView]) -> Vec<SlotId>;

    /// Per-iteration budget (P3 §4.2). The default is the 32/16 placeholder;
    /// the scheduler replaces it with [`IterationBudget::for_context`] so
    /// production follows `n_batch` / `n_ubatch`.
    fn budget(&self) -> IterationBudget {
        IterationBudget::default()
    }

    /// Order used to fill the unified batch. Default: slot id.
    fn rank(&self, active: &[SlotView]) -> Vec<SlotId> {
        let mut ids: Vec<SlotId> = active.iter().map(|s| s.id).collect();
        ids.sort_by_key(|s| s.0);
        ids
    }
}

/// Arrival order. First waiting job gets the first empty slot.
#[derive(Clone, Debug)]
pub struct FifoPolicy {
    budget: IterationBudget,
}

impl Default for FifoPolicy {
    fn default() -> Self {
        Self {
            budget: IterationBudget::default(),
        }
    }
}

impl FifoPolicy {
    pub fn with_budget(budget: IterationBudget) -> Self {
        Self { budget }
    }
}

impl SchedulePolicy for FifoPolicy {
    fn name(&self) -> &'static str {
        "fifo"
    }

    fn budget(&self) -> IterationBudget {
        self.budget
    }

    fn join(&mut self, empty: &[SlotId], waiting: &[WaitingJobView]) -> Vec<(SlotId, usize)> {
        let n = empty.len().min(waiting.len());
        (0..n).map(|i| (empty[i], i)).collect()
    }

    fn evict(&mut self, _active: &[SlotView]) -> Vec<SlotId> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fifo_pairs_in_arrival_order() {
        let mut p = FifoPolicy::default();
        let empty = [SlotId(1), SlotId(0)];
        let waiting = [
            WaitingJobView {
                request_id: 10,
                n_prompt: 4,
            },
            WaitingJobView {
                request_id: 11,
                n_prompt: 4,
            },
        ];
        assert_eq!(
            p.join(&empty, &waiting),
            vec![(SlotId(1), 0), (SlotId(0), 1)]
        );
    }

    #[test]
    fn fifo_joins_nothing_without_empty_slots() {
        let mut p = FifoPolicy::default();
        let waiting = [WaitingJobView {
            request_id: 1,
            n_prompt: 1,
        }];
        assert!(p.join(&[], &waiting).is_empty());
    }

    #[test]
    fn policy_is_object_safe_and_swappable() {
        fn takes(_: Box<dyn SchedulePolicy>) {}
        takes(Box::new(FifoPolicy::default()));
    }

    #[test]
    fn context_budget_fills_n_batch_on_a_single_slot() {
        let b = IterationBudget::for_context(5800, 1024, 1);
        assert_eq!(
            b,
            IterationBudget {
                prefill_max: 5800,
                decode_max: 1,
            }
        );
    }

    #[test]
    fn context_budget_chunks_at_n_ubatch_when_slots_share_the_gpu() {
        let b = IterationBudget::for_context(5800, 1024, 2);
        assert_eq!(
            b,
            IterationBudget {
                prefill_max: 1024,
                decode_max: 2,
            }
        );
    }
}
