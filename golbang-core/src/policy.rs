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
/// `decode_max` bounds how many decode tokens are admitted per iteration so a
/// burst of prefill cannot starve already-running generation. `prefill_max`
/// bounds how many prefill tokens one slot may consume per iteration, so a
/// very long prompt is split across several iterations (chunked prefill).
#[derive(Clone, Copy, Debug)]
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

pub trait SchedulePolicy: Send + 'static {
    fn name(&self) -> &'static str;

    /// Pair empty slots with waiting jobs. Return `(slot, waiting_index)`.
    /// Indices refer to the current `waiting` slice; duplicates are ignored.
    fn join(&mut self, empty: &[SlotId], waiting: &[WaitingJobView]) -> Vec<(SlotId, usize)>;

    /// Extra evictions this iteration. Cancel / EOS / max_tokens are
    /// handled by the scheduler and must not be reimplemented here.
    fn evict(&mut self, active: &[SlotView]) -> Vec<SlotId>;

    /// Per-iteration budget (P3 §4.2). Default keeps decode moving and
    /// chunks prefill into 32-token slices.
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
#[derive(Clone, Debug, Default)]
pub struct FifoPolicy;

impl SchedulePolicy for FifoPolicy {
    fn name(&self) -> &'static str {
        "fifo"
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
        let mut p = FifoPolicy;
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
        assert_eq!(p.join(&empty, &waiting), vec![(SlotId(1), 0), (SlotId(0), 1)]);
    }

    #[test]
    fn fifo_joins_nothing_without_empty_slots() {
        let mut p = FifoPolicy;
        let waiting = [WaitingJobView {
            request_id: 1,
            n_prompt: 1,
        }];
        assert!(p.join(&[], &waiting).is_empty());
    }

    #[test]
    fn policy_is_object_safe_and_swappable() {
        fn takes(_: Box<dyn SchedulePolicy>) {}
        takes(Box::new(FifoPolicy));
    }
}
