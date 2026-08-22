//! Swappable schedule policy. The scheduler only calls this trait;
//! FIFO is the P2 default. P3 chunked-prefill budget hangs off `rank`.

use crate::prefix_cache::common_prefix_len;
use crate::slot::{SlotId, SlotPhase};
use crate::tokenizer::Token;

#[derive(Clone, Copy, Debug)]
pub struct SlotView {
    pub id: SlotId,
    pub phase: SlotPhase,
    pub n_past: u32,
    pub n_generated: u32,
}

/// Idle slot the policy may bind. Prefix KV can still be resident (P5).
#[derive(Clone, Copy, Debug)]
pub struct EmptySlotView<'a> {
    pub id: SlotId,
    /// Token ids whose KV is still in this slot's sequence.
    pub prefix: &'a [Token],
    /// `SlotPrefixCache.prefix_len` (kept in case `prefix` is empty).
    pub prefix_len: u32,
    /// Prefill-end checkpoint length; 0 if none.
    pub ckpt_n: u32,
}

impl EmptySlotView<'_> {
    /// Cells that joining this slot would discard on a mismatch.
    pub fn resident(&self) -> usize {
        self.prefix
            .len()
            .max(self.prefix_len as usize)
            .max(self.ckpt_n as usize)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct WaitingJobView<'a> {
    pub request_id: u64,
    pub n_prompt: u32,
    /// Tokenized prompt when join ran `encode`. Empty for vision / encode miss.
    pub tokens: &'a [Token],
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
    fn join(
        &mut self,
        empty: &[EmptySlotView<'_>],
        waiting: &[WaitingJobView<'_>],
    ) -> Vec<(SlotId, usize)>;

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

/// Arrival order, then prefix affinity.
///
/// Waiting jobs are considered FIFO. Each job takes the idle slot whose
/// resident prefix is the same session (`lcp + 1 >= resident`). If no slot
/// qualifies, it takes the smallest resident prefix so a ping-pong gap does
/// not wipe the other session's checkpoint (Qwen hybrid cannot `seq_rm` to a
/// shared tools head; `ckpt_n > lcp + 1` would full-prefill and clear).
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

/// True when LCP is the same session, not a shared tools/system head.
///
/// Matches `settle_prefix_kv`: a checkpoint is only restored when
/// `ckpt_n <= reuse_len + 1`. One token short covers `<think>` vs `</think>`.
pub(crate) fn is_prefix_affinity(lcp: usize, resident: usize) -> bool {
    lcp > 0 && lcp + 1 >= resident
}

fn prefix_aware_join(
    empty: &[EmptySlotView<'_>],
    waiting: &[WaitingJobView<'_>],
) -> Vec<(SlotId, usize)> {
    let n = empty.len().min(waiting.len());
    if n == 0 {
        return Vec::new();
    }
    let mut used = vec![false; empty.len()];
    let mut out = Vec::with_capacity(n);
    for (wait_idx, job) in waiting.iter().enumerate() {
        if out.len() == n {
            break;
        }
        let Some(slot_idx) = pick_empty_slot(empty, &used, job) else {
            break;
        };
        used[slot_idx] = true;
        out.push((empty[slot_idx].id, wait_idx));
    }
    out
}

fn pick_empty_slot(
    empty: &[EmptySlotView<'_>],
    used: &[bool],
    job: &WaitingJobView<'_>,
) -> Option<usize> {
    let mut best_aff: Option<(usize, usize)> = None;
    let mut best_spare: Option<(usize, usize)> = None;
    for (i, slot) in empty.iter().enumerate() {
        if used[i] {
            continue;
        }
        let lcp = common_prefix_len(slot.prefix, job.tokens);
        let resident = slot.resident();
        if is_prefix_affinity(lcp, resident) {
            if best_aff.is_none_or(|(_, best_lcp)| lcp > best_lcp) {
                best_aff = Some((i, lcp));
            }
        } else if best_spare.is_none_or(|(_, r)| resident < r) {
            best_spare = Some((i, resident));
        }
    }
    best_aff.or(best_spare).map(|(i, _)| i)
}

impl SchedulePolicy for FifoPolicy {
    fn name(&self) -> &'static str {
        "fifo"
    }

    fn budget(&self) -> IterationBudget {
        self.budget
    }

    fn join(
        &mut self,
        empty: &[EmptySlotView<'_>],
        waiting: &[WaitingJobView<'_>],
    ) -> Vec<(SlotId, usize)> {
        prefix_aware_join(empty, waiting)
    }

    fn evict(&mut self, _active: &[SlotView]) -> Vec<SlotId> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idle(id: u32) -> EmptySlotView<'static> {
        EmptySlotView {
            id: SlotId(id),
            prefix: &[],
            prefix_len: 0,
            ckpt_n: 0,
        }
    }

    fn wait(id: u64) -> WaitingJobView<'static> {
        WaitingJobView {
            request_id: id,
            n_prompt: 0,
            tokens: &[],
        }
    }

    #[test]
    fn fifo_pairs_in_arrival_order() {
        let mut p = FifoPolicy::default();
        let empty = [idle(1), idle(0)];
        let waiting = [wait(10), wait(11)];
        assert_eq!(
            p.join(&empty, &waiting),
            vec![(SlotId(1), 0), (SlotId(0), 1)]
        );
    }

    #[test]
    fn shared_tools_head_is_not_affinity() {
        // Production ping-pong: reuse_len=12545, ckpt_n=37233.
        assert!(!is_prefix_affinity(12_545, 37_233));
        assert!(is_prefix_affinity(37_232, 37_233));
        assert!(is_prefix_affinity(37_233, 37_233));
        assert!(!is_prefix_affinity(0, 0));
    }

    #[test]
    fn fifo_joins_nothing_without_empty_slots() {
        let mut p = FifoPolicy::default();
        let waiting = [wait(1)];
        assert!(p.join(&[], &waiting).is_empty());
    }

    #[test]
    fn join_prefers_slot_with_longer_lcp() {
        let mut p = FifoPolicy::default();
        let prefix0: Vec<Token> = (1..21).collect();
        let mut job_tok = prefix0.clone();
        job_tok.push(99);
        // empty listed as slot1 first so FIFO-by-index would have picked 1.
        let empty = [
            EmptySlotView {
                id: SlotId(1),
                prefix: &[],
                prefix_len: 0,
                ckpt_n: 0,
            },
            EmptySlotView {
                id: SlotId(0),
                prefix: &prefix0,
                prefix_len: 20,
                ckpt_n: 20,
            },
        ];
        let waiting = [WaitingJobView {
            request_id: 10,
            n_prompt: job_tok.len() as u32,
            tokens: &job_tok,
        }];
        assert_eq!(p.join(&empty, &waiting), vec![(SlotId(0), 0)]);
    }

    #[test]
    fn join_prefers_empty_slot_when_lcp_is_short() {
        let mut p = FifoPolicy::default();
        // Shared tools head (5) vs a 20-token checkpoint. Affinity would wipe
        // the other session and still full-prefill (ckpt_n > lcp+1).
        let prefix0: Vec<Token> = (1..21).collect();
        let job_tok: Vec<Token> = vec![1, 2, 3, 4, 5, 100, 101, 102];
        let empty = [
            EmptySlotView {
                id: SlotId(0),
                prefix: &prefix0,
                prefix_len: 20,
                ckpt_n: 20,
            },
            EmptySlotView {
                id: SlotId(1),
                prefix: &[],
                prefix_len: 0,
                ckpt_n: 0,
            },
        ];
        let waiting = [WaitingJobView {
            request_id: 11,
            n_prompt: job_tok.len() as u32,
            tokens: &job_tok,
        }];
        assert_eq!(p.join(&empty, &waiting), vec![(SlotId(1), 0)]);
    }

    #[test]
    fn join_two_sessions_keep_their_slots() {
        let mut p = FifoPolicy::default();
        let prefix_a: Vec<Token> = (1..21).collect();
        let prefix_b: Vec<Token> = (50..70).collect();
        let mut job_b = prefix_b.clone();
        job_b.push(7);
        let mut job_a = prefix_a.clone();
        job_a.push(8);
        let empty = [
            EmptySlotView {
                id: SlotId(0),
                prefix: &prefix_a,
                prefix_len: 20,
                ckpt_n: 20,
            },
            EmptySlotView {
                id: SlotId(1),
                prefix: &prefix_b,
                prefix_len: 20,
                ckpt_n: 20,
            },
        ];
        // B arrived first; FIFO-by-index would bind it to slot 0 and wipe A.
        let waiting = [
            WaitingJobView {
                request_id: 1,
                n_prompt: job_b.len() as u32,
                tokens: &job_b,
            },
            WaitingJobView {
                request_id: 2,
                n_prompt: job_a.len() as u32,
                tokens: &job_a,
            },
        ];
        assert_eq!(
            p.join(&empty, &waiting),
            vec![(SlotId(1), 0), (SlotId(0), 1)]
        );
    }

    #[test]
    fn join_lcp_one_below_resident_is_affinity() {
        let mut p = FifoPolicy::default();
        let prefix0: Vec<Token> = (1..11).collect();
        // think-tag: LCP = resident - 1 still restores (ckpt_n <= lcp+1).
        let job_tok: Vec<Token> = (1..10).chain(std::iter::once(99)).collect();
        let empty = [
            EmptySlotView {
                id: SlotId(1),
                prefix: &[],
                prefix_len: 0,
                ckpt_n: 0,
            },
            EmptySlotView {
                id: SlotId(0),
                prefix: &prefix0,
                prefix_len: 10,
                ckpt_n: 10,
            },
        ];
        let waiting = [WaitingJobView {
            request_id: 3,
            n_prompt: job_tok.len() as u32,
            tokens: &job_tok,
        }];
        assert_eq!(p.join(&empty, &waiting), vec![(SlotId(0), 0)]);
    }

    #[test]
    fn join_only_empty_slot_is_used_even_if_lcp_is_short() {
        let mut p = FifoPolicy::default();
        let prefix0: Vec<Token> = (1..21).collect();
        let job_tok: Vec<Token> = vec![9, 8, 7];
        let empty = [EmptySlotView {
            id: SlotId(0),
            prefix: &prefix0,
            prefix_len: 20,
            ckpt_n: 20,
        }];
        let waiting = [WaitingJobView {
            request_id: 4,
            n_prompt: job_tok.len() as u32,
            tokens: &job_tok,
        }];
        assert_eq!(p.join(&empty, &waiting), vec![(SlotId(0), 0)]);
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
