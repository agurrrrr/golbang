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

    /// Bar for same-session affinity (`settle_prefix_kv` restore).
    ///
    /// `resident()` includes generated tokens past the prefill checkpoint
    /// (Qwen think body). The next turn's prompt does not replay those
    /// tokens, so `lcp + 1 >= resident` fails and join picks the empty
    /// slot — then watermark pressure drops the real prefix (#8799 live).
    /// When a checkpoint exists, compare LCP to `ckpt_n` only.
    pub fn affinity_bar(&self) -> usize {
        if self.ckpt_n > 0 {
            self.ckpt_n as usize
        } else {
            self.resident()
        }
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
/// `mixed_prefill_max` bounds **total** prefill tokens in an iteration that
/// already has decode tokens. `0` = decode-only.
///
/// [`Default`] is the P3 unit-test placeholder (32 / 16 / 0). Production must
/// call [`IterationBudget::for_context`] so the chunk size matches llama.cpp
/// `n_batch` / `n_ubatch` — a 32-token cap on a 1024-ubatch context is the
/// DSV4 long-prompt stall (≈15 tok/s, ~100 decode launches per 3k tokens).
///
/// Issue #33: production `--n-batch 2048 --n-ubatch 2048 --n-parallel 2`
/// mixed a leftover 2048-token prefill into the same `llama_decode` as a
/// generating slot and dropped gfx906 decode from ~18 t/s to 0.68 t/s.
/// P3's "other slots decode between ubatches" needs `n_batch > n_ubatch`;
/// when they are equal the mix is one kernel. `mixed_prefill_max = 0`
/// keeps that iteration decode-only. Dual-prefill fairness is the planner
/// splitting leftover across prefilling slots, not this per-slot cap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IterationBudget {
    pub prefill_max: usize,
    pub decode_max: usize,
    /// Total prefill tokens allowed alongside decode. `0` = decode-only.
    pub mixed_prefill_max: usize,
}

impl Default for IterationBudget {
    fn default() -> Self {
        Self {
            prefill_max: 32,
            decode_max: 16,
            mixed_prefill_max: 0,
        }
    }
}

impl IterationBudget {
    /// Align the planner with the llama context it will submit to.
    ///
    /// - `n_parallel == 1`: fill the logical `n_batch`. `llama_decode` splits
    ///   internally by `n_ubatch`, same as llama-server on a single slot.
    /// - `n_parallel > 1`: cap one slot at `n_ubatch`. The planner splits that
    ///   leftover across prefilling slots. Decode-only when any slot is
    ///   generating (`mixed_prefill_max = 0`) — gfx906 cannot mix a large
    ///   prefill chunk into the same `llama_decode` as decode.
    pub fn for_context(n_batch: usize, n_ubatch: usize, n_parallel: usize) -> Self {
        let n_ubatch = n_ubatch.max(1);
        let n_batch = n_batch.max(n_ubatch);
        let n_parallel = n_parallel.max(1);
        Self {
            prefill_max: if n_parallel == 1 { n_batch } else { n_ubatch },
            decode_max: n_parallel,
            mixed_prefill_max: 0,
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
/// Pass [`EmptySlotView::affinity_bar`], not [`EmptySlotView::resident`]:
/// generated think tokens sit past `ckpt_n` and are not in the next prompt.
pub(crate) fn is_prefix_affinity(lcp: usize, bar: usize) -> bool {
    lcp > 0 && lcp + 1 >= bar
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
        if is_prefix_affinity(lcp, slot.affinity_bar()) {
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
    fn join_think_body_past_ckpt_keeps_affinity() {
        let mut p = FifoPolicy::default();
        // Prefill ckpt 10, then 5 generated think tokens (n_past=15).
        // Next prompt is the old prompt plus a tool result — LCP=10.
        let prefix0: Vec<Token> = (1..11).chain(900..905).collect();
        let mut job_tok: Vec<Token> = (1..11).collect();
        job_tok.extend([200, 201, 202]);
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
                prefix_len: 15,
                ckpt_n: 10,
            },
        ];
        let waiting = [WaitingJobView {
            request_id: 5,
            n_prompt: job_tok.len() as u32,
            tokens: &job_tok,
        }];
        assert_eq!(p.join(&empty, &waiting), vec![(SlotId(0), 0)]);
    }

    #[test]
    fn join_think_body_picks_later_ckpt_fork() {
        let mut p = FifoPolicy::default();
        // Two idle forks of the same session. Slot 0 stopped after turn 1
        // (ckpt 10 + think). Slot 1 stopped after turn 2 (ckpt 20 + think).
        // The new prompt continues turn 2. Picking the shorter fork (old
        // resident() spare rule) is the #8799 ping-pong.
        let prefix0: Vec<Token> = (1..11).chain(900..905).collect();
        let prefix1: Vec<Token> = (1..21).chain(910..915).collect();
        let mut job_tok: Vec<Token> = (1..21).collect();
        job_tok.extend([200, 201]);
        let empty = [
            EmptySlotView {
                id: SlotId(0),
                prefix: &prefix0,
                prefix_len: 15,
                ckpt_n: 10,
            },
            EmptySlotView {
                id: SlotId(1),
                prefix: &prefix1,
                prefix_len: 25,
                ckpt_n: 20,
            },
        ];
        let waiting = [WaitingJobView {
            request_id: 6,
            n_prompt: job_tok.len() as u32,
            tokens: &job_tok,
        }];
        assert_eq!(p.join(&empty, &waiting), vec![(SlotId(1), 0)]);
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
                mixed_prefill_max: 0,
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
                mixed_prefill_max: 0,
            }
        );
    }

    #[test]
    fn context_budget_mixed_prefill_is_decode_only_on_equal_n_batch() {
        // Production Qwen: --n-batch 2048 --n-ubatch 2048 --n-parallel 2.
        let b = IterationBudget::for_context(2048, 2048, 2);
        assert_eq!(b.prefill_max, 2048);
        assert_eq!(b.decode_max, 2);
        assert_eq!(b.mixed_prefill_max, 0);
    }
}
