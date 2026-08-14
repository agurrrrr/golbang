//! Unified `llama_batch` plan: active slots → one decode submission.
//! Chunking across iterations is allowed; P3 attaches a budget to the policy.

use crate::policy::IterationBudget;
use crate::slot::{Slot, SlotId, SlotPhase};
use crate::tokenizer::Token;

/// One token in the next `llama_decode` call.
#[derive(Clone, Copy, Debug)]
pub struct BatchToken {
    pub token: Token,
    pub pos: i32,
    pub seq_id: i32,
    pub logits: bool,
}

/// Planner output. `tokens.len()` is always ≤ `n_batch`.
#[derive(Clone, Debug, Default)]
pub struct BatchPlan {
    pub tokens: Vec<BatchToken>,
    /// Slot for each `tokens[i]` with `logits == true`, in that order.
    pub logit_slots: Vec<SlotId>,
    /// Prompt tokens consumed this plan, per prefilling slot.
    pub prefill_consumed: Vec<(SlotId, u32)>,
}

impl BatchPlan {
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

pub struct BatchBuilder {
    pub n_batch: usize,
}

impl BatchBuilder {
    pub fn new(n_batch: usize) -> Self {
        Self {
            n_batch: n_batch.max(1),
        }
    }

    /// Decode tokens first (keep generation moving), then fill leftover
    /// capacity with prefill. Does not mutate slots — apply after decode.
    ///
    /// P3 §4.2: the iteration budget bounds decode tokens per iteration and
    /// splits any single slot's prefill into `prefill_max`-token chunks so a
    /// long prompt does not starve other slots' decode.
    pub fn plan(&self, slots: &[Slot], order: &[SlotId], budget: IterationBudget) -> BatchPlan {
        let mut plan = BatchPlan::default();
        let cap = self.n_batch;

        // Decode pass — bounded by budget.decode_max so a prefill burst cannot
        // push decode out of the iteration.
        let mut decode_taken = 0usize;
        for &id in order {
            if plan.tokens.len() >= cap || decode_taken >= budget.decode_max {
                break;
            }
            let Some(slot) = slots.iter().find(|s| s.id == id) else {
                continue;
            };
            if slot.phase != SlotPhase::Decoding {
                continue;
            }
            let Some(job) = slot.job.as_ref() else {
                continue;
            };
            let Some(tok) = job.pending else {
                continue;
            };
            plan.tokens.push(BatchToken {
                token: tok,
                pos: job.n_past as i32,
                seq_id: id.0 as i32,
                logits: true,
            });
            plan.logit_slots.push(id);
            decode_taken += 1;
            for (k, &draft) in job.drafts.iter().enumerate() {
                if plan.tokens.len() >= cap || decode_taken >= budget.decode_max {
                    break;
                }
                plan.tokens.push(BatchToken {
                    token: draft,
                    pos: job.n_past as i32 + k as i32 + 1,
                    seq_id: id.0 as i32,
                    logits: true,
                });
                plan.logit_slots.push(id);
                decode_taken += 1;
            }
        }

        // Prefill pass — each slot consumes at most `budget.prefill_max` tokens
        // per iteration (chunked prefill).
        for &id in order {
            if plan.tokens.len() >= cap {
                break;
            }
            let Some(slot) = slots.iter().find(|s| s.id == id) else {
                continue;
            };
            if slot.phase != SlotPhase::Prefilling {
                continue;
            }
            let Some(job) = slot.job.as_ref() else {
                continue;
            };
            let remaining = job.prefill_remaining();
            if remaining == 0 {
                continue;
            }
            let take = remaining
                .min(cap - plan.tokens.len())
                .min(budget.prefill_max);
            // n_past already accounts for the reused prefix (prompt_offset).
            // The first prefill token is `prompt_offset + prompt_pos`.
            let start_pos = job.n_past;
            for k in 0..take {
                let idx = job.prefill_cursor() + k;
                let last = idx + 1 == job.prompt_tokens.len();
                plan.tokens.push(BatchToken {
                    token: job.prompt_tokens[idx],
                    pos: (start_pos + k as u32) as i32,
                    seq_id: id.0 as i32,
                    logits: last,
                });
                if last {
                    plan.logit_slots.push(id);
                }
            }
            plan.prefill_consumed.push((id, take as u32));
        }

        plan
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slot::Slot;

    #[test]
    fn prefills_then_stops_at_n_batch() {
        let mut slot = Slot::new(SlotId(0));
        slot.phase = SlotPhase::Prefilling;
        slot.job = Some(crate::slot::ActiveJob::for_test(vec![1, 2, 3, 4, 5]));
        let builder = BatchBuilder::new(3);
        let plan = builder.plan(&[slot], &[SlotId(0)], IterationBudget::default());
        assert_eq!(plan.tokens.len(), 3);
        assert!(plan.logit_slots.is_empty());
        assert_eq!(plan.prefill_consumed, vec![(SlotId(0), 3)]);
        assert!(!plan.tokens[2].logits);
    }

    #[test]
    fn last_prompt_token_requests_logits() {
        let mut slot = Slot::new(SlotId(1));
        slot.phase = SlotPhase::Prefilling;
        slot.job = Some(crate::slot::ActiveJob::for_test(vec![7, 8]));
        let plan = BatchBuilder::new(8).plan(&[slot], &[SlotId(1)], IterationBudget::default());
        assert_eq!(plan.tokens.len(), 2);
        assert!(plan.tokens[0].logits == false);
        assert!(plan.tokens[1].logits);
        assert_eq!(plan.logit_slots, vec![SlotId(1)]);
    }

    /// P3 prefix reuse: with prompt_offset > 0, prefill must start at the
    /// suffix token `[prompt_offset..)` and n_past must be seeded at reuse_len.
    fn slot_with_offset(tokens: Vec<Token>, offset: usize, n_past: u32) -> Slot {
        let mut slot = Slot::new(SlotId(2));
        slot.phase = SlotPhase::Prefilling;
        let mut job = crate::slot::ActiveJob::for_test(tokens);
        job.prompt_offset = offset;
        job.n_past = n_past;
        slot.job = Some(job);
        slot
    }

    #[test]
    fn prefill_starts_at_reused_prefix_suffix() {
        // prompt = [1,2,3,4,5], reused prefix [1,2] → offset=2, n_past=2.
        let slot = slot_with_offset(vec![1, 2, 3, 4, 5], 2, 2);
        let plan = BatchBuilder::new(8).plan(&[slot], &[SlotId(2)], IterationBudget::default());
        assert_eq!(plan.tokens.len(), 3, "only the suffix is prefilled");
        // tokens must be the suffix [3,4,5], positions continue from n_past=2.
        let toks: Vec<i32> = plan.tokens.iter().map(|t| t.token).collect();
        assert_eq!(toks, vec![3, 4, 5]);
        let pos: Vec<i32> = plan.tokens.iter().map(|t| t.pos).collect();
        assert_eq!(pos, vec![2, 3, 4]);
        // last suffix token requests logits.
        assert_eq!(plan.logit_slots, vec![SlotId(2)]);
    }

    #[test]
    fn prefill_chunks_do_not_reemit_reused_prefix() {
        // prompt = [1,2,3,4,5], offset=2. n_batch=2 → two chunks over the suffix.
        let slot = slot_with_offset(vec![1, 2, 3, 4, 5], 2, 2);
        let plan = BatchBuilder::new(2).plan(&[slot], &[SlotId(2)], IterationBudget::default());
        assert_eq!(plan.tokens.len(), 2);
        assert_eq!(plan.prefill_consumed, vec![(SlotId(2), 2)]);
        let toks: Vec<i32> = plan.tokens.iter().map(|t| t.token).collect();
        assert_eq!(toks, vec![3, 4], "first chunk covers suffix head only");
    }

    /// P3 §4.2: a single slot's prefill is chunked by budget.prefill_max,
    /// even when n_batch would allow more — so a long prompt doesn't starve
    /// other slots' decode in one iteration.
    #[test]
    fn prefill_is_chunked_by_budget() {
        // prompt of 100 tokens, offset=0. budget.prefill_max=32 → 32 tokens.
        let slot = slot_with_offset((0..100).collect(), 0, 0);
        let budget = IterationBudget {
            prefill_max: 32,
            decode_max: 16,
        };
        let plan = BatchBuilder::new(4096).plan(&[slot], &[SlotId(2)], budget);
        assert_eq!(
            plan.tokens.len(),
            32,
            "prefill capped at budget.prefill_max"
        );
        assert_eq!(plan.prefill_consumed, vec![(SlotId(2), 32)]);
    }

    /// P3 §4.2: decode is bounded by budget.decode_max so a batch of many
    /// decoding slots cannot crowd out prefill entirely.
    #[test]
    fn decode_is_bounded_by_budget() {
        let mut slots = Vec::new();
        for i in 0..8u32 {
            let mut slot = Slot::new(SlotId(i));
            slot.phase = SlotPhase::Decoding;
            let mut job = crate::slot::ActiveJob::for_test(vec![]);
            job.pending = Some(42 + i as i32);
            slot.job = Some(job);
            slots.push(slot);
        }
        let order: Vec<SlotId> = (0..8).map(SlotId).collect();
        let budget = IterationBudget {
            prefill_max: 32,
            decode_max: 3,
        };
        let plan = BatchBuilder::new(4096).plan(&slots, &order, budget);
        assert_eq!(plan.tokens.len(), 3, "decode capped at budget.decode_max");
        assert_eq!(plan.logit_slots.len(), 3);
    }

    #[test]
    fn decode_includes_pending_then_drafts() {
        let mut slot = Slot::new(SlotId(0));
        slot.phase = SlotPhase::Decoding;
        let mut job = crate::slot::ActiveJob::for_test(vec![]);
        job.n_past = 10;
        job.pending = Some(7);
        job.drafts = vec![8, 9];
        slot.job = Some(job);
        let plan = BatchBuilder::new(16).plan(&[slot], &[SlotId(0)], IterationBudget::default());
        let toks: Vec<i32> = plan.tokens.iter().map(|t| t.token).collect();
        assert_eq!(toks, vec![7, 8, 9]);
        let pos: Vec<i32> = plan.tokens.iter().map(|t| t.pos).collect();
        assert_eq!(pos, vec![10, 11, 12]);
        assert_eq!(plan.logit_slots, vec![SlotId(0), SlotId(0), SlotId(0)]);
        assert!(plan.tokens.iter().all(|t| t.logits));
    }
}
