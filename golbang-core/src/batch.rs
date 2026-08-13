//! Unified `llama_batch` plan: active slots → one decode submission.
//! Chunking across iterations is allowed; P3 attaches a budget to the policy.

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
    pub fn plan(&self, slots: &[Slot], order: &[SlotId]) -> BatchPlan {
        let mut plan = BatchPlan::default();
        let cap = self.n_batch;

        for &id in order {
            if plan.tokens.len() >= cap {
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
        }

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
            let remaining = job.prompt_tokens.len().saturating_sub(job.prompt_pos);
            if remaining == 0 {
                continue;
            }
            let take = remaining.min(cap - plan.tokens.len());
            let start_pos = job.n_past;
            for k in 0..take {
                let idx = job.prompt_pos + k;
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
        let plan = builder.plan(&[slot], &[SlotId(0)]);
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
        let plan = BatchBuilder::new(8).plan(&[slot], &[SlotId(1)]);
        assert_eq!(plan.tokens.len(), 2);
        assert!(plan.tokens[0].logits == false);
        assert!(plan.tokens[1].logits);
        assert_eq!(plan.logit_slots, vec![SlotId(1)]);
    }
}
