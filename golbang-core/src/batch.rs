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
    /// P3 §4.2: the iteration budget bounds decode slots per iteration and
    /// splits any single slot's prefill into `prefill_max`-token chunks.
    ///
    /// Issue #33: when every active slot is prefilling, leftover `n_batch` is
    /// split across those slots so slot 0 cannot take `prefill_max == n_batch`
    /// and leave slot 1 at `n_tokens=0`. When any slot is decoding, total
    /// prefill this iteration is `mixed_prefill_max` (`0` = decode-only). A
    /// 2048-token mix on gfx906 dropped decode from ~18 t/s to 0.68 t/s.
    pub fn plan(&self, slots: &[Slot], order: &[SlotId], budget: IterationBudget) -> BatchPlan {
        self.plan_ex(slots, order, budget, false)
    }

    /// Prefill-only iteration: skip decode tokens so gfx906 does not mix a
    /// large prompt chunk into the same `llama_decode` as generation.
    pub fn plan_prefill_only(
        &self,
        slots: &[Slot],
        order: &[SlotId],
        budget: IterationBudget,
    ) -> BatchPlan {
        self.plan_ex(slots, order, budget, true)
    }

    fn plan_ex(
        &self,
        slots: &[Slot],
        order: &[SlotId],
        budget: IterationBudget,
        skip_decode: bool,
    ) -> BatchPlan {
        let mut plan = BatchPlan::default();
        let cap = self.n_batch;

        // Decode pass — `decode_max` is a slot cap (P3 fairness), not a token
        // cap. Counting drafts as tokens left n_parallel=1 units at ~10 t/s
        // with `/metrics` draft=0 (P7).
        let mut decode_slots = 0usize;
        if !skip_decode {
            for &id in order {
                if plan.tokens.len() >= cap || decode_slots >= budget.decode_max {
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
                decode_slots += 1;
                for (k, &draft) in job.drafts.iter().enumerate() {
                    if plan.tokens.len() >= cap {
                        break;
                    }
                    plan.tokens.push(BatchToken {
                        token: draft,
                        pos: job.n_past as i32 + k as i32 + 1,
                        seq_id: id.0 as i32,
                        logits: true,
                    });
                    plan.logit_slots.push(id);
                }
            }
        }

        let leftover = cap.saturating_sub(plan.tokens.len());
        let prefill_cap = if skip_decode {
            leftover
        } else if decode_slots > 0 {
            leftover.min(budget.mixed_prefill_max)
        } else {
            leftover
        };
        if prefill_cap == 0 {
            return plan;
        }

        let mut prefills: Vec<(SlotId, &crate::slot::ActiveJob)> = Vec::new();
        for &id in order {
            let Some(slot) = slots.iter().find(|s| s.id == id) else {
                continue;
            };
            if slot.phase != SlotPhase::Prefilling {
                continue;
            }
            let Some(job) = slot.job.as_ref() else {
                continue;
            };
            if job.prefill_remaining() == 0 {
                continue;
            }
            prefills.push((id, job));
        }
        if prefills.is_empty() {
            return plan;
        }

        let remaining: Vec<usize> = prefills
            .iter()
            .map(|(_, job)| job.prefill_remaining())
            .collect();
        let quotas = split_prefill_quota(prefill_cap, budget.prefill_max, &remaining);

        for ((id, job), take) in prefills.into_iter().zip(quotas) {
            if take == 0 {
                continue;
            }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MixMode {
    /// Decode first; leftover prefill only up to `mixed_prefill_max`.
    Auto,
    /// Skip decode; fill with prefill. Same kernel never mixes the two.
    PrefillOnly,
}

pub(crate) fn max_prefill_remaining(slots: &[Slot]) -> usize {
    slots
        .iter()
        .filter(|s| s.phase == SlotPhase::Prefilling)
        .filter_map(|s| s.job.as_ref())
        .map(|j| j.prefill_remaining())
        .max()
        .unwrap_or(0)
}

pub(crate) fn has_pending_decode(slots: &[Slot]) -> bool {
    slots.iter().any(|s| {
        s.phase == SlotPhase::Decoding && s.job.as_ref().is_some_and(|j| j.pending.is_some())
    })
}

/// Choose Auto (decode-only when mixed_prefill_max=0) vs a dedicated
/// prefill-only iteration. Short suffixes finish immediately so a 12k
/// generation cannot pin progress at the reused prefix for 13 minutes.
pub(crate) fn decide_mix_mode(
    slots: &[Slot],
    budget: IterationBudget,
    decode_iters_since_yield: u32,
) -> MixMode {
    let remaining = max_prefill_remaining(slots);
    if remaining == 0 || !has_pending_decode(slots) {
        return MixMode::Auto;
    }
    if remaining <= budget.prefill_finish_threshold() {
        return MixMode::PrefillOnly;
    }
    if budget.prefill_yield_every > 0
        && decode_iters_since_yield >= budget.prefill_yield_every as u32
    {
        MixMode::PrefillOnly
    } else {
        MixMode::Auto
    }
}

/// Split `cap` prefill tokens across slots that still have prompt left.
///
/// Each slot is also bound by `per_slot_max` (`budget.prefill_max`). Equal
/// shares first; unused remainder from a short slot goes to still-hungry
/// neighbors so `n_batch` is not left idle.
fn split_prefill_quota(cap: usize, per_slot_max: usize, remaining: &[usize]) -> Vec<usize> {
    let n = remaining.len();
    let mut take = vec![0usize; n];
    if n == 0 || cap == 0 || per_slot_max == 0 {
        return take;
    }
    let mut left = cap;
    loop {
        let hungry: Vec<usize> = (0..n)
            .filter(|&i| take[i] < remaining[i].min(per_slot_max))
            .collect();
        if hungry.is_empty() || left == 0 {
            break;
        }
        let share = (left / hungry.len()).max(1);
        let mut progressed = false;
        for &i in &hungry {
            if left == 0 {
                break;
            }
            let room = remaining[i].min(per_slot_max) - take[i];
            let add = room.min(share).min(left);
            if add == 0 {
                continue;
            }
            take[i] += add;
            left -= add;
            progressed = true;
        }
        if !progressed {
            break;
        }
    }
    take
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
            mixed_prefill_max: 0,
            ..IterationBudget::default()
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
            mixed_prefill_max: 0,
            ..IterationBudget::default()
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

    /// Production Qwen uses n_parallel=1 → decode_max=1. Drafts must still
    /// ride with that one slot or speculative verify never runs.
    #[test]
    fn decode_max_one_slot_keeps_all_drafts() {
        let mut slot = Slot::new(SlotId(0));
        slot.phase = SlotPhase::Decoding;
        let mut job = crate::slot::ActiveJob::for_test(vec![]);
        job.n_past = 4;
        job.pending = Some(1);
        job.drafts = vec![2, 3, 4];
        slot.job = Some(job);
        let budget = IterationBudget {
            prefill_max: 32,
            decode_max: 1,
            mixed_prefill_max: 0,
            ..IterationBudget::default()
        };
        let plan = BatchBuilder::new(16).plan(&[slot], &[SlotId(0)], budget);
        let toks: Vec<i32> = plan.tokens.iter().map(|t| t.token).collect();
        assert_eq!(toks, vec![1, 2, 3, 4]);
        assert_eq!(plan.logit_slots.len(), 4);
    }

    fn prefill_slot(id: u32, n: usize) -> Slot {
        let mut slot = Slot::new(SlotId(id));
        slot.phase = SlotPhase::Prefilling;
        slot.job = Some(crate::slot::ActiveJob::for_test(vec![1; n]));
        slot
    }

    fn decode_slot(id: u32, pending: Token) -> Slot {
        let mut slot = Slot::new(SlotId(id));
        slot.phase = SlotPhase::Decoding;
        let mut job = crate::slot::ActiveJob::for_test(vec![]);
        job.pending = Some(pending);
        slot.job = Some(job);
        slot
    }

    /// Production Qwen: --n-batch 2048 --n-ubatch 2048 --n-parallel 2.
    fn production_np2() -> (BatchBuilder, IterationBudget) {
        (
            BatchBuilder::new(2048),
            IterationBudget::for_context(2048, 2048, 2),
        )
    }

    /// Issue #33: two Prefilling slots must both consume in one iteration.
    /// The old planner gave slot 0 `prefill_max == n_batch` and slot 1 nothing.
    #[test]
    fn two_prefill_slots_both_consume_one_iteration() {
        let (builder, budget) = production_np2();
        let slots = [prefill_slot(0, 33155), prefill_slot(1, 30815)];
        let plan = builder.plan(&slots, &[SlotId(0), SlotId(1)], budget);
        let got: Vec<(u32, u32)> = plan
            .prefill_consumed
            .iter()
            .map(|(id, n)| (id.0, *n))
            .collect();
        assert_eq!(got, vec![(0, 1024), (1, 1024)], "n_batch split across both");
        assert_eq!(plan.tokens.len(), 2048);
    }

    /// Issue #33: leftover from a short slot must not stay unused while the
    /// other still has prompt — otherwise n_batch/n_prefill wastes the GPU.
    #[test]
    fn two_prefill_slots_short_slot_gives_remainder_to_neighbor() {
        let (builder, budget) = production_np2();
        let slots = [prefill_slot(0, 10), prefill_slot(1, 30815)];
        let plan = builder.plan(&slots, &[SlotId(0), SlotId(1)], budget);
        let got: Vec<(u32, u32)> = plan
            .prefill_consumed
            .iter()
            .map(|(id, n)| (id.0, *n))
            .collect();
        assert_eq!(got, vec![(0, 10), (1, 2038)]);
        assert_eq!(plan.tokens.len(), 2048);
    }

    /// Issue #33: a Decoding slot must not share llama_decode with a 2048-token
    /// prefill chunk. mixed_prefill_max=0 → decode-only, batch stays small.
    #[test]
    fn decoding_with_prefill_keeps_the_decode_batch_small() {
        let (builder, budget) = production_np2();
        assert_eq!(budget.mixed_prefill_max, 0);
        let slots = [decode_slot(0, 42), prefill_slot(1, 30815)];
        let plan = builder.plan(&slots, &[SlotId(0), SlotId(1)], budget);
        assert_eq!(
            plan.tokens.len(),
            1,
            "decode-only so gfx906 stays ~18 t/s, not 0.68"
        );
        assert_eq!(plan.logit_slots, vec![SlotId(0)]);
        assert!(
            plan.prefill_consumed.is_empty(),
            "large leftover must not fill with prefill"
        );
        assert_eq!(plan.tokens[0].token, 42);
    }

    /// mixed_prefill_max > 0 still caps the mix so decode is not buried in n_batch.
    #[test]
    fn decoding_with_prefill_honors_mixed_prefill_max() {
        let builder = BatchBuilder::new(2048);
        let budget = IterationBudget {
            prefill_max: 2048,
            decode_max: 2,
            mixed_prefill_max: 64,
            ..IterationBudget::default()
        };
        let slots = [decode_slot(0, 7), prefill_slot(1, 30815)];
        let plan = builder.plan(&slots, &[SlotId(0), SlotId(1)], budget);
        assert_eq!(plan.tokens.len(), 1 + 64);
        assert_eq!(plan.prefill_consumed, vec![(SlotId(1), 64)]);
        assert_eq!(plan.logit_slots[0], SlotId(0));
    }

    #[test]
    fn split_prefill_quota_table() {
        let cases: &[(&str, usize, usize, &[usize], &[usize])] = &[
            ("equal two", 2048, 2048, &[33155, 30815], &[1024, 1024]),
            ("short then long", 2048, 2048, &[10, 30815], &[10, 2038]),
            ("one slot", 2048, 2048, &[33155], &[2048]),
            ("per-slot cap", 2048, 512, &[10000, 10000], &[512, 512]),
            ("empty remaining", 2048, 2048, &[0, 100], &[0, 100]),
            ("odd leftover", 3, 32, &[10, 10], &[2, 1]),
        ];
        for (name, cap, per, remaining, want) in cases {
            let got = split_prefill_quota(*cap, *per, remaining);
            assert_eq!(&got, want, "{name}");
        }
    }

    fn prefill_with_reuse(id: u32, n_prompt: usize, reused: usize) -> Slot {
        let mut slot = Slot::new(SlotId(id));
        slot.phase = SlotPhase::Prefilling;
        let mut job = crate::slot::ActiveJob::for_test(vec![1; n_prompt]);
        job.prompt_offset = reused;
        job.n_past = reused as u32;
        slot.job = Some(job);
        slot
    }

    #[test]
    fn decide_mix_finishes_short_suffix_immediately() {
        // req 170: n_prompt=53446 reused=51111 remaining=2335 ≤ 2*2048.
        let budget = IterationBudget::for_context(2048, 2048, 2);
        let slots = [decode_slot(0, 42), prefill_with_reuse(1, 53446, 51111)];
        assert_eq!(max_prefill_remaining(&slots), 2335);
        assert_eq!(
            decide_mix_mode(&slots, budget, 0),
            MixMode::PrefillOnly,
            "short suffix must not wait for yield_every"
        );
    }

    #[test]
    fn decide_mix_waits_on_long_prefill_until_yield_every() {
        let budget = IterationBudget::for_context(2048, 2048, 2);
        let slots = [decode_slot(0, 42), prefill_slot(1, 50_000)];
        assert_eq!(decide_mix_mode(&slots, budget, 0), MixMode::Auto);
        assert_eq!(decide_mix_mode(&slots, budget, 15), MixMode::Auto);
        assert_eq!(decide_mix_mode(&slots, budget, 16), MixMode::PrefillOnly);
    }

    #[test]
    fn decide_mix_stays_auto_when_only_one_phase() {
        let budget = IterationBudget::for_context(2048, 2048, 2);
        let only_prefill = [prefill_slot(1, 50_000)];
        let only_decode = [decode_slot(0, 42)];
        assert_eq!(decide_mix_mode(&only_prefill, budget, 99), MixMode::Auto);
        assert_eq!(decide_mix_mode(&only_decode, budget, 99), MixMode::Auto);
    }

    #[test]
    fn plan_prefill_only_skips_decode_tokens() {
        let (builder, budget) = production_np2();
        let slots = [decode_slot(0, 42), prefill_slot(1, 30815)];
        let plan = builder.plan_prefill_only(&slots, &[SlotId(0), SlotId(1)], budget);
        assert!(plan.tokens.iter().all(|t| t.seq_id == 1));
        assert_eq!(plan.prefill_consumed, vec![(SlotId(1), 2048)]);
        assert_eq!(plan.tokens.len(), 2048);
        assert!(!plan.logit_slots.contains(&SlotId(0)));
    }

    #[test]
    fn plan_prefill_only_long_prefill_uses_yield_chunk() {
        let (builder, budget) = production_np2();
        let slots = [decode_slot(0, 42), prefill_slot(1, 50_000)];
        let mut yield_budget = budget;
        yield_budget.prefill_max = budget.yield_chunk(50_000);
        assert_eq!(yield_budget.prefill_max, 256);
        let plan = builder.plan_prefill_only(&slots, &[SlotId(0), SlotId(1)], yield_budget);
        assert_eq!(plan.tokens.len(), 256);
        assert_eq!(plan.prefill_consumed, vec![(SlotId(1), 256)]);
    }

    #[test]
    fn default_budget_does_not_yield_a_long_prefill() {
        let budget = IterationBudget::default();
        let slots = [decode_slot(0, 42), prefill_slot(1, 10_000)];
        assert_eq!(decide_mix_mode(&slots, budget, 100), MixMode::Auto);
    }
}
