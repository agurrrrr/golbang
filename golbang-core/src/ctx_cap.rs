//! Dynamic per-slot context caps over a shared KV pool.
//!
//! llama.cpp allocates `T = n_ctx` cells (`n_ctx_cli * n_seq_max`, then padded).
//! That pool is actually shared across sequences only when `--kv-unified` is
//! on (`n_stream = 1`, `n_ctx_seq = T`). Default `kv_unified=false` gives each
//! sequence its own stream of `n_ctx_seq = T / n_seq_max` cells; a solo slot
//! cannot grow past that, even if the scheduler offers a larger cap.
//!
//! Policy (`--single-max-ctx` = `S`, 0 means `S = T`):
//!
//! ```text
//! spec_reserve = (spec_n_max + 1) * n_active   if spec_n_max > 0 else 0
//! T_usable     = T - spec_reserve
//! cap_i        = min(S, max(used_i, T_usable / n_active))
//! effective_i  = max(used_i, min(cap_i, T - others_used - spec_reserve))
//! ```
//!
//! `spec_n_max` is `SpecParams::verify_n_max()` (ngram-mod 64, MTP-only 3,
//! prompt-lookup `pld_k` (default 3), 0 = spec off). The extra cell is the sampled token that sits in front of
//! the draft batch. Grown occupancy is never reduced. A solo slot may grow
//! to `S` so a second slot still has `T - S` cells; when both are below the
//! fair share they split `T_usable` evenly.

/// Resolve CLI `--single-max-ctx`. `0` means the full pool.
pub fn resolve_single_max(single_max: u32, pool: u32) -> u32 {
    if single_max == 0 {
        pool
    } else {
        single_max.min(pool)
    }
}

/// KV cells one speculative verify step may occupy beyond `n_past`.
/// `spec_n_max == 0` means spec is off.
pub fn spec_headroom(spec_n_max: u32) -> u32 {
    if spec_n_max == 0 {
        0
    } else {
        spec_n_max.saturating_add(1)
    }
}

/// Pool-wide reservation so every active slot can verify without stealing
/// another sequence's cells.
pub fn spec_reserve(spec_n_max: u32, n_active: u32) -> u32 {
    spec_headroom(spec_n_max).saturating_mul(n_active.max(1))
}

/// Policy cap for one slot. `n_active` is the number of jobs, not empty
/// slots that still hold prefix KV.
pub fn slot_cap(single_max: u32, used: u32, pool: u32, n_active: u32, spec_n_max: u32) -> u32 {
    if pool == 0 {
        return 0;
    }
    let s = resolve_single_max(single_max, pool);
    let usable = pool.saturating_sub(spec_reserve(spec_n_max, n_active));
    let fair = usable / n_active.max(1);
    s.min(used.max(fair))
}

/// Physical bound: never below `used`, never past the leftover pool after
/// spec draft cells are reserved.
pub fn effective_cap(
    cap: u32,
    used: u32,
    pool: u32,
    others_used: u32,
    spec_n_max: u32,
    n_active: u32,
) -> u32 {
    let room = pool
        .saturating_sub(others_used)
        .saturating_sub(spec_reserve(spec_n_max, n_active));
    used.max(cap.min(room))
}

/// Cap assigned to a slot that is about to join (`used = 0` for policy).
pub fn cap_for_join(
    others_used: u32,
    n_active_after: u32,
    pool: u32,
    single_max: u32,
    spec_n_max: u32,
) -> u32 {
    let policy = slot_cap(single_max, 0, pool, n_active_after, spec_n_max);
    effective_cap(policy, 0, pool, others_used, spec_n_max, n_active_after)
}

/// Admission outcome for a request that already has a candidate `cap`.
///
/// HAL-5 #249: halogen reserves `prompt + max_tokens` positions before a
/// conversation joins. golbang only tracked `used`, so a request could bind
/// with a cap smaller than its generation budget and then be clamped (or hit
/// the pool ceiling) mid-generation. [`admission_decision`] gates the join.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JoinDecision {
    /// `cap >= prompt + max_tokens`: the request can finish on its own cap.
    Admit(u32),
    /// `prompt + max_tokens` fits the pool but not the current free room.
    /// Keep the request waiting in arrival order instead of binding it short.
    Defer,
    /// The reservation is larger than any cap this request could ever get
    /// (it is bigger than the whole pool / solo cap). Waiting would starve
    /// it, so fall back to the legacy behaviour: bind and let `bind_slot`
    /// report `ContextFull` when even the prompt does not fit.
    Clamp(u32),
}

/// Admission reservation: only admit a joiner whose `prompt + max_tokens`
/// fits the cap it would receive. `solo` is the best cap any request could
/// get (empty pool, one active slot); `need > solo` means no amount of
/// waiting helps, so the request is clamped rather than starved.
pub fn admission_decision(
    cap: u32,
    prompt_len: u32,
    max_tokens_req: u32,
    pool: u32,
    single_max: u32,
    spec_n_max: u32,
) -> JoinDecision {
    // `--max-tokens 0` (or an omitted request field) means "no cap", sent as
    // `UNLIMITED_MAX_TOKENS`. It can never fit `prompt + max_tokens` in a finite
    // cap, so treating it as Defer would starve it forever. Admit it like the
    // oversized case: bind clamps `max_tokens` to the remaining room and
    // generation stops at EOS or the context ceiling.
    if max_tokens_req == crate::generate::UNLIMITED_MAX_TOKENS {
        return JoinDecision::Admit(cap);
    }
    let need = prompt_len.saturating_add(max_tokens_req.max(1));
    if cap >= need {
        return JoinDecision::Admit(cap);
    }
    let solo = cap_for_join(0, 1, pool, single_max, spec_n_max);
    if need > solo {
        return JoinDecision::Clamp(cap);
    }
    JoinDecision::Defer
}

/// Effective KV pool after the startup pool-fit retry (HAL-5 #249 item 2).
/// `requested_*` echo the CLI values; `effective_*` are what actually loaded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolFit {
    pub requested_ctx: u32,
    pub requested_ubatch: u32,
    pub effective_ctx: u32,
    pub effective_ubatch: u32,
    /// Number of load retries; `0` means the requested values loaded first try.
    pub downgrades: u32,
}

impl PoolFit {
    pub fn lowered(&self) -> bool {
        self.downgrades > 0
    }
}

/// Successive `(n_ctx_seq, n_ubatch)` candidates tried when a load fails.
///
/// The first entry is always the requested pair. Odd steps halve `n_ubatch`,
/// even steps halve `n_ctx_seq`, so the least disruptive fix (the common
/// 2×V100 `llama_init_from_model returned null` with `--n-ubatch 4096`) is
/// tried first and context is only given up if the ubatch halving did not
/// help. Both knobs stop at a floor and the list is bounded by `MAX_STEPS`,
/// so a hopeless load still exits quickly with a clear error.
pub fn pool_fit_ladder(n_ctx: u32, n_ubatch: u32) -> Vec<(u32, u32)> {
    const MIN_CTX: u32 = 4_096;
    const MIN_UBATCH: u32 = 256;
    const MAX_STEPS: usize = 6;
    let mut ctx = n_ctx.max(1);
    let mut ub = n_ubatch.max(1);
    let mut out = vec![(ctx, ub)];
    for step in 1..MAX_STEPS {
        if step % 2 == 1 {
            if ub > MIN_UBATCH {
                ub = (ub / 2).max(MIN_UBATCH);
            } else if ctx > MIN_CTX {
                ctx = (ctx / 2).max(MIN_CTX);
            } else {
                break;
            }
        } else if ctx > MIN_CTX {
            ctx = (ctx / 2).max(MIN_CTX);
        } else if ub > MIN_UBATCH {
            ub = (ub / 2).max(MIN_UBATCH);
        } else {
            break;
        }
        let cand = (ctx, ub);
        if out.last() != Some(&cand) {
            out.push(cand);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const T: u32 = 80_000;
    const S: u32 = 60_000;
    const OFF: u32 = 0;

    #[test]
    fn resolve_zero_is_full_pool() {
        assert_eq!(resolve_single_max(0, T), T);
        assert_eq!(resolve_single_max(S, T), S);
        assert_eq!(resolve_single_max(99_000, T), T);
    }

    #[test]
    fn spec_reserve_is_n_max_plus_one_times_active() {
        assert_eq!(spec_headroom(0), 0);
        assert_eq!(spec_headroom(3), 4);
        assert_eq!(spec_headroom(64), 65);
        assert_eq!(spec_reserve(0, 2), 0);
        assert_eq!(spec_reserve(3, 1), 4);
        assert_eq!(spec_reserve(64, 1), 65);
        assert_eq!(spec_reserve(64, 2), 130);
    }

    #[test]
    fn solo_caps_at_single_max() {
        // min(S, max(0, T)) = S
        assert_eq!(slot_cap(S, 0, T, 1, OFF), S);
        assert_eq!(effective_cap(S, 0, T, 0, OFF, 1), S);
        assert_eq!(cap_for_join(0, 1, T, S, OFF), S);
    }

    #[test]
    fn grown_solo_keeps_used_second_gets_remainder() {
        // slot1=60000, slot2 joins → slot1 stays 60k, slot2 gets leftover 20k
        let cap1 = slot_cap(S, 60_000, T, 2, OFF);
        assert_eq!(cap1, 60_000);
        assert_eq!(effective_cap(cap1, 60_000, T, 0, OFF, 2), 60_000);
        assert_eq!(cap_for_join(60_000, 2, T, S, OFF), 20_000);
    }

    #[test]
    fn below_fair_share_redistributes_evenly() {
        // slot1=35000, slot2 joins → 40000 / 40000
        let cap1 = slot_cap(S, 35_000, T, 2, OFF);
        assert_eq!(cap1, 40_000);
        assert_eq!(effective_cap(cap1, 35_000, T, 0, OFF, 2), 40_000);
        assert_eq!(cap_for_join(35_000, 2, T, S, OFF), 40_000);
    }

    #[test]
    fn above_fair_share_freezes_and_gives_rest() {
        // slot1=50000 > fair 40000 → freeze at 50000, slot2 gets 30000
        let cap1 = slot_cap(S, 50_000, T, 2, OFF);
        assert_eq!(cap1, 50_000);
        assert_eq!(effective_cap(cap1, 50_000, T, 0, OFF, 2), 50_000);
        assert_eq!(cap_for_join(50_000, 2, T, S, OFF), 30_000);
    }

    #[test]
    fn leave_restores_solo_cap() {
        // slot2 gone, slot1 used=35000 → min(S, T) = 60000
        let cap1 = slot_cap(S, 35_000, T, 1, OFF);
        assert_eq!(cap1, S);
        assert_eq!(effective_cap(cap1, 35_000, T, 0, OFF, 1), S);
    }

    #[test]
    fn leftover_prefix_of_empty_slot_shrinks_physical_room() {
        // slot2 finished but still holds 20000 prefix cells
        let cap1 = slot_cap(S, 35_000, T, 1, OFF);
        assert_eq!(cap1, S);
        assert_eq!(effective_cap(cap1, 35_000, T, 20_000, OFF, 1), S);
        // 60000 prefix leftover: solo job only has 20000 cells left
        assert_eq!(effective_cap(S, 0, T, 60_000, OFF, 1), 20_000);
        assert_eq!(cap_for_join(60_000, 1, T, S, OFF), 20_000);
    }

    #[test]
    fn default_single_max_lets_solo_use_full_pool() {
        assert_eq!(slot_cap(0, 0, T, 1, OFF), T);
        assert_eq!(cap_for_join(0, 1, T, 0, OFF), T);
        assert_eq!(slot_cap(0, 0, T, 2, OFF), 40_000);
        assert_eq!(cap_for_join(0, 2, T, 0, OFF), 40_000);
    }

    #[test]
    fn never_sets_cap_below_used() {
        // physical room 30000 but used already 35000 → freeze at used
        assert_eq!(effective_cap(60_000, 35_000, T, 50_000, OFF, 1), 35_000);
    }

    #[test]
    fn empty_pool() {
        assert_eq!(slot_cap(S, 0, 0, 1, OFF), 0);
        assert_eq!(effective_cap(0, 0, 0, 0, OFF, 1), 0);
        assert_eq!(cap_for_join(0, 1, 0, S, OFF), 0);
    }

    #[test]
    fn spec_off_matches_legacy_caps() {
        assert_eq!(slot_cap(S, 0, T, 1, 64), slot_cap(S, 0, T, 1, OFF));
        let fair = slot_cap(S, 35_000, T, 2, 64);
        assert_eq!(fair, 39_935);
        assert_ne!(fair, slot_cap(S, 35_000, T, 2, OFF));
    }

    #[test]
    fn spec_headroom_splits_fair_share_from_usable_pool() {
        // T_usable = 80000 - 130 = 79870, fair = 39935
        let cap1 = slot_cap(S, 35_000, T, 2, 64);
        assert_eq!(cap1, 39_935);
        assert_eq!(effective_cap(cap1, 35_000, T, 35_000, 64, 2), 39_935);
        assert_eq!(cap_for_join(35_000, 2, T, S, 64), 39_935);
    }

    #[test]
    fn spec_headroom_leaves_draft_cells_when_both_freeze_at_fair() {
        let cap = slot_cap(S, 39_935, T, 2, 64);
        assert_eq!(cap, 39_935);
        assert_eq!(effective_cap(cap, 39_935, T, 39_935, 64, 2), 39_935);
        assert_eq!(80_000 - 39_935 - 39_935, spec_reserve(64, 2));
    }

    #[test]
    fn spec_headroom_clips_leftover_prefix_room() {
        // 60000 prefix leftover, ngram 64, solo → 20000 - 65
        assert_eq!(effective_cap(S, 0, T, 60_000, 64, 1), 19_935);
        assert_eq!(cap_for_join(60_000, 1, T, S, 64), 19_935);
    }

    /// #8565 live numbers: pool 140032, retained 69638, ngram+mtp n_max=64.
    #[test]
    fn spec_headroom_matches_qwen38_8565_pool() {
        const POOL: u32 = 140_032;
        const SOLO: u32 = 128_000;
        const SPEC: u32 = 64;
        let others = 69_638;
        let used = 69_605;
        let policy = slot_cap(SOLO, used, POOL, 1, SPEC);
        assert_eq!(policy, SOLO);
        assert_eq!(
            effective_cap(policy, used, POOL, others, SPEC, 1),
            POOL - others - spec_reserve(SPEC, 1)
        );
        assert_eq!(effective_cap(policy, used, POOL, others, SPEC, 1), 70_329);
        // Dual ~70k would have bound without spec reserve (cap 70016) and then
        // overflow on drafts. With reserve the joiner is below 70k.
        let join = cap_for_join(70_000, 2, POOL, SOLO, SPEC);
        assert!(join < 70_000, "join cap {join} should reject a 70k prompt");
        assert_eq!(join, POOL - 70_000 - spec_reserve(SPEC, 2));
    }

    // ── HAL-5 #249: admission reservation ─────────────────────────────────

    #[test]
    fn admission_admits_when_prompt_plus_max_fits_cap() {
        // others_used 70000, n_active 2 → policy 40000, physical 10000.
        let cap = cap_for_join(70_000, 2, T, 0, OFF);
        assert_eq!(cap, 10_000);
        assert_eq!(
            admission_decision(cap, 4_000, 5_000, T, 0, OFF),
            JoinDecision::Admit(10_000)
        );
        // exact fit is admitted (need == cap)
        assert_eq!(
            admission_decision(cap, 6_000, 4_000, T, 0, OFF),
            JoinDecision::Admit(10_000)
        );
    }

    #[test]
    fn admission_defers_when_reservation_exceeds_free_room() {
        let cap = cap_for_join(70_000, 2, T, 0, OFF);
        // need 13000 > cap 10000, but 13000 <= solo pool 80000: wait.
        assert_eq!(
            admission_decision(cap, 8_000, 5_000, T, 0, OFF),
            JoinDecision::Defer
        );
    }

    #[test]
    fn admission_respects_solo_cap_not_whole_pool() {
        // single_max 60000: solo cap is S, so a 70000 reservation can never fit
        // even though the raw pool is 80000 → clamp, do not wait forever.
        assert_eq!(cap_for_join(0, 1, T, S, OFF), S);
        assert_eq!(
            admission_decision(10_000, 60_000, 10_000, T, S, OFF),
            JoinDecision::Clamp(10_000)
        );
        // 59000 + 1 <= S: still fits solo → defer while room is short.
        assert_eq!(
            admission_decision(10_000, 59_000, 1, T, S, OFF),
            JoinDecision::Defer
        );
    }

    #[test]
    fn admission_clamps_request_larger_than_pool() {
        // Prompt alone exceeds the pool: no amount of waiting helps.
        assert_eq!(
            admission_decision(0, 90_000, 1, T, 0, OFF),
            JoinDecision::Clamp(0)
        );
        // Prompt fits but generation budget cannot: still larger than solo.
        assert_eq!(
            admission_decision(5_000, 70_000, 20_000, T, 0, OFF),
            JoinDecision::Clamp(5_000)
        );
    }

    #[test]
    fn admission_zero_max_tokens_still_needs_one_cell() {
        // max_tokens 0 is normalised to 1 (a job always samples one token).
        assert_eq!(
            admission_decision(5_000, 5_000, 0, T, 0, OFF),
            JoinDecision::Defer,
            "prompt == cap leaves no room for the first token"
        );
        assert_eq!(
            admission_decision(5_001, 5_000, 0, T, 0, OFF),
            JoinDecision::Admit(5_001)
        );
    }

    #[test]
    fn admission_unlimited_always_admits() {
        // `--max-tokens 0`: bind and let generation stop at EOS/context instead
        // of deferring forever against an impossible `prompt + max_tokens`.
        use crate::generate::UNLIMITED_MAX_TOKENS;
        assert_eq!(
            admission_decision(10_000, 59_000, UNLIMITED_MAX_TOKENS, T, S, OFF),
            JoinDecision::Admit(10_000)
        );
        assert_eq!(
            admission_decision(0, 90_000, UNLIMITED_MAX_TOKENS, T, 0, OFF),
            JoinDecision::Admit(0)
        );
    }

    #[test]
    fn admission_spec_reserve_shrinks_the_room() {
        // 64-cell spec reserve tightens both the solo cap and the join cap.
        let cap = cap_for_join(35_000, 2, T, S, 64);
        assert_eq!(cap, 39_935);
        // need 39936 > cap 39935 and > solo (S - 65 = 59935)? no, <= solo → defer
        assert_eq!(
            admission_decision(cap, 35_000, 4_936, T, S, 64),
            JoinDecision::Defer
        );
        // need 39936 > solo 59935 is false, so only the exact boundary matters.
        assert_eq!(
            admission_decision(cap, 35_000, 4_935, T, S, 64),
            JoinDecision::Admit(39_935)
        );
    }

    #[test]
    fn pool_fit_ladder_halves_ubatch_before_ctx() {
        let ladder = pool_fit_ladder(100_000, 4_096);
        assert_eq!(ladder[0], (100_000, 4_096), "requested pair first");
        assert_eq!(ladder[1], (100_000, 2_048), "ubatch is the known failure");
        assert!(
            ladder.iter().any(|&(c, _)| c < 100_000),
            "ctx eventually drops"
        );
        assert!(ladder.len() <= 6, "bounded: {}", ladder.len());
        // monotonic non-increasing on both axes
        for pair in ladder.windows(2) {
            assert!(pair[1].0 <= pair[0].0 && pair[1].1 <= pair[0].1);
        }
    }

    #[test]
    fn pool_fit_ladder_small_request_is_single_step() {
        assert_eq!(pool_fit_ladder(256, 0), vec![(256, 1)]);
    }

    #[test]
    fn pool_fit_ladder_floors_ctx_and_ubatch() {
        let ladder = pool_fit_ladder(8_192, 1_024);
        let last = *ladder.last().unwrap();
        assert_eq!(last, (4_096, 256), "both floors reached");
    }
}
