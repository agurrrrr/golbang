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
//! cap_i = min(S, max(used_i, T / n_active))
//! effective_i = max(used_i, min(cap_i, T - others_used))
//! ```
//!
//! Grown occupancy is never reduced. A solo slot may grow to `S` so a second
//! slot still has `T - S` cells; when both are below the fair share they
//! split `T` evenly.

/// Resolve CLI `--single-max-ctx`. `0` means the full pool.
pub fn resolve_single_max(single_max: u32, pool: u32) -> u32 {
    if single_max == 0 {
        pool
    } else {
        single_max.min(pool)
    }
}

/// Policy cap for one slot. `n_active` is the number of jobs, not empty
/// slots that still hold prefix KV.
pub fn slot_cap(single_max: u32, used: u32, pool: u32, n_active: u32) -> u32 {
    if pool == 0 {
        return 0;
    }
    let s = resolve_single_max(single_max, pool);
    let fair = pool / n_active.max(1);
    s.min(used.max(fair))
}

/// Physical bound: never below `used`, never past the leftover pool.
pub fn effective_cap(cap: u32, used: u32, pool: u32, others_used: u32) -> u32 {
    let room = pool.saturating_sub(others_used);
    used.max(cap.min(room))
}

/// Cap assigned to a slot that is about to join (`used = 0` for policy).
pub fn cap_for_join(others_used: u32, n_active_after: u32, pool: u32, single_max: u32) -> u32 {
    let policy = slot_cap(single_max, 0, pool, n_active_after);
    effective_cap(policy, 0, pool, others_used)
}

#[cfg(test)]
mod tests {
    use super::*;

    const T: u32 = 80_000;
    const S: u32 = 60_000;

    #[test]
    fn resolve_zero_is_full_pool() {
        assert_eq!(resolve_single_max(0, T), T);
        assert_eq!(resolve_single_max(S, T), S);
        assert_eq!(resolve_single_max(99_000, T), T);
    }

    #[test]
    fn solo_caps_at_single_max() {
        // min(S, max(0, T)) = S
        assert_eq!(slot_cap(S, 0, T, 1), S);
        assert_eq!(effective_cap(S, 0, T, 0), S);
        assert_eq!(cap_for_join(0, 1, T, S), S);
    }

    #[test]
    fn grown_solo_keeps_used_second_gets_remainder() {
        // slot1=60000, slot2 joins → slot1 stays 60k, slot2 gets leftover 20k
        let cap1 = slot_cap(S, 60_000, T, 2);
        assert_eq!(cap1, 60_000);
        assert_eq!(effective_cap(cap1, 60_000, T, 0), 60_000);
        assert_eq!(cap_for_join(60_000, 2, T, S), 20_000);
    }

    #[test]
    fn below_fair_share_redistributes_evenly() {
        // slot1=35000, slot2 joins → 40000 / 40000
        let cap1 = slot_cap(S, 35_000, T, 2);
        assert_eq!(cap1, 40_000);
        assert_eq!(effective_cap(cap1, 35_000, T, 0), 40_000);
        assert_eq!(cap_for_join(35_000, 2, T, S), 40_000);
    }

    #[test]
    fn above_fair_share_freezes_and_gives_rest() {
        // slot1=50000 > fair 40000 → freeze at 50000, slot2 gets 30000
        let cap1 = slot_cap(S, 50_000, T, 2);
        assert_eq!(cap1, 50_000);
        assert_eq!(effective_cap(cap1, 50_000, T, 0), 50_000);
        assert_eq!(cap_for_join(50_000, 2, T, S), 30_000);
    }

    #[test]
    fn leave_restores_solo_cap() {
        // slot2 gone, slot1 used=35000 → min(S, T) = 60000
        let cap1 = slot_cap(S, 35_000, T, 1);
        assert_eq!(cap1, S);
        assert_eq!(effective_cap(cap1, 35_000, T, 0), S);
    }

    #[test]
    fn leftover_prefix_of_empty_slot_shrinks_physical_room() {
        // slot2 finished but still holds 20000 prefix cells
        let cap1 = slot_cap(S, 35_000, T, 1);
        assert_eq!(cap1, S);
        assert_eq!(effective_cap(cap1, 35_000, T, 20_000), S);
        // 60000 prefix leftover: solo job only has 20000 cells left
        assert_eq!(effective_cap(S, 0, T, 60_000), 20_000);
        assert_eq!(cap_for_join(60_000, 1, T, S), 20_000);
    }

    #[test]
    fn default_single_max_lets_solo_use_full_pool() {
        assert_eq!(slot_cap(0, 0, T, 1), T);
        assert_eq!(cap_for_join(0, 1, T, 0), T);
        assert_eq!(slot_cap(0, 0, T, 2), 40_000);
        assert_eq!(cap_for_join(0, 2, T, 0), 40_000);
    }

    #[test]
    fn never_sets_cap_below_used() {
        // physical room 30000 but used already 35000 → freeze at used
        assert_eq!(effective_cap(60_000, 35_000, T, 50_000), 35_000);
    }

    #[test]
    fn empty_pool() {
        assert_eq!(slot_cap(S, 0, 0, 1), 0);
        assert_eq!(effective_cap(0, 0, 0, 0), 0);
        assert_eq!(cap_for_join(0, 1, 0, S), 0);
    }
}
