# FN-SPEED-6: qwen4exp sparse flash attention on Volta (PR #28770 + Volta opt-in)

Date: 2026-09-22 · Issue #273 · Unit: `deploy/golbang-cuda-flashnext.service` (:8090)
Host: `cachyos-llm`, 2× Tesla V100-SXM2-16GB + EPYC 7452, CUDA 12.8.

## Decision

**ADOPTED (opt-in).** ggml-org PR #28770 (`CUDA: enable sparse fa for qwen4`,
merged 2026-09-20) is vendored as `patches/cuda/0003`. It alone does **nothing**
on golbang's V100: upstream gates the sparse gather on `turing_mma_available(cc)`
(sm_75+), and the Volta `ncols2` selector picks by divisibility, so qwen4exp's
gqa 12 never selects `ncols2 == 8`. A golbang-local patch `patches/cuda/0004`
opts the V100 in for the DKQ/DV 256 shape only, behind `GOLBANG_VOLTA_SPARSE_FA=1`.

With it enabled, the long-context prefill of the production model improves by a
reproducible **+5~8 %** at ~84k tokens (paired, same prompt, both arms
bit-deterministic), decode is unchanged, and the greedy output is **NUMERIC**
(a different but valid kernel; not bit-identical).

Pin `775aa4edc` → `e7ec442b7` (`patches/cuda/0001`..`0004`).

## Background

Qwen3.8-Flash-Next is `qwen4exp`: `key_length = value_length = 256`,
`head_count / head_count_kv = 24 / 2 = 12` gqa, `indexer.top_k = 2048`. Its QSA
attention (`build_attn_qsa`) builds a per-cell mask (`kq_mask_top_k`) in which
only the selected key cells are finite. PR #28770 makes the CUDA FA kernel gather
exactly those cells (`ggml_cuda_flash_attn_ext_compact_mask`) into one index list
per group of `ncols1` queries (union of the group's visible columns), so the FA
scan is over `ncols1 * n_kv_max` columns instead of the whole KV. It adds the
shapes `(DKQ 256, DV 256, ncols1 1|8, ncols2 8)` and enables sparse in the
qwen4exp graph (`build_attn_mha(..., top_k->ne[0], ...)`, replacing the
`TODO: enable sparse attention`).

Two upstream gates block V100:

1. `ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse` requires
   `turing_mma_available(cc)`; with `CMAKE_CUDA_ARCHITECTURES=70` the highest
   compiled arch is 700 < 750, so it always returns false.
2. The new `may_use_sparse` shapes are `ncols2 == 8`, but Volta's
   `switch_ncols2` picks `ncols2` by divisibility (`gqa_ratio % 8`, `% 4`, `% 2`);
   gqa 12 gives `ncols2 = 4`, so the `ncols2 8` case is never instantiated.

The `ncols1 == 1` sparse variant is also not Volta-compilable (`ncols1*ncols2 = 8
< 32` hits `NO_DEVICE_CODE` under `VOLTA_MMA_AVAILABLE`); the `ncols1 == 8`,
`ncols2 == 8` variant (64) is. So the fix is scoped to the DKQ/DV 256,
prefill/verify shape (`Q->ne[1] > 4`, `K->ne[1] >= max(4096, 16*n_kv_max)`);
decode keeps the default Volta tiling. `patch 0004` is inert unless
`GOLBANG_VOLTA_SPARSE_FA=1`.

The sparse gate activates at `n_kv >= max(4096, 2*8*2048) = 32768` for this
model, matching the PR's "32768 context" note.

## Setup

- tree: base `911f6cdc8a` + `patches/cuda/0001` (MTP #28243) + `0002` (lazy
  direct #28136) + `0003` (sparse FA #28770) + `0004` (Volta opt-in) = new
  `EXPECTED_SHA_CUDA e7ec442b7`, built `-DGGML_CUDA=ON
  -DCMAKE_CUDA_ARCHITECTURES=70`, CUDA 12.8 / g++-14.
- server: production flags (`--n-gpu-layers 99 --n-cpu-moe 42 --flash-attn on
  --tensor-split 11,1 --lazy-mode on-direct --n-ctx 100000 --n-batch 4096
  --n-ubatch 2048 --n-threads 32 --n-rs-seq 1 --n-parallel 1`), MTP on
  (`--spec-draft-n-max 4 --spec-draft-p-min 0.90`).
- prompt `a1` = 83,693 tokens (ik_llama.cpp `github-data`, RNG-shuffled), sent
  greedy (`temperature 0, top_k 1`), 100 output tokens, fresh server start per
  arm so the prefix cache is empty.
- `GOLBANG_VOLTA_SPARSE_FA=1` is the only difference between arms; with it unset
  the binary is byte-for-byte the baseline behaviour.

## Activation (instrumented)

The experimental tree logs once per process (also in the adopted patch):

```
INFO llama: [golbang] sparse FA active: DKQ=256 ncols1=8 n_kv_max=2051 n_kv=34816 n_queries=2048
```

`n_kv` 34816 is the first ubatch at or above the 32816 threshold; the log fires
only on the first activation, so it is printed once for an 84k prefill.

## A/B — same 83,693-token prompt, greedy 100 tokens

| arm | run | prefill t/s | decode t/s | md5 | len |
|-----|-----|------------:|-----------:|-----|----:|
| OFF (dense, baseline) | 1 | 197.0 | 13.10 | `03a5c724…` | 467 |
| OFF | 2 | 198.6 | 12.22 | `03a5c724…` | 467 |
| ON (`0004` v1, unconditional) | 1 | 206.8 | 12.26 | `86387400…` | 470 |
| ON (`0004` v1) | 2 | 208.1 | 12.50 | `86387400…` | 470 |
| ON (`0004` v2, thresholded) | 1 | 213.1 | 12.48 | `86387400…` | 470 |

- Prefill: OFF median 197.8 t/s, ON median 208.1 t/s → **+5.2 %** (range
  +4.5~+7.7 %). The `v1 → v2` change only restores the default dense tiling for
  the first (sub-32768) ubatches; the output md5 is identical, so the gain is the
  sparse kernel, not the dense `ncols2 8` tiling.
- Decode: OFF 12.2–13.1, ON 12.3–12.5 — flat within the MTP acceptance spread
  (draft 26–27, accepted 26). No regression.
- Determinism: each arm reproduces its own output exactly; the two arms differ
  from each other (NUMERIC). The first ~30 output characters are identical, then
  the fp-accumulation-order difference branches the greedy stream.

## Interpretation

The PR mechanism works on Volta: the gather compacts the QSA mask and the FA scan
skips the masked columns. The gain is prefill-only and modest because golbang's
prefill is also bound by the 18 GiB model read and the CPU expert GEMV; only the
ubatches whose `n_kv > 32768` (the tail ~60 % of an 84k prefill) see the sparse
kernel, and attention is only one part of their cost. This is the same shape of
result as FN-SPEED-4 (lazy-direct, +4.4 % prefill) and FN-SPEED-5 (QSA gather,
rejected only because it could not reach the MTP verify batch). Here the new
upstream PR does cover the multi-query tile, so the gain is reachable in
production.

## Caveats

- `patch 0004` is a golbang-local change to upstream CUDA code. It only affects
  `DKQ/DV == 256`, `ncols2 == 8`, `GOLBANG_VOLTA_SPARSE_FA=1`, so other CUDA
  models (DSV4 etc.) are untouched. Rebase it on the next CUDA pin bump.
- Output grade is NUMERIC, not byte-identical. If a strictly bit-identical V100
  path is ever needed, unset `GOLBANG_VOLTA_SPARSE_FA`.
- The gain is only for long prompts (`n_kv > 32816`). Short prompts and decode
  are unchanged.
- Only one 84k prompt (a1) was paired; two additional prompts were measured with
  temperature 1.0 and are recorded raw only (`off_b1_pp.json`, `off_b2_pp.json`,
  `on_a2`/`on_a3`).

## Reproduce

```bash
# tree + build (adds patches 0003/0004)
scripts/build-llama.sh cuda --force --dir /home/agurrrrr/code/local-llm/llama.cpp-cuda-upstream
PATH=~/.cargo/bin:$PATH CARGO_TARGET_DIR=target-cuda GOLBANG_GPU=cuda \
  GOLBANG_LLAMA_DIR=/home/agurrrrr/code/local-llm/llama.cpp-cuda-upstream \
  GOLBANG_LLAMA_BIN_DIR=/home/agurrrrr/code/local-llm/llama.cpp-cuda-upstream/build/bin \
  cargo build -p golbang-server --release
# run the unit's command with GOLBANG_VOLTA_SPARSE_FA=0/1 and compare prefill;
# raw payloads under docs/bench/raw/fn6-sparse-fa/.
```
