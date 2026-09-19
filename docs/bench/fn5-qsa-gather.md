# FN-SPEED-5: QSA gather sparse attention (PR #28213 / #28244)

Date: 2026-09-20 · Issue #258 · Unit: `deploy/golbang-cuda-flashnext.service` (:8090)
Host: `cachyos-llm`, 2× Tesla V100-SXM2-16GB + EPYC 7452, CUDA 12.8.

## Decision

**NOT ADOPTED.** The patch applies, builds and runs, and the gather path really
engages. But it does not help the production unit: the unit runs MTP speculative
decoding, and PR #28213's gather is **single-token-per-stream only**, so it can
only cover the single-token *seed* step of each speculative iteration, never the
multi-token verify batch. In two independent, fully reproducible controlled
rounds (greedy) at ~80k tokens, turning the gather on made decode **20.7 %
slower** (19.48 → 15.44 t/s) and cut the MTP draft count from 130 to 56. With
MTP off the patch does give a consistent **+4–5 %** at 77k/90k, but MTP off is
itself ~30 % slower than the production (MTP on) path, so that gain is not
reachable in production.

Pin, `patches/cuda/`, and the unit are **unchanged**. The experimental tree and
patch are recorded here only.

## Patch under test

| PR | commit | files | result |
|----|--------|-------|--------|
| ggml-org#28244 | `949079535` (closed, not merged) | `models.h`, `qwen4exp.cpp` | **does not apply** to pin `775aa4edc` — conflicts in `src/models/qwen4exp.cpp` (its base predates the current `build_attn_qsa` mask build) |
| ggml-org#28213 | `beed2f78a` (open) | `llama-graph.cpp`, `models.h`, `qwen4exp.cpp` | **applies cleanly** (`git apply --check` rc 0, no fuzz), `llama.h` unchanged (1646 lines) |

#28213 was chosen because it applies without conflict and exposes the runtime
A/B lever `QWEN4EXP_QSA_GATHER=0` (an escape hatch), which lets the same binary
serve both arms. It gathers the selected K/V cells into a compact buffer and
runs dense attention over them, deriving the mask from the existing per-cell
bias; it is gated on `n_tokens == n_stream` (single token per stream) and
`n_kv >= 4*width`, with `width = GGML_PAD(indexer_top_k + r - 1, 256) = 2304`.
Prompt processing and batched inference keep the existing masked scan.

## Setup

- tree: base `911f6cdc8a` + `patches/cuda/0001` (MTP #28243) + `patches/cuda/0002`
  (lazy direct reads #28136) = `775aa4edc` + #28213, built `-DGGML_CUDA=ON
  -DCMAKE_CUDA_ARCHITECTURES=70`, CUDA 12.8 / g++-14.
- server: production flags (`--n-gpu-layers 99 --n-cpu-moe 42 --flash-attn on
  --tensor-split 11,1 --lazy-mode on-direct --n-ctx 100000 --n-batch 4096
  --n-ubatch 2048 --n-threads 32 --n-rs-seq 1 --n-parallel 1`). MTP arm adds the
  Unsloth MTP head with `--spec-draft-n-max 4 --spec-draft-p-min 0.90`.
- prompts (ik_llama.cpp `github-data`, RNG-shuffled, distinct seeds): `p80` =
  77 472 tok, `p95` = 90 534 tok, `id13k` = 10 202 tok, `short` = 87 tok.
- configs: `m0g0`/`m0g0b` (MTP off, gather off, repeat for drift), `m0g1`
  (MTP off, gather on), `m1g0`/`m1g1` (MTP on, gather off/on). Raw payloads in
  `docs/bench/raw/fn5-qsa-gather/`. Each context is a fresh server start (cold
  prefix cache), decode 200 tokens. A concurrent HIP server on :8084 is always
  present; it is common to all arms.

## Gather engagement (instrumented)

`[QSA-GATHER-DBG]` was added to the experimental tree only.

- **MTP off**: every decode request runs `n_tokens=1 n_stream=1` and
  `n_kv >= 9216`, so `gather=1` on every generated token (the 87-token prompt is
  below the gate and stays on scan).
- **MTP on**: each speculative iteration issues a single-token *seed* step
  (`n_tokens=1` → `gather=1`) and then a multi-token verify step
  (`n_tokens=2..5` → `gather=0`). So the patch is live for roughly half the
  `llama_decode` calls and dead for the verify batch. Sample
  (`gather-debug-m1g1.txt`):

  ```
  n_tokens=1 n_stream=1 n_kv=77568 width=2304 gather=1
  n_tokens=2 n_stream=1 n_kv=77568 width=2304 gather=0
  n_tokens=1 n_stream=1 n_kv=77568 width=2304 gather=1
  n_tokens=4 n_stream=1 n_kv=77568 width=2304 gather=0
  ```

## A/B — MTP off (patch's intended single-token decode)

decode t/s, 200 tokens, temperature 1.0. `m0g0b` is a second gather-off run.

| prompt | tokens | `m0g0` off | `m0g1` on | `m0g0b` off | gather vs off |
|--------|-------:|-----------:|----------:|------------:|--------------:|
| short  | 87     | 23.95 | 24.22 | 24.29 | noise |
| id13k  | 10 202 | 21.50 | 21.14 | 22.28 | noise |
| p80    | 77 472 | 14.61 | **15.27** | 14.72 | **+4.1 %** |
| p95    | 90 534 | 13.81 | **14.54** | 13.86 | **+5.1 %** |

Prefill is flat across arms (p80 199.2/205.1/198.9 t/s, p95
198.8/192.1/199.0 t/s), as the PR says. The gather-off drift between the two
control runs is +0.8 % (p80) and +0.4 % (p95), so the ~+5 % at depth is above
drift and matches the PR's depth-dependent claim.

## A/B — MTP on (production path)

First, sampled decode (temperature 1.0), one run each:

| prompt | tokens | `m1g0` off | `m1g1` on |
|--------|-------:|-----------:|----------:|
| short  | 87     | 27.04 | 27.62 |
| id13k  | 10 202 | 23.79 | 22.61 |
| p80    | 77 472 | 16.64 | 14.82 |

The sampled runs are confounded by sampling: the gather arm generated a
different token stream with far fewer drafts (`draft_n` 87 → 46 at p80). A
controlled **greedy** A/B (temperature 0) was therefore run twice, interleaved,
and is fully reproducible:

| run | prompt | tokens | gather off t/s | drafts (acc/tot) | gather on t/s | drafts (acc/tot) |
|-----|--------|-------:|---------------:|-----------------:|--------------:|-----------------:|
| r1 | id13k | 10 202 | 23.29 | 83/84 | 21.76 | 63/64 |
| r1 | p80   | 77 472 | **19.01** | 120/130 | **15.26** | 54/56 |
| r2 | id13k | 10 202 | 23.06 | 83/84 | 21.72 | 63/64 |
| r2 | p80   | 77 472 | **19.94** | 120/130 | **15.61** | 54/56 |

Greedy output is deterministic: off round1 == off round2 and on round1 == on
round2, but off ≠ on from character 2 on. The patch changes the seed-step
logits, the generated stream changes, the MTP draft count at 80k falls
130 → 56, and decode drops **19.48 → 15.44 t/s (−20.7 %)** (13k: 23.18 → 21.74,
−6.2 %).

## Output grade

Greedy 200-token output, first 400 chars compared (`analyze` in the raw dir):

| context | pair | result |
|---------|------|--------|
| 87 tok  | off ~ on (both MTP modes) | **bit-identical** (gather never engages below the gate) |
| 10 202 tok, MTP off | `m0g0` ~ `m0g1` | diverge at char 2; `m0g0` 952 ch, `m0g1` 880 ch |
| 10 202 tok, MTP on  | `m1g0` ~ `m1g1` | diverge at char 2; 778 vs 856 ch |
| 77 472 tok, MTP on  | `m1g0` ~ `m1g1` | diverge at char 2; 712 vs 899 ch |

Both arms stay coherent; the divergence is numeric, i.e. grade **NUMERIC**, as
the issue anticipated. We did not establish that either path is the more correct
one, so the divergence is a further reason not to take it into production.

## Why the production path gets no gain

1. #28213 is single-token-per-stream by construction (`GGML_ASSERT(top_k->ne[1]
   == 1)`). The MTP verify batch carries `1 + n_drafts` tokens on the same
   stream, so it must fall back to the mask scan; only the seed step gathers.
2. Gather's advantage is skipping the O(n_kv) attention scan, but on this unit
   the per-token cost is dominated by the CPU MoE expert GEMV (30 of 32 cores,
   GPUs ~35 %/5 %). The attention scan is not the binding constraint, which is
   why even the MTP-off gain is only ~5 %, not the PR's +50 % on an
   attention-bound A6000 host.
3. Enabling gather also disables the per-block bias optimisation
   (`blk_bias = !gather && ...`), so the per-cell bias is built and uploaded per
   step — extra host work on an already CPU-bound decode.
4. Net in production: no improvement, a reproducible ~20 % regression at 80k
   through the draft-dynamics change, and a divergent output.

## Artifacts

- `docs/bench/raw/fn5-qsa-gather/` — raw request payloads (`m0g0`, `m0g1`,
  `m0g0b`, `m1g0`, `m1g1`, `m1ab/`) and the two `gather-debug-*.txt` excerpts.
- No change to `patches/cuda/`, `golbang-sys/build.rs`, `scripts/build-llama.sh`,
  or `deploy/golbang-cuda-flashnext.service`. `EXPECTED_SHA_CUDA` remains
  `775aa4edc`.
- `cargo build -p golbang-server --release` (`GOLBANG_GPU=cuda`) succeeds on the
  unchanged production pin.
