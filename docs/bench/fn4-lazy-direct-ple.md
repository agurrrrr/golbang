# FN-SPEED-4: PLE lazy direct reads (`--lazy-mode on-direct`, PR #28136)

Date: 2026-09-20 · Issue #257 · Unit: `deploy/golbang-cuda-flashnext.service` (:8090)
Host: `cachyos-llm`, 2× Tesla V100-SXM2-16GB + EPYC 7452, CUDA 12.8.

## Decision

**ADOPTED.** `--lazy-mode on-direct` (ggml-org PR #28136, vendored as
`patches/cuda/0002`) is now in the unit. It removes ~99% of major faults and
gives a small but reproducible prefill gain (paired medians **+4.4%** cold,
**+3.6%/+4.5%** warm in both run orders) with **identical decode** and
**byte-identical greedy output**. Device read bytes do not change.

This is far from the PR's "2–3× cold prefill" on other hosts. Here the cold
prefill is bound by the 18 GiB model read and the CPU expert GEMV, not by the
PLE table's page faults; eliminating those faults is worth only a few percent.

## Background

`per_layer_token_embd.weight` (`qwen4exp` PLE n-gram table) is 27.4 GiB
(320,001,536 rows × 90 bytes) and `TENSOR_READ_LAZY`. The default `auto` lazy
mode serves it by demand-faulting the mmap: one 4 KiB page per ~90-byte row.
PR #28136 adds `--lazy-mode on-direct` (`LLAMA_LAZY_MODE_DIRECT`); the arch
stages all row indices of a ubatch host-side and reads them with sorted,
deduplicated, parallel `pread()`s, so the graph never touches the mapping.
The reported wins (DGX Spark, Strix Halo, Xeon+NVMe) are on cold, diverse-text
prefill where mmap scatter-gather was the bottleneck.

Note golbang did not expose `lazy_mode` at all (it used the llama default
`auto`). This change adds `--lazy-mode <off|auto|on|on-direct>` to
`golbang-server` (`LoadParams.lazy_mode` → `llama_model_params.lazy_mode`).

## Pin / patch composition

- base `911f6cdc8a` (ggml-org `origin/master`, 2026-09-18)
- + `patches/cuda/0001-qwen4exp-mtp-draft-head.patch` (PR #28243, MTP)
- + `patches/cuda/0002-qwen4exp-lazy-direct-reads.patch` (PR #28136)
- final tree `775aa4edc8ec16cb0d1a4876c05ca851d361a99e` = new `EXPECTED_SHA_CUDA`

PR #28136's base is older (`67a17c17c`, 2026-09-03). Its two commits
(`90fde1f7f` qwen4exp direct reads, `c6a9e5c9a` shared reader + gemma4) were
cherry-picked onto base+0001. The only conflict was the `--lazy-mode` help
line in `tools/llama-bench/llama-bench.cpp` (upstream removed the deprecated
`-mmp`/`-dio` lines since the PR's base); the HEAD text was kept and
`on-direct` added. `llama.h` grew 1645 → 1646 lines (one enum value); the C
API golbang binds is unchanged. Rollback: revert to `53b1389d0` (base+0001,
`patches/cuda/0001` only).

## Method

All requests go through the OpenAI endpoint with `max_tokens` 1–4; prefill
rate is `timings.prompt_per_second`, faults/IO are read from
`/proc/<pid>/stat` and `/proc/<pid>/io` around each request. Prompts are
shuffled concatenations of `ik_llama.cpp/github-data` markdown (diverse
English technical text; ~2.9 chars/token) with distinct seeds per rep so the
server's KV prefix cache never hits.

- **Cold**: `sync; echo 3 > drop_caches`, restart server, then one ~80k
  prefill (3 distinct prompts).
- **Warm-80k**: on the server left running after the cold reps, 3 more ~80k
  prefills (distinct prompts). One rep (`warm_2`) exceeded `n_ctx` (the corpus
  token density varies) so it is excluded; n=2.
- **Warm-15k**: fresh server, one 55k-token warmup (loads the model into page
  cache), then 5 distinct ~13k prompts. Run twice: `on` first, then
  `on-direct` first, to control for thermal/order drift.
- **Decode/identity**: interleaved `on, on-direct, on, on-direct`, 4×1500-token
  short decodes each, plus a greedy 200-token identity probe.

The `on` mode is the correct baseline: only `per_layer_token_embd` is marked
lazy, and at 27.4 GiB it is already lazy under production `auto`, so `auto ≡
on` here. (The PR also changed the loader so `auto` skips marked tensors
≤4 GiB while `on`/`on-direct` keep all of them; qwen4exp marks only the PLE
table, so the modes coincide.)

## Results — cold 80k (drop_caches + restart), 3 reps

| prompt | tokens | on t/s | on-direct t/s | Δ | on majflt | direct majflt | on MiB | direct MiB |
|--|--:|--:|--:|--:|--:|--:|--:|--:|
| cold_1 | 79469 | 182.3 | 187.1 | +2.7% | 176622 | 2632 | 18670.7 | 18670.7 |
| cold_2 | 79613 | 180.4 | 188.4 | +4.4% | 256262 | 3066 | 19011.0 | 19011.0 |
| cold_3 | 84151 | 172.8 | 184.3 | +6.7% | 368968 | 2873 | 19465.9 | 19465.9 |

paired Δ mean **+4.6%**, median **+4.4%** (n=3). Major faults −98.9%.
Read MiB identical per prompt.

## Results — warm 80k (same server after the cold reps)

| prompt | tokens | on t/s | on-direct t/s | Δ | on majflt | direct majflt | on MiB | direct MiB |
|--|--:|--:|--:|--:|--:|--:|--:|--:|
| warm_1 | 92840 | 179.1 | 184.9 | +3.2% | 172676 | 39 | 688.3 | 688.3 |
| warm_2 | — | — | — | — | — | — | — | — |
| warm_3 | 99831 | 184.5 | 181.4 | −1.7% | 62875 | 13 | 250.0 | 250.0 |

paired Δ mean/median **+0.8%** (n=2). `warm_2` was rejected (prompt > context).

## Results — warm ~15k (fresh server, warmup + 5 prompts)

Order A: `on` first.

| prompt | tokens | on t/s | on-direct t/s | Δ | on majflt | direct majflt |
|--|--:|--:|--:|--:|--:|--:|
| wu | 54795 | 190.8 | 198.5 | +4.0% | 112699 | 2695 |
| t1 | 22216 | 212.8 | 214.5 | +0.8% | 37796 | 41 |
| t2 | 13478 | 193.9 | 212.6 | +9.6% | 84245 | 74 |
| t3 | 13405 | 209.0 | 214.4 | +2.6% | 29778 | 20 |
| t4 | 15319 | 206.1 | 216.3 | +4.9% | 42245 | 26 |
| t5 | 14170 | 219.7 | 226.8 | +3.2% | 18160 | 14 |

paired Δ mean **+4.2%**, median **+3.6%** (n=6).

Order B: `on-direct` first (reversed).

| prompt | tokens | on t/s | on-direct t/s | Δ | on majflt | direct majflt |
|--|--:|--:|--:|--:|--:|--:|
| wu | 54795 | 192.1 | 198.4 | +3.3% | 112698 | 2696 |
| t1 | 22216 | 209.4 | 222.1 | +6.1% | 37796 | 41 |
| t2 | 13478 | 193.4 | 215.3 | +11.3% | 84245 | 74 |
| t3 | 13405 | 208.1 | 216.6 | +4.1% | 29778 | 20 |
| t4 | 15319 | 204.1 | 214.3 | +5.0% | 42245 | 26 |
| t5 | 14170 | 220.2 | 226.4 | +2.8% | 18160 | 14 |

paired Δ mean **+5.4%**, median **+4.5%** (n=6). All 12 warm pairs are positive.

The `on` major-fault counts are identical across both orders (e.g. 84245 for
t2), i.e. the PLE page touches are deterministic; `on-direct` cuts them to
tens. `io_read_bytes` is identical per prompt in every pair because the block
layer reads 4 KiB per scattered row either way — direct reads change *how* the
reads are issued (parallel, deduplicated, no synchronous fault), not *how
many bytes* leave the device.

## Results — decode and identity (interleaved)

| round | on t/s (n=4) | on-direct t/s (n=4) |
|--|--:|--:|
| 1 | 31.30, 32.37, 31.69, 31.96 | 32.89, 32.53, 30.03, 31.77 |
| 2 | 31.94, 31.18, 31.99, 32.10 | 32.22, 32.37, 31.80, 31.33 |

Median **on 31.95 t/s, on-direct 32.01 t/s (+0.2%)**, n=8 each — no decode
regression. (An initial non-interleaved pair showed on-direct ~4% slower; the
interleaved reruns show that was thermal/startup noise.) Greedy 200-token
responses from all four servers were **byte-identical** (len 383, same md5),
matching the PR's "greedy outputs bit-identical" claim.

## Interpretation

- The PR mechanism works exactly as advertised: major faults fall ~99%
  (cold 369k → 2.9k, warm ~84k → ~70), because the graph no longer
  demand-faults the PLE mapping.
- The throughput gain here is only ~4–5%. Golbang's cold prefill reads ~18 GiB
  of model weights and runs 42 CPU expert layers; PLE is a small slice. The
  warm case is cheap in absolute I/O (76–688 MiB of PLE blocks, well under a
  second of NVMe time), so removing the faults only shaves the synchronous
  stalls off a compute-bound prefill.
- The device still reads 4 KiB per scattered row, so no I/O volume is saved;
  the win is concurrency + no fault stalls. This is why the effect is modest
  on an NVMe host whose cold bottleneck is the model, unlike the PR's
  fault-serialized test boxes.
- Decode is unaffected: with `n_tokens=1` the gather uses a single worker, so
  there is no thread-contention regression against the 30-core CPU expert GEMV
  (the issue's stated risk), and the reader never makes the table resident.

## Adoption

- `deploy/golbang-cuda-flashnext.service`: `--lazy-mode on-direct` added
  (default stays `auto`; `on`/`on-direct` require mmap).
- `golbang-sys/build.rs`: `EXPECTED_SHA_CUDA` →
  `775aa4edc8ec16cb0d1a4876c05ca851d361a99e`, `EXPECTED_LLAMA_H_LINES` 1645 →
  1646.
- `scripts/build-llama.sh cuda`: applies `cuda/0001` + `cuda/0002`.
- `patches/README.md`: table and cuda section updated.

Production was rebuilt (`scripts/build-llama.sh cuda --force`,
`cargo build -p golbang-server --release` with `GOLBANG_GPU=cuda`), the unit
reinstalled and restarted. Startup log confirms
`direct reads enabled ... 128 threads` and `MTP draft context ready`; `/v1/models`
returns 200.

## Caveats

- The gain is small; if prefill is not the user-visible bottleneck, the main
  benefit is the fault reduction (less page-cache churn under pressure).
- The direct reader uses up to `2 × hardware_concurrency` = 128 in-flight
  workers per gather (hardcoded in the PR). Prefill is faster despite the
  oversubscription, and decode uses one worker, but this is worth revisiting if
  the CPU topology changes or another CPU-heavy service shares the host.
- PR #28136 is still open upstream; `patches/cuda/0002` is pinned to its current
  head. Rebase on the next CUDA pin bump.

## Reproduce

```bash
# tree + build
scripts/build-llama.sh cuda --force --dir /home/agurrrrr/code/local-llm/llama.cpp-cuda-upstream
PATH=~/.cargo/bin:$PATH CARGO_TARGET_DIR=target-cuda GOLBANG_GPU=cuda \
  GOLBANG_LLAMA_DIR=/home/agurrrrr/code/local-llm/llama.cpp-cuda-upstream \
  GOLBANG_LLAMA_BIN_DIR=/home/agurrrrr/code/local-llm/llama.cpp-cuda-upstream/build/bin \
  cargo build -p golbang-server --release
# harness scripts and raw records
ls docs/bench/raw/fn4-lazy-direct/
python3 docs/bench/raw/fn4-lazy-direct/analyze.py   # reprints the tables above
```
