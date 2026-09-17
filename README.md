# golbang

**English** · [한국어](README_KO.md)

> **golbang** (골뱅이, "snail") — a GGUF LLM inference server that serves **current
> large and MoE models** on AMD MI50 (gfx906) and NVIDIA V100.
> Serving, scheduling, and batching are Rust; GPU math calls a SHA-pinned
> llama.cpp (ggml) through the C ABI.
> The serving surface stays minimal: a single OpenAI-compatible chat API.

*OpenAI-compatible GGUF inference server. Rust orchestration (axum + tokio) on top of
SHA-pinned llama.cpp backends (HIP / CUDA / Vulkan), built to serve current large and
MoE models (Qwen3.8, DeepSeek-V4-Flash, GLM-5.3-Flash) on aging hardware like gfx906
that mainstream stacks are leaving behind. Single binary, no Python.*

![Rust](https://img.shields.io/badge/Rust-edition%202024-dea584?logo=rust&logoColor=white)
![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)
![API](https://img.shields.io/badge/API-OpenAI%20chat%20completions-6e56cf)
![GPU](https://img.shields.io/badge/backends-HIP%20%C2%B7%20CUDA%20%C2%B7%20Vulkan-orange)

Because it uses the **same GPU kernels** as llama-server, decode speed is roughly at
parity. What golbang aims at is not throughput but the **scheduling policy,
cancellation, immediate 503, and the lightweight control plane**, and its main stage is
getting **current large and MoE models** onto old cards like **gfx906 (MI50) and sm_70
(V100)** that are seeing thinner support. What is actually being served is in
[What it serves](#what-it-serves-mi50-and-v100), and the path that shortens prefill for
long agent conversations is in [Prefill shortcut: host prefix snapshots (P8)](#prefill-shortcut-host-prefix-snapshots-p8).
Measurements are reported as-is, not cherry-picked, in [Limitations](#limitations) and
[`docs/bench/`](docs/bench/).

## Table of contents

- [Why it exists](#why-it-exists)
- [At a glance](#at-a-glance)
- [What it serves (MI50 and V100)](#what-it-serves-mi50-and-v100)
- [Quick start](#quick-start)
- [Features](#features)
- [Built-in web UI (`--webui`)](#built-in-web-ui---webui)
- [Prefill shortcut: host prefix snapshots (P8)](#prefill-shortcut-host-prefix-snapshots-p8)
- [Configuration](#configuration)
- [Build (per backend)](#build-per-backend)
- [Deployment examples](#deployment-examples)
- [Benchmark summary](#benchmark-summary)
- [Architecture](#architecture)
- [Limitations](#limitations)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Why it exists

There were five requirements.

1. **Rust** — own the orchestration in Rust.
2. **GGUF** — keep using the existing model ecosystem as-is.
3. **Concurrency that is easier to reason about than llama.cpp** — make
   join/evict/priority a replaceable policy at iteration boundaries, and return an
   immediate 503 under overload instead of waiting.
4. **gfx906 (MI50)** — it has to run on a card whose ROCm support is shrinking.
5. **A single binary** — no Python, no launcher.

A pure-Rust GPU kernel could not be validated on gfx906, so the design settled on a
**hybrid**. Rust owns 100% of request lifetime, slots, policy, sampling, templates, and
the reasoning/tool parsers; only GPU math calls the SHA-pinned `libllama`/`libggml-*`
through unsafe FFI. That re-writing the same kernels in Rust does not speed up decode was
confirmed by [rocprof measurement](docs/bench/p6.md), so the kernel-rewrite phase is closed.

## At a glance

```
┌──────────────────────────────────────────────────────────┐
│  golbang-server  (tokio + axum, single binary)           │
│                                                          │
│  HTTP /v1/chat/completions                               │
│    → ChatML or minijinja                                 │
│    → bounded queue (immediate 503 + Retry-After if full) │
│    → Scheduler  (join / evict / chunked prefill)         │
│         │  slot changes only at iteration boundaries     │
│         ▼                                                │
│  Engine (Mutex<Model>, spawn_blocking)                   │
│    tokenize · llama_decode · sample · MTP · vision       │
│         │  unsafe FFI                                    │
│         ▼                                                │
│  golbang-sys  →  libllama.so + libggml-{hip,cuda,vulkan} │
└──────────────────────────────────────────────────────────┘
```

- **Orchestration = 100% Rust.** HTTP, slots, policy, sampling, templates, reasoning/tool parsers.
- **GPU kernels = proven llama.cpp.** Rust owns and calls them; the math is `ggml`.
- **Artifact = one binary.** The `.so` files are linked into the same process.

## What it serves (MI50 and V100)

This project does not stop at small demo models. Below is the list of current large and
MoE models actually running via the systemd units in `deploy/`. On MoE models the key to
VRAM budgeting is `--n-cpu-moe` (pin the expert weights of the first N blocks to CPU) — a
single value decides whether the model fits on two cards or OOMs.

### MI50 (gfx906, HIP / Vulkan)

| Model | Quantization / config | Unit |
|-------|-----------------------|------|
| Qwen3.8-27B | UD-Q4_K_XL, 100k ctx, KV q8_0, MTP+ngram speculative | `golbang-qwen38` (HIP), `golbang-qwen38-vulkan` (Vulkan) |
| Qwen3.8-27B | IST-DASLab GSQ-RCO IQ3_S (non-uniform, 11.8 GB) + MTP, 100k ctx, KV q8_0, `--n-ubatch 2048` | `golbang-qwen38-gsq` (:8084) |
| DeepSeek-V4-Flash-0731 | UD-IQ2_M, `--n-cpu-moe 32` | `golbang-deepseek` |
| GLM-5.3-Flash | AJ-IQ2_XXS, `--n-cpu-moe 42` | `golbang-glm53flash` |
| Qwen3.8-27B Uncensored | Q6_K | `golbang-qwen48` |

### V100 (sm_70, CUDA 12.8, 2× 16 GiB)

| Model | Quantization / config | Unit |
|-------|-----------------------|------|
| Qwen3.8-27B | UD-Q4_K_XL + MTP, `--tensor-split` | `golbang-cuda-qwen38` |
| Qwen3.8-Flash-Next | UD-Q4_K_XL, `--n-cpu-moe 42` + external MTP head (`draft-mtp` n-max 2), 100k ctx | `golbang-cuda-flashnext` |
| DeepSeek-V4.1 | native runtime (`GOLBANG_GPU=ds41-cuda`) | `golbang-server-ds41-cuda` |

Vulkan is a backend for running the same models through a different kernel path. CUDA and
HIP are each built separately; there is no runtime switch that lets one binary see both
GPUs (see [Build](#build-per-backend)).

## Quick start

### Prerequisites

| Item | Value |
|------|-------|
| Rust | edition 2024 (1.85+ recommended) |
| llama.cpp | `scripts/build-llama.sh <backend>` builds from a public base + [`patches/`](patches/) |
| GPU (HIP) | gfx906 (MI50) + ROCm. `HSA_OVERRIDE_GFX_VERSION=9.0.6` |
| GPU (CUDA) | sm_70+ (V100 uses CUDA 12.8 — CUDA 13 dropped compute_70) |
| GPU (Vulkan) | Any Vulkan device. `GGML_VULKAN=ON` cmake dir |

`golbang-sys/build.rs` **checks the git SHA and `.so` bytes** of the llama.cpp tree. If
you mix headers/libraries that differ from the pin, the build fails with a **hard error**
instead of silently breaking. The tree path is set with `GOLBANG_LLAMA_DIR` (default
`<repo>/vendor/<tree>`), and the cmake output path with `GOLBANG_LLAMA_BIN_DIR`. A tree
produced by `scripts/build-llama.sh` passes the pin check via a `.golbang-llama-pin`
marker, while the `.so` byte and header checks still run.

### Build and run

```bash
# 1) Reproducibly build the pinned llama.cpp core from public source (HIP here).
#    Fetch the published base commit by SHA, apply patches/, and cmake-build into vendor/.
scripts/build-llama.sh hip          # or cuda / vulkan / ds41 / ds41-cuda

# 2) Build golbang (the backend is chosen at build time by GOLBANG_GPU; no runtime switch)
CARGO_TARGET_DIR=target-hip GOLBANG_GPU=hip \
  cargo build -p golbang-server --release

# 3) Smoke-test with a small model
./target-hip/release/golbang-server \
  --model /path/to/Qwen3-0.6B-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8088 \
  --n-parallel 2 --queue-size 2 --policy fifo
```

```bash
# Streaming
curl -N -X POST http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Hello"}],
       "stream":true,"max_tokens":32}'

# Non-streaming
curl -sS -X POST http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Hello"}],
       "stream":false,"max_tokens":32}'

# Model card / metrics
curl -s http://127.0.0.1:8088/v1/models
curl -s http://127.0.0.1:8088/metrics | head
```

The default bind is `127.0.0.1`. When exposing it externally, always set **`--api-key`**
together with `--host 0.0.0.0` (the middleware checks `Authorization: Bearer` /
`X-Api-Key`), and terminate TLS at a reverse proxy. Keep keyless instances on the LAN only.

## Features

- OpenAI `POST /v1/chat/completions` — `stream:true` SSE, `stream:false` JSON
- OpenAI `GET /v1/models` (alias `GET /models`) — the one loaded model, with a
  llama-server-compatible `meta`
- **continuous batching** — slot pool + replaceable `SchedulePolicy` + chunked prefill.
  join/evict/cancel happen **only at iteration boundaries**, right after `llama_decode` returns
- **overload contract** — when the bounded queue is full it returns **an immediate 503 +
  `Retry-After: 1`** without waiting for decode (llama-server may wait in the same
  situation. That is a choice, not a bug)
- **multi-turn prefix cache** — slot-local LCP reuse + cross-slot prefix snapshots.
  The 2nd turn of a 3k-token conversation drops prefill to a few tokens
  ([P5 record](docs/bench/p5.md)). For host prefix snapshots extended beyond a slot, see
  [P8](#prefill-shortcut-host-prefix-snapshots-p8)
- chat templates — hardcoded Qwen ChatML (default) / GGUF `tokenizer.chat_template` via
  minijinja (`--jinja`) / `--chat-template-file` override
- **reasoning** — `--reasoning-format deepseek|auto` splits think tags into OpenAI
  `reasoning_content`. When a request does not give a thinking budget, it reserves
  `max(1024, 15% of max_tokens)` for the answer and force-closes think (halogen answer
  room). So even a small `max_tokens` does not end with empty `content`. It reads request
  fields `reasoning_budget`, `reasoning_effort`, `enable_thinking` and the thinking-budget
  aliases (`reasoning_budget_tokens`, `max_thinking_tokens`, `thinking_budget[_tokens]`,
  `thinking_token_budget`, `thinking.budget_tokens`, `reasoning.max_tokens`,
  `chat_template_kwargs.*`)
- **tool calls** — inject request `tools` into the template, re-parse DSML/Qwen/Hermes
  output into OpenAI `tool_calls`
- **vision** — `image_url` / `input_image` via `--mmproj` (`mtmd`)
- **speculative decoding** — `--spec-type draft-mtp,ngram-mod,prompt-lookup`. Verifies
  `[sampled, draft…]` against the target. PLD (prompt lookup) turns on only for greedy,
  solo generation, and appends `K` tokens after the previous occurrence of the last
  `N`-token suffix onto the MTP chain (the halogen way). It raises mean accepted length on
  copy-heavy turns, but on units where MTP is already saturated, like Flash-Next, the
  decode gain is around 1%, so it is off by default (`docs/bench/pld-flashnext.md`, issue #247)
- `/metrics` — Prometheus. Tokens, TTFT/ITL histograms, draft accept, slot occupancy, 503.
  It also exports `llamacpp:` aliases (`prompt_tokens_total`, `tokens_predicted_total`,
  `requests_processing`, `requests_deferred`, `prompt_tokens_cached_total`, etc.) with the
  same values so llama.cpp/llama-swap dashboards read them unmodified
- `--prompt-progress` (on by default) — attach a llama-server-style `prompt_progress` to
  SSE to keep the connection alive during long prefills
- **built-in web UI** (`--webui`, on by default) — serves a minimal smoke UI at `GET /`.
  The first screen is full-width chat, and API key setup and real-time inference speed
  (prefill/decode tok/s, cache reuse, queue, 503) based on `/metrics` are in the **tabs of
  the ⚙ settings popup**. Each response also shows that request's `timings`. The Svelte
  static build is embedded into the binary with `include_bytes!`, so no runtime files and
  no Node at build time are needed. Static HTML is served unauthenticated, but
  `/v1/chat/completions` still requires a key
- `--api-key` / `--alias` / `--rpc` / `--tensor-split` — same meaning as llama-server

**What is missing:** embeddings, rerank, scheduling policies other than `fifo`, pure-Rust
GPU kernels (closed after the P6 gate failed). The built-in web UI is only the minimum
needed for smoke checks; sessions, RAG, and multi-model are left to external clients like
Open WebUI.

## Built-in web UI (`--webui`)

Like llama-server, golbang keeps a "start the server and check in the browser" path. The
scope, however, is narrowed to a **smoke UI that confirms within 3 seconds that the model
is alive right after deployment**. The first load shows only the chat screen; everything
else lives in the tabs of the popup that opens when you press the **⚙ Settings** button in
the top right.

1. **Smoke chat** — full-width. Actually issues requests to move the counters, and shows
   each response's `timings` (prompt/predicted tok/s, cache hits, draft acceptance) and
   `reasoning_content`.
2. **API key setup** — stored only in browser `sessionStorage` and sent as
   `Authorization: Bearer …`. Cleared when the tab closes; never written to disk.
3. **Inference speed stats** — polls `/metrics` to show prefill/decode tok/s, cache reuse
   rate, draft acceptance rate, processing/queued, slot occupancy, 503, effective `n_ctx`,
   and queue depth.

The UI does not send `temperature`, `max_tokens`, or context length; it uses the server
configuration as-is. Sampling defaults are `--temperature` / `--top-p` / `--top-k`
(default `0.8` / `0.95` / `40`, llama-server parity), and they are applied when a request
omits the values. Before those defaults existed, llama.cpp's raw defaults (temperature 1.0,
top_k 0, top_p 1.0) were used as-is, so chat requests that did not specify a temperature
sampled across all 248k tokens and collapsed into a list of multilingual words.

The source is the Svelte + Vite app in [`webui/`](webui/), built by
`vite-plugin-singlefile` into a single `webui/dist/index.html` that is committed to the
repository. `golbang-server` embeds this file with `include_bytes!`, so **the final binary
build needs no Node.** Rebuild the UI only when changing it with
`cd webui && npm install && npm run build`, and commit `dist/index.html` along with it.
Disable the route with `--webui false` (or `GOLBANG_WEBUI=false`).

## Prefill shortcut: host prefix snapshots (P8)

Agents (tool loops) re-send the accumulated prompt every turn. Prefilling from zero each
turn recomputes 30k–70k tokens for minutes at a time. Where P5 computed "only the 2nd-turn
suffix" inside the same slot, **P8 extends that prefix path beyond the slot, across chains,
and past a watermark.** Not a single GPU kernel was changed (diff 0); only the Rust
scheduler was touched. It builds on `Engine::seq_state_get`/`set`
(`llama_state_seq_*_ext`, `PARTIAL_ONLY`) that P5 already used, with no new FFI.

| Track | What it does |
|-------|--------------|
| **P8-A** semantic anchor chain | Instead of overwriting the end-of-prefill snapshot with a single one, keep it as a chain in ascending length. A bind continues from the longest anchor with `n_tokens ≤ reuse_len+1` |
| **P8-B** slot-external common prefix | Redefine `PrefixStore` as a host `SeqCheckpoint` map. Keep tool/system heads (stub windows 6144–16384, stride 2048) globally, and when another session attaches to an empty slot, restore and then compute only the suffix |
| **P8-C** watermark survival | At 85% KV occupancy, `clear_seq` only the GPU sequence while keeping the host chain and store (HiCache L2) |

Constants: `CHAIN_MAX_ANCHORS=8`, `HOST_RAM_CAP=2 GiB`, stub windows 6144–16384 (stride 2048).
Utility confirmed in the production journal — Qwen3.8-27B restored a 33671-token request
from anchor 31767 and attached only the 1905-token suffix in 15.4s (4–5 minutes from zero).
DeepSeek-V4-Flash restored 34989 of 35805 tokens and computed only the 816-token suffix.
Journal markers are `prefix restored from checkpoint`, `host prefix snapshot promoted`,
`host snapshot restored`/`host snapshot miss`, and
`retained prefix kv released; host anchors kept`.

**Honest limitations (not exaggerated).** P8 does not eliminate every full prefill.

- Empty-slot restore (B) opens with `--n-parallel 2`. With `n_parallel=1` there is no
  empty slot, so only same-slot affinity (A) hits.
- Store promotion survives only if the first long prefill lands exactly on a stub-window
  boundary (6144/8192/12288…). DeepSeek units use `-b 5800`, which skips the boundary.
- Removing `tools` drops the common prefix to LCP ≈ 3787, outside the stub windows
  (6144–16384), so it recomputes in full.
- Watermark survival (C) is locked by code and unit tests, but in the observed production
  window the 85% watermark did not fire, so live confirmation is still pending.

The full rationale, constants, and journal table are in [`docs/bench/p8.md`](docs/bench/p8.md).

### Disk tier: session restore after restart (HAL-4 #248, experimental)

P8's snapshots are all in host RAM and vanish when the process restarts. Passing
`--prefix-cache-dir <DIR>` writes snapshots to `<DIR>/<key-hash>.ckpt` within an
`--prefix-cache-disk-gib` (default 64) LRU and, on startup, scans only directory headers to
reload the index. After a restart, when the same prefix attaches to an empty slot, it is
restored with `seq_state_full_set`.

- The key = **engine build + weight path/mtime/size + `n_ctx` + settings that change the
  stored bytes + token prefix**. A different fingerprint makes the file header ignored, so
  it misses (preventing silent wrong answers).
- Disk files are **full-state** (base KV + recurrent + indexer). `PARTIAL_ONLY` cannot
  continue from a fresh context without a non-SWA base.
- **A restore that needs a rollback is refused.** Only a continuation whose restored length
  is shorter than the prompt is restored; an exact re-send that has to rewind the last
  token falls back to full prefill. Reconstructing the QSA/indexer cells of a rewound token
  is impossible and would change the answer.
- Saving is off the request path (an existing capture point) and uses a temp file + rename.
- Without `--prefix-cache-dir` it stays RAM-only (P8). On tmpfs/ramfs/overlay it warns and
  falls back to RAM.

**Grade NUMERIC, off by default.** Flash-Next measurement (2× V100, 9965 tokens): after a
restart, a continuation prompt restored with byte-identical message to cold, and TTFT went
from 42.3s to 10.6s (about 4×), reproduced twice. However, in one exact re-send the output
diverged because the MTP draft state was not in the snapshot. So it is left opt-in rather
than enabled on production units. A follow-up that also stores the MTP draft context is
needed. The full rationale is in the wiki `p8-host-prefix-snapshots`.

### KV admission reservation + pool-fit (HAL-5 #249)

halogen reserves **prompt + max_tokens positions** at request admission in the shared KV
pool, and makes requests wait in arrival order if they do not fit. golbang counted only
`used`, so a request that began a long generation could hit the pool ceiling mid-flight and
fail mid-generation.

- **Reservation:** if the cap `cap_for_join` provides cannot hold `prompt + max_tokens`,
  the request is not attached to a slot but kept at the front of the waiting queue (arrival
  order). A request whose `prompt + max_tokens` cannot fit under any cap (larger than the
  pool) is clamped and proceeds as before to avoid starvation. If the wait exceeds the
  timeout (`--timeout-secs`), it is failed with `Timeout`.
- **pool-fit:** with `--kv-pool-fit`, on a load failure
  (`llama_init_from_model returned null`) it retries by halving `n_ubatch`, and then lowers
  `n_ctx` if that still fails. The lowered values are left in the log and in the
  `/metrics` `golbang_pool_*` gauges. Without the flag, it is a clear startup error.
- Observability: `golbang_joins_deferred_total`, `golbang_joins_clamped_total`,
  `golbang_queue_timeouts_total`, `golbang_pool_ctx_effective`,
  `golbang_pool_ubatch_effective`, `golbang_pool_fit_downgrades_total`.

Flash-Next measurement (2× V100, MTP, `--kv-unified`, n_parallel 2): two long generations
(each prompt 6052, completion 1500) both completed with HTTP 200 simultaneously, and there
was no `failed to find a memory slot`. The second request, on which the reservation
applied, increased `joins_deferred_total` and waited with `requests_deferred 1`. pool-fit
downgraded to `(100000,2048)` after a `--n-ubatch 4096` load failure and started. The full
rationale is in the wiki `hal5-kv-admission-poolfit`.

## Configuration

Most flags are identical to `GOLBANG_*` environment variables. `--help` is the final
source of truth; below is a summary of the frequently used ones.

| Flag | Default | Meaning |
|------|---------|---------|
| `--model` | `GOLBANG_MODEL` | GGUF path (falls back to `GOLBANG_TEST_MODEL` for tests) |
| `--host` / `--port` | `127.0.0.1` / `8088` | Bind (defaults that do not collide with llama-server's `:8080`) |
| `--n-ctx` | `256` | **Default share per slot.** Total KV pool = `n_ctx × n_parallel` |
| `--n-parallel` | `2` | Number of slots, i.e. llama `n_seq_max` |
| `--single-max-ctx` | `0` = whole pool | Upper bound for a solo slot. When a second slot attaches, the pool is re-divided |
| `--kv-unified` | off | llama shared KV stream. Needed for a solo to use the whole pool |
| `--queue-size` | `2` | Waiting queue. 503 when full |
| `--n-gpu-layers` | `99` | GPU offload layers |
| `--n-cpu-moe` | `0` | Pin expert weights of the first N blocks to CPU (MoE VRAM budgeting) |
| `--n-batch` / `--n-ubatch` | `0` = auto | Logical batch / physical ubatch |
| `--n-threads` | `0` | llama decode threads (0 = library default) |
| `--n-rs-seq` | `1` | Number of recurrent snapshots. Auto-raised when MTP is on |
| `--flash-attn` | `auto` | `auto` \| `on` \| `off` |
| `--no-mmap` | off | `LLAMA_LOAD_MODE_NONE` |
| `--alias` | file name | Response `model` field |
| `--jinja` | off | Apply the GGUF template via minijinja |
| `--chat-template-file` | none | Override the GGUF template |
| `--reasoning-format` | `none` | `none` \| `deepseek` \| `deepseek-legacy` \| `auto` |
| `--reasoning-effort` / `--reasoning-budget` | template default / answer room | think control. When a request does not specify a budget, reserve `max(1024, 15%)` for the answer |
| `--temperature` / `--top-p` / `--top-k` | `0.8` / `0.95` / `40` | Server sampling defaults (llama-server parity). Request fields override each |
| `--mmproj` | none | CLIP/projector GGUF (vision) |
| `--spec-type` | empty | `draft-mtp`, `ngram-mod`, `prompt-lookup`/`pld` (comma-separated) |
| `--spec-draft-n-max` / `--spec-draft-p-min` | `3` / `0.90` | MTP draft cap / minimum probability |
| `--spec-pld-n` / `--spec-pld-k` | `3` / `3` | prompt lookup suffix length / continuation draft length (greedy+solo only) |
| `--policy` | `fifo` | Only `fifo` is implemented. The trait is replaceable |
| `--timeout-secs` | none | Request generation time limit |
| `--api-key` | none | Repeat or comma-separated. `/metrics` `/models` are exempt |
| `--prompt-progress` | on | SSE `prompt_progress` events |
| `--prefix-cache-dir` | none | prefix snapshot disk tier. Session restore after restart (HAL-4 #248) |
| `--prefix-cache-disk-gib` | `64` | Disk tier LRU cap (GiB). `0` disables |
| `--kv-pool-fit` | off | On load failure, retry by halving `n_ubatch` → `n_ctx` (HAL-5 #249) |
| `--rpc` / `--tensor-split` | none | Same meaning as llama-server (ggml-rpc offload) |

Request fields: `temperature`, `top_p`, `top_k`, `max_tokens`, `seed`, `stop`, `tools`,
`tool_choice`, `reasoning_effort`, `reasoning_budget`, `enable_thinking`, `image_url`,
`return_progress`. The thinking budget also accepts the aliases above and nested objects
(`thinking`, `reasoning`, `chat_template_kwargs`).

## Build (per backend)

**CUDA and HIP are built separately — there is no unified build.**
`GOLBANG_GPU` decides the llama.cpp tree, SHA pin, and linked library, and
`CARGO_TARGET_DIR` separates the artifacts. There is **no** `--gpu` runtime flag that lets
one binary see both GPUs (intentional design).

| Backend | `GOLBANG_GPU` | Links against | Notes |
|---------|---------------|---------------|-------|
| HIP | `hip` (default) | `libggml-hip.so` | Includes gfx906 byte checks |
| CUDA | `cuda` | `libggml-cuda.so` | V100 uses CUDA 12.8 |
| Vulkan | `vulkan` | `libggml-vulkan.so` | Same tree, `build-vulkan/` cmake dir |
| DSV4.1 runtime | `ds41` / `ds41-cuda` | Same recipe as above | Separate DeepSeek-V4.1 native-pointer tree |

`cargo test --workspace` runs SSE / JSON / empty-messages 4xx / 503-during-decode when
`GOLBANG_TEST_MODEL` (a small GGUF) is present, and skips otherwise.

### Where the core comes from

- **Source build (recommended)** — `scripts/build-llama.sh <backend>` fetches the
  published base commit by SHA, applies [`patches/`](patches/), and cmake-builds into
  `vendor/<tree>/`. `hip`/`vulkan` use `glm5next` + the gfx906 furnace port, `ds41` uses
  vcruz305 `runtime/deepseek41` + the gfx906 port, and `cuda` is upstream + the qwen4exp
  MTP patch. Each patch's base SHA, provenance, and apply order are in
  [`patches/README.md`](patches/README.md).
- **Binary tarball** — `scripts/package-release.sh <backend>` collects `golbang-server`
  and the matching llama.cpp/ggml `.so` files into `lib/`, and produces a relocatable
  `dist/golbang-<ver>-<backend>-<arch>.tar.gz` with an `$ORIGIN/lib` rpath. With only the
  ROCm/CUDA driver installed, the user runs `./run.sh --model …` (the GPU runtime is not
  statically bundled).

The current pinned llama.cpp SHA is the `EXPECTED_SHA_*` constants in `golbang-sys/build.rs`.
A tree reproduced by `build-llama.sh` passes the pin check via a `.golbang-llama-pin`
marker; any other HEAD is a hard error (bypassed only with `GOLBANG_LLAMA_ALLOW_DRIFT=1`).
When raising the pin, update the base/patches and `EXPECTED_SHA_*` together. Do not mix
headers and `.so` files from a sibling tree (another branch/fork).

## Deployment examples

`deploy/` has example systemd units. Environment-specific values (absolute paths, model
names, keys) have been removed, so fill them in for your setup.

- The units are mutually `Conflicts=` — only one per GPU
- API keys are injected via `EnvironmentFile=-/etc/golbang/secrets.env`
  (see `deploy/secrets.env.example`. **Do not leave plaintext keys in unit files**)
- Common HIP environment: `HSA_OVERRIDE_GFX_VERSION=9.0.6`,
  `LD_LIBRARY_PATH=<llama.cpp build/bin>`

## Benchmark summary

Measurements are not hidden; all of them are committed. Raw JSON is in `docs/bench/raw/`,
and reproduction scripts are in `scripts/`. The table below states only numbers; the point
of comparison is the single sentence in the [intro](#golbang).

| Comparison | Result | Document |
|------------|--------|----------|
| vs llama-server (MI50, DSV4-Flash IQ2_M) | decode ≈ 8 t/s. 660-token prefill at parity; 2540 tokens favors llama due to `-ub` difference. 4-way overload: golbang suite wall 6.4s vs llama 11.5s (503 immediate rejection) | [`golbang-vs-llama-server.md`](docs/bench/golbang-vs-llama-server.md) |
| vs llama-server (RTX 3060) | decode 13.2–13.4 t/s, prefill golbang **1.4–1.7×** | [`golbang-vs-llama-cuda.md`](docs/bench/golbang-vs-llama-cuda.md) |
| P7 Qwen3.8-27B decode band | 16/200-token band met, 84/long-form band within ±3% | [`p7.md`](docs/bench/p7.md) |
| Multi-turn prefix cache (P5) | 3k-token re-request wall 34s → 4s | [`p5.md`](docs/bench/p5.md) |
| Host prefix snapshots (P8) | 30k–70k full prefill reduced to a suffix of hundreds–thousands of tokens (see [P8](#prefill-shortcut-host-prefix-snapshots-p8) above for limits) | [`p8.md`](docs/bench/p8.md) |

**Using llama-server directly may be better for you.** golbang is not a GPU-kernel
competition; it is a project for testing a Rust-written scheduler and control plane. For
anyone who needs a pure llama.cpp deployment, this repository is not an alternative.

## Architecture

The workspace has three crates. unsafe exists only in the FFI calls of `golbang-sys` and
`golbang-core`.

| Crate | Role |
|-------|------|
| **golbang-server** | axum HTTP, SSE/JSON, `/v1/models`, `/metrics`, API key middleware |
| **golbang-core** | scheduler, slots, policy, engine, templates, sampler, prefix cache, speculative, vision, tools |
| **golbang-sys** | `llama.h` / `mtmd.h` / `llama-ext.h` bindgen + C++ shim. Exposes only the C ABI |

The path of one request:

1. **HTTP** — inspect `messages` and apply the template. `image_url` becomes a marker +
   bytes.
2. **Submit** — `try_submit` is non-blocking. 503 when the queue is full.
3. **join** — attach to an empty slot **only right after this decode returns**.
4. **bind** — after tokenization, reuse the LCP with the slot prefix. The reused span is
   not prefilled. If it does not match in the slot, widen the view with host prefix
   snapshots — slot-local anchor chain (P8-A), empty-slot restore from the global
   `PrefixStore` (P8-B). Only when both miss is it a full prefill
   ([P8](#prefill-shortcut-host-prefix-snapshots-p8)).
5. **plan** — decoding slots first, then prefill in the remaining space. Do not mix a large
   prefill and a decode into one `llama_decode` (`mixed_prefill_max`).
6. **GPU** — decode → sample → MTP draft under the `Mutex<Model>` of a single
   `spawn_blocking` worker.
7. **Emit** — SSE `delta` or JSON. The reasoning/tool parsers cut at tag boundaries.

Detailed implementation notes are staged in [`docs/ROADMAP.md`](docs/ROADMAP.md) and
`docs/work-orders/`.

## Limitations

- **Same kernels, same ceiling.** The bottleneck of MoE models is not the GPU kernel but
  CPU expert offload (`n-cpu-moe`).
- **Long-prefill configuration.** On cards tight on VRAM, `--n-ubatch` must be lowered, so
  long prefills can be slower than llama-server's large `-ub`.
- **P8 prefix restore is conditional.** Empty-slot restore opens with `--n-parallel 2`,
  promotion survives only if the first long prefill lands on a stub-window boundary, and
  removing `tools` drops the common prefix outside the window, forcing a full recompute.
  See [P8](#prefill-shortcut-host-prefix-snapshots-p8) for details and unconfirmed conditions.
- **Only FIFO scheduling.** The trait is opened for replacement, but there are no other
  implementations.
- **Reasoning output grade is NUMERIC.** The answer room and forced think close change the
  timing at which `</think>` closes (= the output branch), so it is **not byte-identical**
  to serial greedy. A request that specifies a budget takes precedence, and with a small
  `max_tokens` it reduces think to 1 token and leaves room for the answer.
- **Vision prefix.** Image requests clear the slot KV (no prefix hit).
- **Single model.** One model per process. A multi-model gateway is out of scope.
- **The built-in web UI is for smoke tests.** No session storage, RAG, multi-model, or tool
  editing. Leave such work to external clients like Open WebUI. UI assets are a committed
  build artifact; rebuild with Node only when changing them.
- **A gfx906 laboratory artifact.** Testing outside MI50/V100 generations (3090, MI210,
  etc.) is insufficient.

## Documentation

| Location | Contents |
|----------|----------|
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | P0–P7 phases and completion criteria (checkboxes) |
| [`docs/work-orders/`](docs/work-orders/) | Per-phase execution orders |
| [`docs/bench/`](docs/bench/) | Performance records + `raw/` source JSON |
| [`docs/bench/p8.md`](docs/bench/p8.md) | Host prefix snapshots: background, A/B/C, constants, production journal, honest limits |
| [`scripts/`](scripts/) | `build-llama.sh` (core reproducible build), `package-release.sh` (binary tarball), benchmark reproduction scripts |
| [`patches/`](patches/) | llama.cpp pin patches + base SHA provenance (`patches/README.md`) |
| [`deploy/`](deploy/) | Example systemd units |
| [`README_KO.md`](README_KO.md) | Korean README |

## Contributing

Issues and PRs are welcome. Rust code must pass `cargo fmt --all` and `cargo clippy`, and
changes touching the FFI boundary should explain the SHA pin check in
`golbang-sys/build.rs`. Opening an **issue first** before a new feature is recommended to
align direction — this project makes keeping the serving surface narrow an explicit goal.

## License

MIT OR Apache-2.0 (dual license, same as the workspace `license` in `Cargo.toml`). It is
the same license family as llama.cpp, compatible with GGUF ecosystem practice.

---

> ***"But we have this treasure in jars of clay, to show that the surpassing power
> belongs to God and not to us."*** — 2 Corinthians 4:7
