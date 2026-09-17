# golbang

[English](README.md) · **한국어**

> **골뱅이(golbang)** 는 AMD MI50(gfx906)과 NVIDIA V100에서 **최신 대형·MoE 모델을 서빙**하는
> GGUF LLM 추론 서버입니다.
> 서빙·스케줄·배칭은 Rust가 맡고, GPU 수학은 SHA로 고정한 llama.cpp(ggml)를 C ABI로 호출합니다.
> 서빙 표면은 OpenAI 호환 채팅 API 하나로 최소한을 유지합니다.

*OpenAI 호환 GGUF 추론 서버입니다. SHA로 고정한 llama.cpp 백엔드(HIP / CUDA / Vulkan) 위에
axum과 tokio로 Rust 오케스트레이션을 올렸습니다. 주류 스택이 떠나가는 gfx906 같은 노후
하드웨어에서 최신 대형·MoE 모델(Qwen3.8, DeepSeek-V4-Flash, GLM-5.3-Flash)을 서빙하기 위해
만들었습니다. 단일 바이너리이며 파이썬이 필요하지 않습니다.*

![Rust](https://img.shields.io/badge/Rust-edition%202024-dea584?logo=rust&logoColor=white)
![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)
![API](https://img.shields.io/badge/API-OpenAI%20chat%20completions-6e56cf)
![GPU](https://img.shields.io/badge/backends-HIP%20%C2%B7%20CUDA%20%C2%B7%20Vulkan-orange)

golbang은 llama-server와 **같은 GPU 커널**을 사용하므로 decode 속도는 대체로 비슷합니다.
golbang이 겨냥하는 지점은 처리량이 아니라 **스케줄 정책, 취소, 즉시 503, 가벼운 제어면**이며,
주 무대는 지원이 줄어드는 **gfx906(MI50)과 sm_70(V100)** 같은 오래된 카드에
**최신 대형·MoE 모델을 올리는 일**입니다.
무엇을 실제로 서빙하는지는 [무엇을 서빙하나](#무엇을-서빙하나-mi50와-v100)에 적었고,
긴 에이전트 대화의 prefill을 줄이는 경로는 [프리필 단축: 호스트 접두 스냅샷 (P8)](#프리필-단축-호스트-접두-스냅샷-p8)에 적었습니다.
실측값은 [한계](#한계)와 [`docs/bench/`](docs/bench/)에 좋은 숫자만 골라 쓰지 않고 있는 그대로 적었습니다.

## 목차

- [왜 만들었나](#왜-만들었나)
- [한눈에](#한눈에)
- [무엇을 서빙하나 (MI50와 V100)](#무엇을-서빙하나-mi50와-v100)
- [빠른 시작](#빠른-시작)
- [기능](#기능)
- [내장 웹 UI (`--webui`)](#내장-웹-ui---webui)
- [프리필 단축: 호스트 접두 스냅샷 (P8)](#프리필-단축-호스트-접두-스냅샷-p8)
- [설정](#설정)
- [빌드 (백엔드별)](#빌드-백엔드별)
- [배포 예시](#배포-예시)
- [벤치마크 요약](#벤치마크-요약)
- [아키텍처](#아키텍처)
- [한계](#한계)
- [문서](#문서)
- [기여](#기여)
- [라이선스](#라이선스)

## 왜 만들었나

처음에 세운 요구는 다섯 가지였습니다.

1. **Rust** — 오케스트레이션을 Rust로 직접 소유하고 싶었습니다.
2. **GGUF** — 기존 모델 생태계를 그대로 활용하고 싶었습니다.
3. **llama.cpp보다 다루기 쉬운 동시성** — join/evict/우선순위를 iteration 경계에서
   교체 가능한 정책으로 만들고, 과부하에는 기다리지 않고 즉시 503을 반환하고자 했습니다.
4. **gfx906(MI50)** — ROCm 지원이 줄어드는 카드에서도 동작해야 했습니다.
5. **단일 바이너리** — 파이썬과 런처 없이 배포하고 싶었습니다.

순수 Rust GPU 커널은 gfx906에서 검증할 방법이 없어 **하이브리드** 구조로 확정했습니다.
Rust가 요청 수명, 슬롯, 정책, 샘플링, 템플릿, reasoning/tool 파서를 100% 소유하고,
GPU 연산만 SHA로 고정한 `libllama`/`libggml-*`를 unsafe FFI로 호출합니다.
같은 커널을 Rust로 다시 작성해도 decode가 빨라지지 않는다는 사실은
[rocprof 실측](docs/bench/p6.md)으로 확인했고, 그래서 커널 재작성 단계는 닫았습니다.

## 한눈에

```
┌──────────────────────────────────────────────────────────┐
│  golbang-server  (tokio + axum, 단일 바이너리)            │
│                                                          │
│  HTTP /v1/chat/completions                               │
│    → ChatML 또는 minijinja                               │
│    → bounded 큐 (가득 차면 즉시 503 + Retry-After)        │
│    → Scheduler  (join / evict / chunked prefill)         │
│         │  iteration 경계에서만 슬롯 변경                 │
│         ▼                                                │
│  Engine (Mutex<Model>, spawn_blocking)                   │
│    tokenize · llama_decode · sample · MTP · vision       │
│         │  unsafe FFI                                    │
│         ▼                                                │
│  golbang-sys  →  libllama.so + libggml-{hip,cuda,vulkan} │
└──────────────────────────────────────────────────────────┘
```

- **오케스트레이션은 100% Rust입니다.** HTTP, 슬롯, 정책, 샘플링, 템플릿, reasoning/tool 파서를 모두 Rust가 맡습니다.
- **GPU 커널은 검증된 llama.cpp입니다.** Rust가 소유하고 호출하되 수식은 `ggml`이 담당합니다.
- **산출물은 바이너리 하나입니다.** `.so` 파일은 같은 프로세스에 링크합니다.

## 무엇을 서빙하나 (MI50와 V100)

이 프로젝트는 데모용 소형 모델에서 멈추지 않습니다. 아래는 `deploy/`의 systemd 유닛으로
실제로 돌리고 있는 최신 대형·MoE 모델 목록입니다. MoE 모델에서는 VRAM 배분의 핵심이
`--n-cpu-moe`(앞 N개 블록의 expert 가중치를 CPU에 고정)입니다. 이 값 하나로 두 장의 카드에
모델이 들어가는지 OOM이 나는지가 갈립니다.

### MI50 (gfx906, HIP / Vulkan)

| 모델 | 양자화·구성 | 유닛 |
|------|-------------|------|
| Qwen3.8-27B | UD-Q4_K_XL, 100k ctx, KV q8_0, MTP+ngram speculative | `golbang-qwen38` (HIP), `golbang-qwen38-vulkan` (Vulkan) |
| Qwen3.8-27B | IST-DASLab GSQ-RCO IQ3_S(non-uniform, 11.8 GB)+MTP, 100k ctx, KV q8_0, `--n-ubatch 2048` | `golbang-qwen38-gsq` (:8084) |
| DeepSeek-V4-Flash-0731 | UD-IQ2_M, `--n-cpu-moe 32` | `golbang-deepseek` |
| GLM-5.3-Flash | AJ-IQ2_XXS, `--n-cpu-moe 42` | `golbang-glm53flash` |
| Qwen3.8-27B Uncensored | Q6_K | `golbang-qwen48` |

### V100 (sm_70, CUDA 12.8, 2× 16 GiB)

| 모델 | 양자화·구성 | 유닛 |
|------|-------------|------|
| Qwen3.8-27B | UD-Q4_K_XL + MTP, `--tensor-split` | `golbang-cuda-qwen38` |
| Qwen3.8-Flash-Next | UD-Q4_K_XL, `--n-cpu-moe 42` + 외부 MTP head (`draft-mtp` n-max 2), 100k ctx | `golbang-cuda-flashnext` |
| DeepSeek-V4.1 | 네이티브 런타임 (`GOLBANG_GPU=ds41-cuda`) | `golbang-server-ds41-cuda` |

Vulkan은 같은 모델을 다른 커널 경로로 실행해 볼 수 있는 백엔드입니다. CUDA와 HIP은
각각 따로 빌드하며, 하나의 바이너리가 두 GPU를 모두 인식하는 런타임 스위치는 없습니다
([빌드](#빌드-백엔드별) 참조).

## 빠른 시작

### 준비물

| 항목 | 값 |
|------|-----|
| Rust | edition 2024 (1.85+ 권장) |
| llama.cpp | `scripts/build-llama.sh <backend>`가 공개 base + [`patches/`](patches/)로 빌드 |
| GPU (HIP) | gfx906 (MI50) + ROCm. `HSA_OVERRIDE_GFX_VERSION=9.0.6` |
| GPU (CUDA) | sm_70+ (V100은 CUDA 12.8 — CUDA 13은 compute_70 미지원) |
| GPU (Vulkan) | Any Vulkan device. `GGML_VULKAN=ON` cmake dir |

`golbang-sys/build.rs`는 llama.cpp 트리의 **git SHA와 `.so` 바이트를 검사**합니다.
핀과 다른 헤더나 라이브러리를 섞으면 조용히 깨지는 대신 빌드가 **hard error**로 실패합니다.
트리 경로는 `GOLBANG_LLAMA_DIR`(기본 `<repo>/vendor/<tree>`), cmake 산출물 경로는
`GOLBANG_LLAMA_BIN_DIR`로 지정합니다. `scripts/build-llama.sh`가 만든 트리는
`.golbang-llama-pin` 마커로 핀 검사를 통과하며, `.so` 바이트와 헤더 검사는 그대로 수행합니다.

### 빌드와 실행

```bash
# 1) 핀된 llama.cpp 코어를 공개 소스에서 재현 빌드한다 (예: HIP).
#    게시된 base 커밋을 SHA로 받아 patches/를 적용하고 cmake로 빌드 → vendor/.
scripts/build-llama.sh hip          # 또는 cuda / vulkan / ds41 / ds41-cuda

# 2) golbang 빌드 (백엔드는 GOLBANG_GPU 빌드 타임 결정. 런타임 스위치 없음)
CARGO_TARGET_DIR=target-hip GOLBANG_GPU=hip \
  cargo build -p golbang-server --release

# 3) 소형 모델 스모크
./target-hip/release/golbang-server \
  --model /path/to/Qwen3-0.6B-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8088 \
  --n-parallel 2 --queue-size 2 --policy fifo
```

```bash
# 스트리밍
curl -N -X POST http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Hello"}],
       "stream":true,"max_tokens":32}'

# 비스트리밍
curl -sS -X POST http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Hello"}],
       "stream":false,"max_tokens":32}'

# 모델 카드 / 메트릭
curl -s http://127.0.0.1:8088/v1/models
curl -s http://127.0.0.1:8088/metrics | head
```

기본 바인드 주소는 `127.0.0.1`입니다. 외부에 공개할 때는 `--host 0.0.0.0`과 함께
**`--api-key`를 반드시 설정하고**(미들웨어가 `Authorization: Bearer` / `X-Api-Key`를 검사),
TLS는 리버스 프록시에서 종료하기를 권장합니다. 키가 없는 인스턴스는 LAN 전용으로 두십시오.

## 기능

- OpenAI `POST /v1/chat/completions` — `stream:true`는 SSE, `stream:false`는 JSON으로 응답합니다.
- OpenAI `GET /v1/models` (별칭 `GET /models`) — 로드한 모델 한 개를 llama-server 호환 `meta`와 함께 제공합니다.
- **continuous batching** — 슬롯 풀과 교체 가능한 `SchedulePolicy`, chunked prefill을 사용합니다.
  join/evict/cancel은 `llama_decode`가 반환된 직후인 **iteration 경계에서만** 일어납니다.
- **과부하 계약** — bounded 큐가 가득 차면 decode를 기다리지 않고 **즉시 503과 `Retry-After: 1`** 을 반환합니다.
  llama-server는 같은 상황에서 대기할 수 있지만, 이는 버그가 아니라 선택입니다.
- **다턴 prefix cache** — 슬롯 로컬 LCP 재사용과 크로스 슬롯 접두 스냅샷을 사용합니다.
  3k 토큰 대화의 2턴째 prefill이 몇 토큰으로 줄어듭니다 ([P5 기록](docs/bench/p5.md)).
  슬롯 밖으로 넓힌 호스트 접두 스냅샷은 [P8](#프리필-단축-호스트-접두-스냅샷-p8)을 참조하십시오.
- 채팅 템플릿 — Qwen ChatML 하드코딩(기본), GGUF `tokenizer.chat_template`의 minijinja 적용(`--jinja`),
  `--chat-template-file` 재정의를 지원합니다.
- **reasoning** — `--reasoning-format deepseek|auto`로 think 태그를 OpenAI `reasoning_content`로 분리합니다.
  요청이 thinking budget을 주지 않으면 `max(1024, max_tokens의 15%)`를 답변용으로 남기고
  think를 강제로 종료합니다(halogen answer room). 따라서 작은 `max_tokens`에서도 빈 `content`로
  끝나지 않습니다. 요청 필드 `reasoning_budget`, `reasoning_effort`, `enable_thinking`과
  thinking budget 별칭(`reasoning_budget_tokens`, `max_thinking_tokens`, `thinking_budget[_tokens]`,
  `thinking_token_budget`, `thinking.budget_tokens`, `reasoning.max_tokens`,
  `chat_template_kwargs.*`)을 읽습니다.
- **tool calls** — 요청의 `tools`를 템플릿에 주입하고, DSML/Qwen/Hermes 출력을 OpenAI `tool_calls`로 재파싱합니다.
- **vision** — `--mmproj`로 `image_url` / `input_image`를 처리합니다 (`mtmd`).
- **speculative decoding** — `--spec-type draft-mtp,ngram-mod,prompt-lookup`을 지원합니다. 타깃에서 `[sampled, draft…]`를 검증합니다.
  PLD(prompt lookup)는 greedy 단독 생성일 때만 켜지고, 마지막 `N`토큰 접미의 이전 출현 뒤 `K`토큰을
  MTP 사슬 뒤에 이어 붙입니다(halogen 방식). 복사가 많은 턴에서는 mean accepted length가 늘지만,
  Flash-Next처럼 MTP가 이미 포화된 유닛에서는 decode 이득이 1% 정도라 기본적으로 꺼져 있습니다
  (`docs/bench/pld-flashnext.md`, 이슈 #247).
- `/metrics` — Prometheus 형식입니다. 토큰, TTFT/ITL 히스토그램, draft accept, 슬롯 점유, 503을 노출합니다.
  `llamacpp:` 별칭(`prompt_tokens_total`, `tokens_predicted_total`, `requests_processing`,
  `requests_deferred`, `prompt_tokens_cached_total` 등)도 같은 값으로 내보내므로 llama.cpp/llama-swap
  대시보드가 수정 없이 읽습니다.
- `--prompt-progress`(기본 on) — SSE에 llama-server식 `prompt_progress`를 실어 긴 프리필 동안 연결을 유지합니다.
- **내장 웹 UI** (`--webui`, 기본 on) — `GET /`에서 최소 스모크 UI를 제공합니다. 첫 화면은
  채팅 전폭이고, API 키 설정과 `/metrics` 기반 실시간 추론 속도(prefill/decode tok/s, 캐시
  재사용률, 대기열, 503)는 **⚙ 설정 버튼 팝업의 탭**에 있습니다. 응답마다 그 요청의 `timings`도
  표시합니다. Svelte 정적 빌드를 `include_bytes!`로 바이너리에 넣으므로 런타임 파일도, 빌드 시
  Node도 필요하지 않습니다. 정적 HTML은 비인증으로 공개되지만 `/v1/chat/completions`는 여전히 키를
  요구합니다.
- `--api-key` / `--alias` / `--rpc` / `--tensor-split` — llama-server와 같은 의미입니다.

**없는 것:** embeddings, rerank, `fifo` 이외 스케줄 정책, 순수 Rust GPU 커널(P6 gate 실패로 닫힘)입니다.
내장 웹 UI는 스모크 확인용 최소 기능뿐이고, 세션·RAG·멀티모델은 Open WebUI 같은 외부 클라이언트에 맡깁니다.

## 내장 웹 UI (`--webui`)

llama-server처럼 "서버를 띄우면 브라우저에서 바로 확인"하는 경로를 golbang에도 두었습니다. 다만
범위는 **배포 직후 모델이 살아 있는지 3초 안에 확인하는 스모크 UI**로 좁혔습니다. 첫 접속에는
채팅 화면만 보이고, 나머지는 우측 상단 **⚙ 설정** 버튼을 눌렀을 때 뜨는 팝업의 탭에서 확인합니다.

1. **스모크 채팅** — 전폭 화면입니다. 요청을 실제로 보내 통계를 움직여 보고, 응답마다 `timings`
   (prompt/predicted tok/s, 캐시 히트, 드래프트 수락)와 `reasoning_content`를 표시합니다.
2. **API 키 설정** — 브라우저 `sessionStorage`에만 저장하고 `Authorization: Bearer …`로
   전달합니다. 탭을 닫으면 지워지고 디스크에는 남지 않습니다.
3. **추론 속도 통계** — `/metrics`를 폴링해 prefill/decode tok/s, 캐시 재사용률, 드래프트 수락률,
   처리 중/대기열, 슬롯 점유, 503, 유효 `n_ctx`, 큐 깊이를 보여 줍니다.

`temperature`, `max_tokens`, context length는 UI가 보내지 않고 서버 설정값을 그대로 사용합니다.
샘플링 기본값은 `--temperature` / `--top-p` / `--top-k`(기본 `0.8` / `0.95` / `40`, llama-server
parity)이고, 요청이 값을 주지 않으면 이 기본값이 적용됩니다. 이 기본값이 없던 시절에는
llama.cpp의 raw 기본값(temperature 1.0, top_k 0, top_p 1.0)이 그대로 사용되어, 온도를 명시하지
않은 채팅 요청이 24.8만 토큰 전체에서 표집되어 다국어 단어 나열로 무너졌습니다.

원본은 [`webui/`](webui/)에 있는 Svelte + Vite 앱이고, `vite-plugin-singlefile`로 단일
`webui/dist/index.html`을 만들어 리포지토리에 커밋합니다. `golbang-server`는 이 파일을
`include_bytes!`로 포함하므로 **최종 바이너리 빌드에는 Node가 필요하지 않습니다.** UI를 수정할 때만
`cd webui && npm install && npm run build`로 다시 빌드하고 `dist/index.html`을 함께 커밋합니다.
`--webui false`(또는 `GOLBANG_WEBUI=false`)로 라우트를 끌 수 있습니다.

## 프리필 단축: 호스트 접두 스냅샷 (P8)

에이전트(툴 루프)는 턴마다 지금까지의 프롬프트를 누적해 다시 보냅니다. 매 턴을 0부터
prefill하면 30k~70k 토큰을 몇 분씩 다시 계산하게 됩니다. P5가 같은 슬롯 안에서 "2턴째 접미만"
계산했다면, **P8은 그 접두 경로를 슬롯 밖과 체인, 워터마크 이후로 늘립니다.** GPU 커널은
한 줄도 고치지 않았고(diff 0), Rust 스케줄러만 수정했습니다. 바탕은 이미 P5가 사용하던
`Engine::seq_state_get`/`set`(`llama_state_seq_*_ext`, `PARTIAL_ONLY`)이고 새 FFI는 없습니다.

| 갈래 | 하는 일 |
|------|---------|
| **P8-A** 의미 앵커 체인 | 프리필 끝 스냅샷을 한 장으로 덮어쓰지 않고 길이 오름차순 체인으로 남깁니다. bind는 `n_tokens ≤ reuse_len+1`인 가장 긴 앵커에서 이어 갑니다 |
| **P8-B** 슬롯 밖 공통 접두 | `PrefixStore`를 호스트 `SeqCheckpoint` 맵으로 재정의합니다. 도구/시스템 head(스텁 창 6144–16384, 스트라이드 2048)를 전역에 두고, 다른 세션이 빈 슬롯에 붙으면 복원한 뒤 접미만 계산합니다 |
| **P8-C** 워터마크 생존 | 85% KV 점유에서 GPU 시퀀스만 `clear_seq`하고 호스트 체인과 스토어는 남깁니다 (HiCache L2) |

상수는 `CHAIN_MAX_ANCHORS=8`, `HOST_RAM_CAP=2 GiB`, 스텁 창 6144–16384(스트라이드 2048)입니다.
생산 저널에서 확인한 효용은 다음과 같습니다. Qwen3.8-27B는 33671토큰 요청을 앵커 31767에서
복원한 뒤 접미 1905토큰만 15.4초에 붙였습니다(0부터 계산하면 4~5분입니다). DeepSeek-V4-Flash는
35805토큰 중 34989를 복원하고 접미 816토큰만 계산했습니다. 저널 판별 문구는
`prefix restored from checkpoint`, `host prefix snapshot promoted`,
`host snapshot restored`/`host snapshot miss`,
`retained prefix kv released; host anchors kept`입니다.

**정직한 한계입니다(과장하지 않습니다).** P8이 모든 전량 prefill을 없애지는 않습니다.

- 빈 슬롯 복원(B)은 `--n-parallel 2`에서 열립니다. `n_parallel=1`이면 빈 슬롯이 없어 같은 슬롯 affinity(A)만 히트합니다.
- 스토어 승격은 첫 긴 prefill이 스텁 창 경계(6144/8192/12288…)에 정확히 착지해야 남습니다. DeepSeek 유닛은 `-b 5800`을 쓰므로 경계를 건너뜁니다.
- `tools`를 빼면 공통 접두가 LCP 약 3787로 스텁 창(6144–16384) 밖으로 떨어져 전량을 다시 계산합니다.
- 워터마크 생존(C)은 코드와 단위 테스트로 잠갔으나, 관측한 생산 창에서 85% 워터마크가 발화하지 않아 라이브 확인은 아직 하지 못했습니다.

전체 근거와 상수, 저널 표는 [`docs/bench/p8.md`](docs/bench/p8.md)에 있습니다.

### 디스크 tier: 재시작 후 세션 복원 (HAL-4 #248, 실험적)

P8의 스냅샷은 모두 호스트 RAM에 있어 프로세스가 재시작하면 사라집니다. `--prefix-cache-dir
<DIR>`을 지정하면 `--prefix-cache-disk-gib`(기본 64) LRU 범위 안에서 스냅샷을
`<DIR>/<key-hash>.ckpt`로 남기고, 기동할 때 디렉터리 헤더만 스캔해 인덱스를 다시 적재합니다.
재시작 뒤 같은 접두가 빈 슬롯에 붙으면 `seq_state_full_set`으로 복원합니다.

- 키는 **엔진 빌드 + 가중치 경로/mtime/size + `n_ctx` + 저장 바이트를 바꾸는 설정 +
  토큰 접두**입니다. 지문이 다르면 파일 헤더를 무시하고 미스로 처리합니다(조용한 오답 방지).
- 디스크 파일은 **full-state**(base KV + recurrent + indexer)입니다. `PARTIAL_ONLY`는
  fresh context에서 비-SWA base가 없어 이어받을 수 없습니다.
- **롤백이 필요한 복원은 거부합니다.** 복원 길이가 프롬프트보다 짧은 연속(continuation)
  일 때만 복원하고, 정확 재전송처럼 마지막 토큰을 되감아야 하면 전량 prefill로 폴백합니다.
  되감은 토큰의 QSA/indexer 셀을 재구성할 수 없어 답이 갈리기 때문입니다.
- 저장은 요청 경로 밖(기존 캡처 지점)에서 수행하고 temp 파일과 rename을 사용합니다.
- `--prefix-cache-dir`를 지정하지 않으면 RAM 전용(P8)을 그대로 유지합니다. tmpfs/ramfs/overlay이면 경고
  후 RAM으로 폴백합니다.

**등급은 NUMERIC이고 기본값은 off입니다.** Flash-Next 실측(2×V100, 9965토큰)에서 재시작 후 연속
프롬프트 복원은 cold와 message가 byte-identical이었고 TTFT가 42.3s에서 10.6s로(약 4배)
줄었습니다(2회 재현). 다만 exact 재전송 1회에서 MTP draft 상태가 스냅샷에 없어 출력이
갈렸습니다. 그래서 생산 유닛에는 켜지 않고 opt-in으로 둡니다. MTP draft context까지
저장하는 후속 작업이 필요합니다. 전체 근거는 위키 `p8-host-prefix-snapshots`에 있습니다.

### KV admission 예약 + pool-fit (HAL-5 #249)

halogen은 공유 KV 풀에서 요청 admission 시 **prompt + max_tokens position을 예약**하고,
감당하지 못하면 도착 순서로 대기시킵니다. golbang은 `used`만 계산해서 긴 생성을 시작한 요청이
도중에 풀 천장을 만나 mid-generation 실패를 낼 수 있었습니다.

- **예약:** `cap_for_join`이 준 cap이 `prompt + max_tokens`를 담지 못하면 슬롯에 붙이지
  않고 waiting 앞쪽(도착 순서)에 남깁니다. `prompt + max_tokens`가 어떤 cap으로도 들어가지
  못하는(풀보다 큰) 요청은 기아를 막기 위해 기존처럼 클램프해 진행합니다. 대기 중
  타임아웃(`--timeout-secs`)이 지나면 `Timeout`으로 실패시킵니다.
- **pool-fit:** `--kv-pool-fit`이면 로드 실패(`llama_init_from_model returned null`) 시
  `n_ubatch`를 절반씩 낮춰 재시도하고, 그래도 실패하면 `n_ctx`를 낮춥니다. 낮춘 값은
  로그와 `/metrics`의 `golbang_pool_*` 게이지로 남깁니다. 미지정이면 명확한 기동 에러입니다.
- 관측 지표: `golbang_joins_deferred_total`, `golbang_joins_clamped_total`,
  `golbang_queue_timeouts_total`, `golbang_pool_ctx_effective`, `golbang_pool_ubatch_effective`,
  `golbang_pool_fit_downgrades_total`.

Flash-Next 실측(2×V100, MTP, `--kv-unified`, n_parallel 2)에서는 긴 생성 2개(각 prompt 6052,
completion 1500)가 동시에 HTTP 200으로 완주했고 `failed to find a memory slot`이 발생하지 않았습니다.
예약이 걸리는 두 번째 요청은 `joins_deferred_total`이 증가하고 `requests_deferred 1`로
대기했습니다. pool-fit은 `--n-ubatch 4096` 로드 실패 후 `(100000,2048)`로 하향해 기동했습니다.
전체 근거는 위키 `hal5-kv-admission-poolfit`에 있습니다.

## 설정

대부분의 플래그는 `GOLBANG_*` 환경 변수와 동일합니다. `--help`가 최종 기준이고,
아래는 자주 사용하는 것들을 요약한 것입니다.

| 플래그 | 기본 | 의미 |
|--------|------|------|
| `--model` | `GOLBANG_MODEL` | GGUF 경로 (테스트용 `GOLBANG_TEST_MODEL` 폴백) |
| `--host` / `--port` | `127.0.0.1` / `8088` | 바인드 (llama-server `:8080`와 겹치지 않는 기본값) |
| `--n-ctx` | `256` | **슬롯당 기본 몫입니다.** 총 KV 풀 = `n_ctx × n_parallel` |
| `--n-parallel` | `2` | 슬롯 수이며 llama `n_seq_max`입니다 |
| `--single-max-ctx` | `0` = 풀 전체 | 솔로 슬롯의 상한입니다. 두 번째 슬롯이 붙으면 풀을 다시 나눕니다 |
| `--kv-unified` | off | llama 공유 KV 스트림입니다. 솔로가 풀 전체를 쓰려면 필요합니다 |
| `--queue-size` | `2` | 대기 큐입니다. 가득 차면 503 |
| `--n-gpu-layers` | `99` | GPU 오프로드 레이어 |
| `--n-cpu-moe` | `0` | 앞 N개 블록의 expert 가중치를 CPU에 고정 (MoE VRAM 배분) |
| `--n-batch` / `--n-ubatch` | `0` = auto | 논리 배치 / 물리 ubatch |
| `--n-threads` | `0` | llama decode 스레드 (0 = 라이브러리 기본) |
| `--n-rs-seq` | `1` | recurrent snapshot 수입니다. MTP가 켜지면 자동으로 상향합니다 |
| `--flash-attn` | `auto` | `auto` \| `on` \| `off` |
| `--no-mmap` | off | `LLAMA_LOAD_MODE_NONE` |
| `--alias` | 파일명 | 응답 `model` 필드 |
| `--jinja` | off | GGUF 템플릿을 minijinja로 적용합니다 |
| `--chat-template-file` | 없음 | GGUF 템플릿 재정의 |
| `--reasoning-format` | `none` | `none` \| `deepseek` \| `deepseek-legacy` \| `auto` |
| `--reasoning-effort` / `--reasoning-budget` | 템플릿 기본 / answer room | think를 제어합니다. 요청이 budget을 명시하지 않으면 `max(1024, 15%)`를 답변용으로 남깁니다 |
| `--temperature` / `--top-p` / `--top-k` | `0.8` / `0.95` / `40` | 서버 샘플링 기본값입니다(llama-server parity). 요청 필드가 각각 덮어씁니다 |
| `--mmproj` | 없음 | CLIP/projector GGUF (vision) |
| `--spec-type` | 빈 값 | `draft-mtp`, `ngram-mod`, `prompt-lookup`/`pld` (콤마) |
| `--spec-draft-n-max` / `--spec-draft-p-min` | `3` / `0.90` | MTP draft 상한 / 최소 확률 |
| `--spec-pld-n` / `--spec-pld-k` | `3` / `3` | prompt lookup 접미 길이 / 연속 초안 길이 (greedy·단독 시만) |
| `--policy` | `fifo` | 현재 `fifo`만 구현했습니다. trait은 교체할 수 있습니다 |
| `--timeout-secs` | 없음 | 요청 생성 시간 제한 |
| `--api-key` | 없음 | 반복하거나 콤마로 구분합니다. `/metrics`와 `/models`는 면제입니다 |
| `--prompt-progress` | on | SSE `prompt_progress` 이벤트 |
| `--prefix-cache-dir` | 없음 | prefix 스냅샷 디스크 tier. 재시작 후 세션 복원 (HAL-4 #248) |
| `--prefix-cache-disk-gib` | `64` | 디스크 tier LRU 상한(GiB)입니다. `0`이면 비활성화합니다 |
| `--kv-pool-fit` | off | 로드 실패 시 `n_ubatch`→`n_ctx` 순으로 절반씩 낮춰 재시도 (HAL-5 #249) |
| `--rpc` / `--tensor-split` | 없음 | llama-server와 동일 의미 (ggml-rpc 오프로드) |

요청 필드로는 `temperature`, `top_p`, `top_k`, `max_tokens`, `seed`, `stop`,
`tools`, `tool_choice`, `reasoning_effort`, `reasoning_budget`, `enable_thinking`,
`image_url`, `return_progress`를 받습니다. thinking budget은 위 별칭과 중첩 객체(`thinking`,
`reasoning`, `chat_template_kwargs`)로도 받습니다.

## 빌드 (백엔드별)

**CUDA와 HIP은 각각 빌드하며, 통합 빌드는 하지 않습니다.**
`GOLBANG_GPU`가 llama.cpp 트리와 SHA 핀, 링크 라이브러리를 결정하고,
`CARGO_TARGET_DIR`로 산출물을 분리합니다. 하나의 바이너리가 두 GPU를 모두 인식하는
`--gpu` 런타임 플래그는 **없습니다**(의도된 설계입니다).

| 백엔드 | `GOLBANG_GPU` | 링크 대상 | 비고 |
|--------|---------------|-----------|------|
| HIP | `hip` (기본) | `libggml-hip.so` | gfx906 바이트 검사 포함 |
| CUDA | `cuda` | `libggml-cuda.so` | V100은 CUDA 12.8 |
| Vulkan | `vulkan` | `libggml-vulkan.so` | 같은 트리, `build-vulkan/` cmake dir |
| DSV4.1 런타임 | `ds41` / `ds41-cuda` | 위와 동일 레시피 | DeepSeek-V4.1 네이티브 포인터 트리 별도 |

`cargo test --workspace`는 `GOLBANG_TEST_MODEL`(소형 GGUF)이 있으면
SSE, JSON, 빈 messages 4xx, decode 중 503까지 실행하고, 없으면 건너뜁니다.

### 코어를 어떻게 얻나

- **소스 빌드(권장)** — `scripts/build-llama.sh <backend>`가 게시된 base 커밋을
  SHA로 받아 [`patches/`](patches/)를 적용하고 `vendor/<tree>/`에 cmake로 빌드합니다.
  `hip`/`vulkan`은 `glm5next`와 gfx906 furnace 포트, `ds41`은 vcruz305
  `runtime/deepseek41`과 gfx906 포트, `cuda`는 업스트림과 qwen4exp MTP 패치를 사용합니다.
  각 패치의 base SHA와 출처, 적용 순서는 [`patches/README.md`](patches/README.md)에 있습니다.
- **바이너리 tarball** — `scripts/package-release.sh <backend>`가 `golbang-server`와
  매칭되는 llama.cpp/ggml `.so`를 `lib/`에 모아 `$ORIGIN/lib` 기준으로 재배치 가능한
  `dist/golbang-<ver>-<backend>-<arch>.tar.gz`를 만듭니다. 사용자는 ROCm/CUDA
  드라이버만 있으면 `./run.sh --model …`으로 실행합니다(GPU 런타임은 정적으로 묶지 않습니다).

현재 고정한 llama.cpp SHA는 `golbang-sys/build.rs`의 `EXPECTED_SHA_*` 상수가 기준입니다.
`build-llama.sh`로 재현한 트리는 `.golbang-llama-pin` 마커로 핀 검사를 통과하고,
그 밖의 HEAD는 hard error입니다(`GOLBANG_LLAMA_ALLOW_DRIFT=1`로만 우회합니다). 핀을 올릴 때는
base와 패치, `EXPECTED_SHA_*`를 함께 갱신합니다. 형제 트리(다른 브랜치나 fork)의 헤더와
`.so`를 섞지 마십시오.

## 배포 예시

`deploy/`에 systemd 유닛 예시가 있습니다. 실사용 환경에 맞춘 값(절대경로, 모델명, 키)은
제거했으므로 각자 환경에 맞게 채우시면 됩니다.

- 유닛들은 서로 `Conflicts=`이므로 한 장의 GPU에는 하나만 올릴 수 있습니다.
- API 키는 `EnvironmentFile=-/etc/golbang/secrets.env`로 주입합니다
  (`deploy/secrets.env.example` 참고. **유닛 파일에 평문 키를 남기지 마십시오**).
- HIP 공통 환경은 `HSA_OVERRIDE_GFX_VERSION=9.0.6`, `LD_LIBRARY_PATH=<llama.cpp build/bin>`입니다.

## 벤치마크 요약

측정값을 숨기지 않고 전부 커밋합니다. raw JSON은 `docs/bench/raw/`, 재현 스크립트는 `scripts/`에 있습니다.
아래 표는 숫자만 적었고, 비교 취지는 [도입부](#golbang)의 한 문장으로 갈음합니다.

| 비교 | 결과 | 문서 |
|------|------|------|
| vs llama-server (MI50, DSV4-Flash IQ2_M) | decode 약 8 t/s입니다. 660토큰 prefill은 동급이고, 2540토큰은 `-ub` 차이로 llama가 우위입니다. 과부하 4-way는 golbang 스위트 wall 6.4s 대 llama 11.5s입니다(503 즉시 거절) | [`golbang-vs-llama-server.md`](docs/bench/golbang-vs-llama-server.md) |
| vs llama-server (RTX 3060) | decode 13.2–13.4 t/s, prefill은 golbang **1.4–1.7×** | [`golbang-vs-llama-cuda.md`](docs/bench/golbang-vs-llama-cuda.md) |
| P7 Qwen3.8-27B decode 밴드 | 16/200토큰 밴드 충족, 84/장문 밴드는 ±3% 내외 | [`p7.md`](docs/bench/p7.md) |
| 다턴 prefix cache (P5) | 3k 토큰 재요청 wall 34s → 4s | [`p5.md`](docs/bench/p5.md) |
| 호스트 접두 스냅샷 (P8) | 30k~70k 전량 prefill을 접미 수백~수천 토큰으로 줄입니다(한계는 위 [P8](#프리필-단축-호스트-접두-스냅샷-p8) 참조) | [`p8.md`](docs/bench/p8.md) |

**llama-server를 그냥 사용하는 편이 더 나을 수도 있습니다.** golbang은 GPU 커널 경쟁이 아니라
Rust로 작성한 스케줄러와 제어면을 테스트하는 프로젝트입니다. 순수 llama.cpp 배포가 필요한
분께 이 저장소는 대안이 아닙니다.

## 아키텍처

워크스페이스는 세 크레이트로 구성됩니다. unsafe는 `golbang-sys`와 `golbang-core`의 FFI 호출에만 있습니다.

| 크레이트 | 역할 |
|----------|------|
| **golbang-server** | axum HTTP, SSE/JSON, `/v1/models`, `/metrics`, API 키 미들웨어 |
| **golbang-core** | 스케줄러, 슬롯, 정책, 엔진, 템플릿, 샘플러, prefix cache, speculative, vision, tools |
| **golbang-sys** | `llama.h` / `mtmd.h` / `llama-ext.h` bindgen + C++ shim. C ABI만 노출 |

한 요청이 지나는 경로는 다음과 같습니다.

1. **HTTP** — `messages`를 검사하고 템플릿을 적용합니다. `image_url`은 마커와 바이트로 바뀝니다.
2. **제출** — `try_submit`은 non-blocking입니다. 큐가 가득 차면 503을 반환합니다.
3. **join** — 빈 슬롯에 **이번 decode가 반환된 직후에만** 붙입니다.
4. **bind** — 토큰화 후 슬롯 prefix와 LCP를 재사용합니다. 재사용 구간은 prefill하지 않습니다.
   슬롯에서 맞지 않으면 호스트 접두 스냅샷으로 시야를 넓힙니다. 슬롯 로컬 앵커 체인(P8-A)과
   전역 `PrefixStore`의 빈 슬롯 복원(P8-B)을 사용합니다. 어느 쪽도 미스일 때만 전량 prefill입니다
   ([P8](#프리필-단축-호스트-접두-스냅샷-p8)).
5. **plan** — decoding 슬롯을 우선하고 남는 칸에 prefill을 배치합니다. 큰 prefill과 decode를 한
   `llama_decode`에 섞지 않습니다(`mixed_prefill_max`).
6. **GPU** — `spawn_blocking` 단일 워커의 `Mutex<Model>`에서 decode → 샘플 → MTP draft를 수행합니다.
7. **방출** — SSE `delta` 또는 JSON으로 내보냅니다. reasoning/tool 파서가 태그 단위로 자릅니다.

상세 구현 메모는 [`docs/ROADMAP.md`](docs/ROADMAP.md)와 `docs/work-orders/`에 단계별로 있습니다.

## 한계

- **같은 커널, 같은 천장입니다.** MoE 모델의 병목은 GPU 커널이 아니라 CPU expert 오프로드(`n-cpu-moe`)입니다.
- **장문 prefill 설정입니다.** VRAM이 빠듯한 카드에서는 `--n-ubatch`를 낮춰야 해서
  llama-server의 큰 `-ub`보다 긴 prefill이 느릴 수 있습니다.
- **P8 접두 복원은 조건부입니다.** 빈 슬롯 복원은 `--n-parallel 2`에서 열리고, 승격은 첫 긴
  prefill이 스텁 창 경계에 착지해야 남으며, `tools`를 빼면 공통 접두가 창 밖으로
  떨어져 전량을 다시 계산합니다. 세부와 미확인 조건은 [P8](#프리필-단축-호스트-접두-스냅샷-p8)을 참조하십시오.
- **스케줄 정책은 FIFO뿐입니다.** trait은 교체할 수 있게 열어두었지만 추가 구현은 없습니다.
- **reasoning 출력 등급은 NUMERIC입니다.** answer room과 think budget 강제 종료는 `</think>`가
  닫히는 시점(=출력 분기)을 바꾸므로 serial greedy와 **byte-identical이 아닙니다**. 요청이
  budget을 명시하면 그 값이 우선하고, 작은 `max_tokens`에서는 think를 1토큰으로 줄인 뒤
  답변 공간을 남깁니다.
- **비전 prefix입니다.** 이미지 요청은 슬롯 KV를 비웁니다(prefix hit이 없습니다).
- **단일 모델입니다.** 프로세스당 모델 하나이며, 멀티 모델 게이트웨이는 범위 밖입니다.
- **내장 웹 UI는 스모크용입니다.** 세션 저장, RAG, 멀티모델, 도구 편집은 없습니다. 그런 작업은 Open WebUI 같은
  외부 클라이언트에 맡깁니다. UI 자산은 커밋된 빌드 산출물이라 수정할 때만 Node로 다시 빌드합니다.
- **gfx906 실험실 산출물입니다.** MI50/V100 외 세대(3090, MI210 등)에서는 테스트가 부족합니다.

## 문서

| 위치 | 내용 |
|------|------|
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | P0–P7 단계와 완료 기준 (체크박스) |
| [`docs/work-orders/`](docs/work-orders/) | 단계별 실행 지시서 |
| [`docs/bench/`](docs/bench/) | 성능 기록 + `raw/` 원본 JSON |
| [`docs/bench/p8.md`](docs/bench/p8.md) | 호스트 접두 스냅샷: 배경·A/B/C·상수·생산 저널·정직한 한계 |
| [`scripts/`](scripts/) | `build-llama.sh`(코어 재현 빌드), `package-release.sh`(바이너리 tarball), 벤치 재현 스크립트 |
| [`patches/`](patches/) | llama.cpp 핀 패치 + base SHA provenance (`patches/README.md`) |
| [`deploy/`](deploy/) | systemd 유닛 예시 |

## 기여

Issue와 PR을 환영합니다. Rust 코드는 `cargo fmt --all`과 `cargo clippy`를 통과해야 하고,
FFI 경계를 수정하는 변경은 `golbang-sys/build.rs`의 SHA 핀 검사를 함께 설명해 주시기 바랍니다.
새 기능보다 **먼저 issue**를 열어 방향을 맞추는 것을 권장합니다. 이 프로젝트는
서빙 표면을 좁게 유지하는 것을 명시적 목표로 삼고 있습니다.

## 라이선스

MIT OR Apache-2.0 (dual license)이며, `Cargo.toml`의 workspace `license`와 동일합니다.
llama.cpp와 같은 라이선스 계열이라 GGUF 생태계 관행과 호환됩니다.
