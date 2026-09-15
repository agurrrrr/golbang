# golbang

> **골뱅이** — AMD MI50(gfx906)과 NVIDIA V100으로 **최신 대형·MoE 모델을 서빙**하는
> GGUF LLM 추론 서버.
> 서빙·스케줄·배칭은 Rust, GPU 수학은 SHA 고정된 llama.cpp(ggml)을 C ABI로 호출한다.
> 서빙 표면은 OpenAI 호환 채팅 API 한 개로 최소한에 집중한다.

*OpenAI-compatible GGUF inference server. Rust orchestration (axum + tokio) on top of
SHA-pinned llama.cpp backends (HIP / CUDA / Vulkan), built to serve current large and
MoE models (Qwen3.8, DeepSeek-V4-Flash, GLM-5.3-Flash) on aging hardware like gfx906
that mainstream stacks are leaving behind. Single binary, no Python.*

![Rust](https://img.shields.io/badge/Rust-edition%202024-dea584?logo=rust&logoColor=white)
![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)
![API](https://img.shields.io/badge/API-OpenAI%20chat%20completions-6e56cf)
![GPU](https://img.shields.io/badge/backends-HIP%20%C2%B7%20CUDA%20%C2%B7%20Vulkan-orange)

llama-server와 **같은 GPU 커널**을 쓰기 때문에 decode 속도는 대체로 동률이다.
golbang이 노리는 지점은 처리량이 아니라 **스케줄 정책·취소·즉시 503·경량 제어면**이고,
주 무대는 지원이 얇아지는 **gfx906(MI50)와 sm_70(V100)** 처럼 오래된 카드로
**최신 대형·MoE 모델을 올리는 것**이다.
무엇을 실제로 서빙하는지는 [무엇을 서빙하나](#무엇을-서빙하나-mi50와-v100)에,
긴 에이전트 대화의 prefill을 줄이는 경로는 [프리필 단축: 호스트 접두 스냅샷 (P8)](#프리필-단축-호스트-접두-스냅샷-p8)에 적어뒀다.
실측은 [한계](#한계)와 [`docs/bench/`](docs/bench/)에 좋은 숫자만 골라 쓰지 않고 있는 그대로 적어뒀다.

## 목차

- [왜 만들었나](#왜-만들었나)
- [한눈에](#한눈에)
- [무엇을 서빙하나 (MI50와 V100)](#무엇을-서빙하나-mi50와-v100)
- [빠른 시작](#빠른-시작)
- [기능](#기능)
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

요구는 다섯 가지였다.

1. **Rust** — 오케스트레이션을 Rust로 소유하고 싶다.
2. **GGUF** — 기존 모델 생태계를 그대로 쓴다.
3. **llama.cpp보다 다루기 쉬운 동시성** — join/evict/우선순위를 iteration 경계에서
   교체 가능한 정책으로 만들고, 과부하에는 대기 없이 즉시 503을 낸다.
4. **gfx906 (MI50)** — ROCm 지원이 줄어드는 카드에서도 돌아야 한다.
5. **단일 바이너리** — 파이썬/런처 없이.

순수 Rust GPU 커널은 gfx906에서 검증할 방법이 없어 **하이브리드**로 확정했다.
Rust가 요청 수명·슬롯·정책·샘플링·템플릿·reasoning/tool 파서를 100% 소유하고,
GPU 연산만 SHA 고정된 `libllama`/`libggml-*`를 unsafe FFI로 호출한다.
같은 커널을 Rust로 다시 써도 decode는 빨라지지 않는다는 것은
[rocprof 실측](docs/bench/p6.md)으로 확인했고, 그래서 커널 재작성 단계는 닫았다.

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

- **오케스트레이션 = 100% Rust.** HTTP, 슬롯, 정책, 샘플링, 템플릿, reasoning/tool 파서.
- **GPU 커널 = 검증된 llama.cpp.** Rust가 소유·호출하되 수식은 `ggml`이다.
- **산출물 = 바이너리 하나.** `.so`는 같은 프로세스에 링크한다.

## 무엇을 서빙하나 (MI50와 V100)

이 프로젝트는 데모용 소형 모델로 멈추지 않는다. 아래는 `deploy/`의 systemd 유닛으로
실제 돌리고 있는 최신 대형·MoE 모델 목록이다. MoE 모델에서는 VRAM 배분의 열쇠가
`--n-cpu-moe`(앞 N개 블록의 expert 가중치를 CPU에 고정)다 — 이 값 하나로 카드 두 장의
한계에 모델이 들어가냐 OOM이냐가 갈린다.

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
| Qwen3.8-Flash-Next | UD-Q4_K_XL, `--n-cpu-moe 48` (expert 전부 CPU), 100k ctx | `golbang-cuda-flashnext` |
| DeepSeek-V4.1 | 네이티브 런타임 (`GOLBANG_GPU=ds41-cuda`) | `golbang-server-ds41-cuda` |

Vulkan은 같은 모델을 다른 커널 경로로 돌려볼 수 있는 백엔드다. CUDA와 HIP은
각각 따로 빌드하며, 하나의 바이너리가 두 GPU를 모두 보는 런타임 스위치는 없다
([빌드](#빌드-백엔드별) 참조).

## 빠른 시작

### 준비물

| 항목 | 값 |
|------|-----|
| Rust | edition 2024 (1.85+ 권장) |
| llama.cpp | 아래 [빌드](#빌드-백엔드별)의 핀된 SHA로 직접 빌드 |
| GPU (HIP) | gfx906 (MI50) + ROCm. `HSA_OVERRIDE_GFX_VERSION=9.0.6` |
| GPU (CUDA) | sm_70+ (V100은 CUDA 12.8 — CUDA 13은 compute_70 미지원) |
| GPU (Vulkan) | Any Vulkan device. `GGML_VULKAN=ON` cmake dir |

`golbang-sys/build.rs`가 llama.cpp 트리의 **git SHA와 `.so` 바이트를 검사**한다.
핀과 다른 헤더/라이브러리를 섞으면 조용히 깨지는 대신 빌드가 **hard error**로 실패한다.
트리 경로는 `GOLBANG_LLAMA_DIR`, cmake 산출물 경로는 `GOLBANG_LLAMA_BIN_DIR`으로 지정한다.

### 빌드와 실행

```bash
# 1) 핀된 llama.cpp을 백엔드별로 빌드해둔다 (예: HIP)
git -C "$GOLBANG_LLAMA_DIR" checkout <핀 SHA>
cmake -B build -DGGML_HIP=ON && cmake --build build --config Release

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

기본 바인드는 `127.0.0.1`이다. 외부에 열 때는 `--host 0.0.0.0`과 함께
**`--api-key`를 반드시 걸고**(미들웨어가 `Authorization: Bearer` / `X-Api-Key`를 검사),
TLS은 리버스 프록시에서 종결하는 것을 권한다. 키 없는 인스턴스는 LAN 전용으로 두자.

## 기능

- OpenAI `POST /v1/chat/completions` — `stream:true` SSE, `stream:false` JSON
- OpenAI `GET /v1/models` (별칭 `GET /models`) — 로드된 모델 한 개, llama-server 호환 `meta`
- **continuous batching** — 슬롯 풀 + 교체 가능한 `SchedulePolicy` + chunked prefill.
  join/evict/cancel은 `llama_decode` 반환 직후 **iteration 경계에서만** 일어난다
- **과부하 계약** — bounded 큐가 가득이면 decode를 기다리지 않고 **즉시 503 + `Retry-After: 1`**
  (llama-server는 같은 상황에서 대기할 수 있다. 버그가 아니라 선택이다)
- **다턴 prefix cache** — 슬롯 로컬 LCP 재사용 + 크로스 슬롯 접두 스냅샷.
  3k 토큰 대화의 2턴째 prefill이 수 토큰으로 줄어든다 ([P5 기록](docs/bench/p5.md)).
  슬롯 밖으로 넓힌 호스트 접두 스냅샷은 [P8](#프리필-단축-호스트-접두-스냅샷-p8) 참조
- 채팅 템플릿 — Qwen ChatML 하드코딩(기본) / GGUF `tokenizer.chat_template`을 minijinja로(`--jinja`) /
  `--chat-template-file` 재정의
- **reasoning** — `--reasoning-format deepseek|auto`로 think 태그를 OpenAI `reasoning_content`로 분리.
  `reasoning_effort`(CLI + 요청 필드)와 `--reasoning-budget`(초과 시 think 종료 강제) 지원
- **tool calls** — 요청 `tools`를 템플릿에 주입, DSML/Qwen/Hermes 출력을 OpenAI `tool_calls`로 재파싱
- **vision** — `--mmproj`로 `image_url` / `input_image` (`mtmd`)
- **speculative decoding** — `--spec-type draft-mtp,ngram-mod`. 타깃에서 `[sampled, draft…]` 검증
- `/metrics` — Prometheus. 토큰, TTFT/ITL 히스토그램, draft accept, 슬롯 점유, 503
- `--prompt-progress`(기본 on) — SSE에 llama-server식 `prompt_progress`를 실어 긴 프리필 동안 연결 유지
- `--api-key` / `--alias` / `--rpc` / `--tensor-split` — llama-server와 같은 의미

**없는 것:** embeddings, rerank, `fifo` 이외 스케줄 정책, 순수 Rust GPU 커널(P6 gate 실패로 닫힘).

## 프리필 단축: 호스트 접두 스냅샷 (P8)

에이전트(툴 루프)는 턴마다 지금까지의 프롬프트를 누적해 다시 보낸다. 매 턴을 0부터
prefill하면 30k~70k 토큰을 몇 분씩 다시 계산한다. P5가 같은 슬롯 안에서 "2턴째 접미만"
계산했다면, **P8은 그 접두 경로를 슬롯 밖·체인·워터마크 이후로 늘린다.** GPU 커널은
한 줄도 고치지 않았고(diff 0), Rust 스케줄러만 건드렸다. 바탕은 이미 P5가 쓰던
`Engine::seq_state_get`/`set`(`llama_state_seq_*_ext`, `PARTIAL_ONLY`)이고 새 FFI는 없다.

| 갈래 | 하는 일 |
|------|---------|
| **P8-A** 의미 앵커 체인 | 프리필 끝 스냅샷을 한 장으로 덮어쓰지 않고 길이 오름차순 체인으로 남긴다. bind는 `n_tokens ≤ reuse_len+1`인 가장 긴 앵커에서 이어 간다 |
| **P8-B** 슬롯 밖 공통 접두 | `PrefixStore`를 호스트 `SeqCheckpoint` 맵으로 재정의. 도구/시스템 head(스텁 창 6144–16384, 스트라이드 2048)를 전역에 두고, 다른 세션이 빈 슬롯에 붙으면 복원 후 접미만 |
| **P8-C** 워터마크 생존 | 85% KV 점유에서 GPU 시퀀스만 `clear_seq`하고 호스트 체인·스토는 남긴다 (HiCache L2) |

상수: `CHAIN_MAX_ANCHORS=8`, `HOST_RAM_CAP=2 GiB`, 스텁 창 6144–16384(스트라이드 2048).
생산 저널에서 확인된 효용 — Qwen3.8-27B는 33671토큰 요청을 앵커 31767에서 복원한 뒤
접미 1905토큰만 15.4초에 붙였다(0부터면 4~5분). DeepSeek-V4-Flash는 35805토큰 중
34989를 복원하고 접미 816만 돌렸다. 저널 판별 문구는 `prefix restored from checkpoint`,
`host prefix snapshot promoted`, `host snapshot restored`/`host snapshot miss`,
`retained prefix kv released; host anchors kept`다.

**정직한 한계(과장하지 않는다).** P8은 모든 전량 prefill을 없애지 않는다.

- 빈 슬롯 복원(B)은 `--n-parallel 2`에서 열린다. `n_parallel=1`이면 빈 슬롯이 없어 같은 슬롯 affinity(A)만 히트한다.
- 스토 승격은 첫 긴 prefill이 스텁 창 경계(6144/8192/12288…)에 정확히 착지해야 남는다. DeepSeek 유닛은 `-b 5800`이라 경계를 건너뛴다.
- `tools`를 빼면 공통 접두가 LCP 약 3787로 스텁 창(6144–16384) 밖으로 떨어져 전량으로 다시 돈다.
- 워터마크 생존(C)은 코드와 단위 테스트로 잠겼으나, 관측한 생산 창에서 85% 워터마크가 발화하지 않아 라이브 확인은 아직이다.

전체 근거·상수·저널 표는 [`docs/bench/p8.md`](docs/bench/p8.md)에 있다.

## 설정

대부분의 플래그는 `GOLBANG_*` 환경 변수와 동일하다. `--help`가 최종 진실이고,
아래는 자주 쓰는 것들의 요약이다.

| 플래그 | 기본 | 의미 |
|--------|------|------|
| `--model` | `GOLBANG_MODEL` | GGUF 경로 (테스트용 `GOLBANG_TEST_MODEL` 폴백) |
| `--host` / `--port` | `127.0.0.1` / `8088` | 바인드 (llama-server `:8080`와 겹치지 않는 기본값) |
| `--n-ctx` | `256` | **슬롯당 기본 몫.** 총 KV 풀 = `n_ctx × n_parallel` |
| `--n-parallel` | `2` | 슬롯 수이자 llama `n_seq_max` |
| `--single-max-ctx` | `0` = 풀 전체 | 솔로 슬롯의 상한. 두 번째 슬롯이 붙으면 풀을 다시 나눈다 |
| `--kv-unified` | off | llama 공유 KV 스트림. 솔로가 풀 전체를 쓰려면 필요 |
| `--queue-size` | `2` | 대기 큐. 가득 차면 503 |
| `--n-gpu-layers` | `99` | GPU 오프로드 레이어 |
| `--n-cpu-moe` | `0` | 앞 N개 블록의 expert 가중치를 CPU에 고정 (MoE VRAM 배분) |
| `--n-batch` / `--n-ubatch` | `0` = auto | 논리 배치 / 물리 ubatch |
| `--n-threads` | `0` | llama decode 스레드 (0 = 라이브러리 기본) |
| `--n-rs-seq` | `1` | recurrent snapshot 수. MTP 켜지면 자동 상향 |
| `--flash-attn` | `auto` | `auto` \| `on` \| `off` |
| `--no-mmap` | off | `LLAMA_LOAD_MODE_NONE` |
| `--alias` | 파일명 | 응답 `model` 필드 |
| `--jinja` | off | GGUF 템플릿을 minijinja로 적용 |
| `--chat-template-file` | 없음 | GGUF 템플릿 재정의 |
| `--reasoning-format` | `none` | `none` \| `deepseek` \| `deepseek-legacy` \| `auto` |
| `--reasoning-effort` / `--reasoning-budget` | 템플릿 기본 / 무제한 | think 제어 |
| `--mmproj` | 없음 | CLIP/projector GGUF (vision) |
| `--spec-type` | 빈 값 | `draft-mtp`, `ngram-mod` (콤마) |
| `--spec-draft-n-max` / `--spec-draft-p-min` | `3` / `0.90` | MTP draft 상한 / 최소 확률 |
| `--policy` | `fifo` | 현재 `fifo`만 구현. trait은 교체 가능 |
| `--timeout-secs` | 없음 | 요청 생성 시간 제한 |
| `--api-key` | 없음 | 반복 또는 콤마. `/metrics` `/models`는 면제 |
| `--prompt-progress` | on | SSE `prompt_progress` 이벤트 |
| `--rpc` / `--tensor-split` | 없음 | llama-server와 동일 의미 (ggml-rpc 오프로드) |

요청 필드: `temperature`, `top_p`, `top_k`, `max_tokens`, `seed`, `stop`,
`tools`, `tool_choice`, `reasoning_effort`, `reasoning_budget`, `image_url`, `return_progress`.

## 빌드 (백엔드별)

**CUDA와 HIP은 각각 빌드한다 — 통합 빌드는 하지 않는다.**
`GOLBANG_GPU`가 llama.cpp 트리·SHA 핀·링크 라이브러리를 결정하고,
`CARGO_TARGET_DIR`로 산출물을 분리한다. 하나의 바이너리가 두 GPU를 모두 보는
`--gpu` 런타임 플래그는 **없다** (의도된 설계).

| 백엔드 | `GOLBANG_GPU` | 링크 대상 | 비고 |
|--------|---------------|-----------|------|
| HIP | `hip` (기본) | `libggml-hip.so` | gfx906 바이트 검사 포함 |
| CUDA | `cuda` | `libggml-cuda.so` | V100은 CUDA 12.8 |
| Vulkan | `vulkan` | `libggml-vulkan.so` | 같은 트리, `build-vulkan/` cmake dir |
| DSV4.1 런타임 | `ds41` / `ds41-cuda` | 위와 동일 레시피 | DeepSeek-V4.1 네이티브 포인터 트리 별도 |

`cargo test --workspace`는 `GOLBANG_TEST_MODEL`(소형 GGUF)이 있을 때
SSE / JSON / 빈 messages 4xx / decode 중 503까지 돌고, 없으면 스킵한다.

현재 pinned llama.cpp SHA는 `golbang-sys/build.rs`의 `EXPECTED_SHA_*` 상수가 진실이다.
업스트림 따라잡기는 워크플로 일부다 — 핀을 올릴 때는 해당 SHA로 트리를 재빌드한 뒤
상수만 바꾼다. 형제 트리(다른 브랜치/fork)의 헤더와 `.so`를 섞지 말 것.

## 배포 예시

`deploy/`에 systemd 유닛 예시가 있다. 실사용 환경 맞춤값(절대경로·모델명·키)은
제거했으니 자기 환경에 맞게 채우면 된다.

- 유닛들은 서로 `Conflicts=` — 한 장의 GPU에는 하나만
- API 키는 `EnvironmentFile=-/etc/golbang/secrets.env`로 주입한다
  (`deploy/secrets.env.example` 참고. **유닛 파일에 평문 키를 남기지 말 것**)
- HIP 공통 환경: `HSA_OVERRIDE_GFX_VERSION=9.0.6`, `LD_LIBRARY_PATH=<llama.cpp build/bin>`

## 벤치마크 요약

측정을 숨기지 않고 전부 커밋한다. raw JSON은 `docs/bench/raw/`, 재현 스크립트는 `scripts/`다.
아래 표는 숫자만 적고, 비교 취지는 [도입부](#golbang)의 한 문장으로 갈음한다.

| 비교 | 결과 | 문서 |
|------|------|------|
| vs llama-server (MI50, DSV4-Flash IQ2_M) | decode ~8 t/s. 660토큰 prefill 동급, 2540토큰은 `-ub` 차이로 llama 우위. 과부하 4-way는 golbang 스위트 wall 6.4s vs llama 11.5s (503 즉시 거절) | [`golbang-vs-llama-server.md`](docs/bench/golbang-vs-llama-server.md) |
| vs llama-server (RTX 3060) | decode 13.2–13.4 t/s, prefill은 golbang **1.4–1.7×** | [`golbang-vs-llama-cuda.md`](docs/bench/golbang-vs-llama-cuda.md) |
| P7 Qwen3.8-27B decode 밴드 | 16/200토큰 밴드 충족, 84/장문 밴드는 ±3% 내외 | [`p7.md`](docs/bench/p7.md) |
| 다턴 prefix cache (P5) | 3k 토큰 재요청 wall 34s → 4s | [`p5.md`](docs/bench/p5.md) |
| 호스트 접두 스냅샷 (P8) | 30k~70k 전량 prefill을 접미 수백~수천 토큰으로 (한계는 위 [P8](#프리필-단축-호스트-접두-스냅샷-p8) 참조) | [`p8.md`](docs/bench/p8.md) |

**llama-server를 그냥 쓰는 게 더 나을 수도 있다.** golbang은 GPU 커널 경쟁이 아니라
Rust로 쓴 스케줄러와 제어면을 테스트하는 프로젝트다. 순수 llama.cpp 배포가 필요한
사람에게 이 저장소는 대안이 아니다.

## 아키텍처

워크스페이스는 세 크레이트. unsafe는 `golbang-sys`와 `golbang-core`의 FFI 호출에만 있다.

| 크레이트 | 역할 |
|----------|------|
| **golbang-server** | axum HTTP, SSE/JSON, `/v1/models`, `/metrics`, API 키 미들웨어 |
| **golbang-core** | 스케줄러, 슬롯, 정책, 엔진, 템플릿, 샘플러, prefix cache, speculative, vision, tools |
| **golbang-sys** | `llama.h` / `mtmd.h` / `llama-ext.h` bindgen + C++ shim. C ABI만 노출 |

한 요청의 경로:

1. **HTTP** — `messages`를 검사하고 템플릿을 입힌다. `image_url`은 마커 + 바이트로 바뀐다.
2. **제출** — `try_submit`은 non-blocking. 큐 가득지면 503.
3. **join** — 빈 슬롯에, **이번 decode 반환 직후에만** 붙인다.
4. **bind** — 토큰화 후 슬롯 prefix와 LCP 재사용. 재사용 구간은 prefill하지 않는다.
   슬롯에서 맞지 않으면 호스트 접두 스냅샷으로 시야를 넓힌다 — 슬롯 로컬 앵커 체인(P8-A),
   전역 `PrefixStore`에서 빈 슬롯 복원(P8-B). 어느 쪽도 미스일 때만 전량 prefill이다
   ([P8](#프리필-단축-호스트-접두-스냅샷-p8)).
5. **plan** — decoding 슬롯 우선, 남는 칸에 prefill. 큰 prefill과 decode를 한
   `llama_decode`에 섞지 않는다 (`mixed_prefill_max`).
6. **GPU** — `spawn_blocking` 단일 워커의 `Mutex<Model>`에서 decode → 샘플 → MTP draft.
7. **방출** — SSE `delta` 또는 JSON. reasoning/tool 파서가 태그 단위로 자른다.

상세 구현 메모는 [`docs/ROADMAP.md`](docs/ROADMAP.md)와 `docs/work-orders/`에 단계별로 있다.

## 한계

- **같은 커널, 같은 천장.** MoE 모델의 병목은 GPU 커널이 아니라 CPU expert 오프로드(`n-cpu-moe`)다.
- **장문 prefill 설정.** VRAM이 빠듯한 카드에서는 `--n-ubatch`를 낮춰야 해서
  llama-server의 큰 `-ub`보다 긴 prefill이 느릴 수 있다.
- **P8 접두 복원은 조건부.** 빈 슬롯 복원은 `--n-parallel 2`에서 열리고, 승격은 첫 긴
  prefill이 스텁 창 경계에 착지해야 남으며, `tools`를 빼면 공통 접두가 창 밖으로
  떨어져 전량으로 다시 돈다. 세부와 미확인 조건은 [P8](#프리필-단축-호스트-접두-스냅샷-p8) 참조.
- **스케줄 정책은 FIFO뿐.** trait은 교체 가능하게 열어뒀지만 추가 구현이 없다.
- **비전 prefix.** 이미지 요청은 슬롯 KV를 비운다 (prefix hit 없음).
- **단일 모델.** 프로세스당 모델 하나. 멀티 모델 게이트웨이는 범위 밖이다.
- **gfx906 실험실 산물.** MI50/V100 외 세대(3090, MI210 등)에서는 테스트가 부족하다.

## 문서

| 위치 | 내용 |
|------|------|
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | P0–P7 단계와 완료 기준 (체크박스) |
| [`docs/work-orders/`](docs/work-orders/) | 단계별 실행 지시서 |
| [`docs/bench/`](docs/bench/) | 성능 기록 + `raw/` 원본 JSON |
| [`docs/bench/p8.md`](docs/bench/p8.md) | 호스트 접두 스냅샷: 배경·A/B/C·상수·생산 저널·정직한 한계 |
| [`scripts/`](scripts/) | 벤치 재현 스크립트 (표준 라이브러리만 사용) |
| [`deploy/`](deploy/) | systemd 유닛 예시 |

## 기여

Issue와 PR을 환영한다. Rust 코드는 `cargo fmt --all`과 `cargo clippy`를 통과해야 하고,
FFI 경계를 건드리는 변경은 `golbang-sys/build.rs`의 SHA 핀 검사를 함께 설명해달라.
새 기능보다 **먼저 issue**를 열어 방향을 맞추는 것을 권한다 — 이 프로젝트는
서빙 표면을 좁게 유지하는 것을 명시적 목표로 한다.

## 라이선스

MIT OR Apache-2.0 (dual license, `Cargo.toml`의 workspace `license`와 동일).
llama.cpp와 동일 라이선스 계열이라 GGUF 생태계 관행과 호환된다.
