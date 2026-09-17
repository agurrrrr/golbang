# golbang vs llama-server 종합 비교

> 측정일: **2026-08-14** (KST). 같은 호스트, 같은 모델, 같은 GPU, **같은 프롬프트 스위트**를 한쪽씩 켜서 잰 통제 A/B.
> 원본 JSON: `docs/bench/raw/golbang-2026-08-14.json`, `docs/bench/raw/llama-server-2026-08-14.json`
> 재현: `scripts/compare_golbang_llama.py`
> 결과는 사후에 목표치를 맞추지 않고 있는 그대로 적는다.

## 한 줄

**decode tok/s 로는 이기지 않는다. 같은 커널·같은 `n_cpu_moe=32` 천장이다.**
골방이 llama.cpp보다 나은 지점은 처리량이 아니라 **제어면, 소유권, 관측성, 그리고 그 천장을 32 GiB MI50에서 켜기까지 쌓인 서빙 레이어**다.
긴 프롬프트 전량 prefill은 오늘 측정에서 llama-server가 더 빨랐다 (`-ub 5800` vs `--n-ubatch 1024`).

---

## 1. 무엇을, 어떻게 쟀나

이전 비교(#299)는 다른 날의 실제 대화라 통제 A/B가 아니었다.
오늘은 생산 유닛을 **그대로** 한쪽씩 기동했다.

순서:

1. `golbang-deepseek.service` 가동 중 → 스위트 측정 → 중지
2. `llama-server.service` 기동 → 같은 스위트 측정 → 중지
3. 생산 유닛 `golbang-deepseek.service` 복구

양쪽 모두 `:8080`, 같은 API 키, `temperature=0`, `stream=false`, `max_tokens=32`(큐 프로브만 16).
프롬프트는 스크립트가 만든 고정 문단 반복 + 한국어 지시라 서버 간에 내용이 같다.

| 케이스 | 의도 |
|--------|------|
| short ×2 | 짧은 프롬프트 (`1+1=`). 2번째는 같은 프롬프트 prefix 재사용 |
| medium | ~660 토큰 전량 prefill |
| long | ~2540 토큰 전량 prefill (대화급) |
| turn1 | 직전 long과 같은 프롬프트 → 슬롯 prefix 재사용 |
| turn2 | long 히스토리 + assistant + 후속 질문 → 다턴 suffix |
| queue 4-way | `n_parallel=1` 에서 동시 4요청. 거절 vs 대기 |

GPU 클럭은 양측 동일하게 `perflevel=manual`, sclk bitmask level 8, mclk 1000 MHz를 걸었다.
유휴 시 표시는 925 MHz였고, 추론 중에는 같은 정책이 적용된다.

---

## 2. 환경 — 구동 옵션이 왜 이런가

이 숫자들은 “기본값으로 켰더니 나온” 값이 아니다.
32 GiB MI50에 DeepSeek-V4-Flash IQ2_M(84.68 GiB, 284.33B, expert 256/used 6)을 올리려다 얻은 타협이다.

| | golbang-deepseek | llama-server |
|--|------------------|--------------|
| 유닛 | `/etc/systemd/system/golbang-deepseek.service` | `/etc/systemd/system/llama-server.service` |
| 바이너리 | `target/release/golbang-server` (`d1d47c2`) | `llama.cpp.new/build/bin/llama-server` |
| 컴퓨트 트리 | 핀 `llama.cpp` **`5b474eb69`** + `libggml-hip.so` | **`e700bfb37`** (`llama.cpp.new`) |
| 모델 | `DeepSeek-V4-Flash-0731-UD-IQ2_M` 3샤드 | 동일 파일 |
| GPU | MI50 32 GiB, gfx906, ROCm 7.2 | 동일 |
| HIP env | `HSA_OVERRIDE_GFX_VERSION=9.0.6` `ROCBLAS_USE_HIPBLASLT=0` | 동일 |
| 오프로드 | `--n-gpu-layers 99 --n-cpu-moe 32 --flash-attn on` | `-ngl 99 -ncmoe 32 -fa 1` |
| 컨텍스트 | `--n-ctx 60000 --n-parallel 1 --n-threads 8` | `-c 60000 --parallel 1 -t 8` |
| batch | `--n-batch 5800 --n-ubatch 1024` | `-b 5800 -ub 5800` |
| DSV4 | `--n-rs-seq 1 --jinja --reasoning-format deepseek` | `--jinja --reasoning-format deepseek` |
| 큐 | `--queue-size 1` | llama-server 내부 슬롯 큐 (거절 없음) |
| listen | `0.0.0.0:8080` | 동일 |
| VRAM (벤치 직후) | **32397 → 32687 MiB** | **31304 → 31695 MiB** |

### 옵션별 삽질 요약

1. **`--n-cpu-moe 32` / `-ncmoe 32`**
   `n_gpu_layers=99` 만으로는 85 GiB가 32 GiB에 안 들어간다.
   `blk.0–31` 라우팅 expert만 CPU mmap. attention·공유 expert·나머지 ~11층 expert는 GPU.
   층당 ~2 GiB라 지금 장부에서는 31로도 못 내린다. 상세: 위키 `n-cpu-moe-vram`.

2. **`--n-ubatch 1024` (골방만)**
   핀 SHA에서 `-ub 5800` 은 compute 버퍼 ~5 GiB → `cudaMalloc` OOM.
   논리 `n_batch=5800`은 유지하고 물리 ubatch만 1024 (compute ~2.2 GiB).
   60k KV ~2.9 GiB + 가중치 ~27 GiB 위에 남는 주머니가 그것뿐이다.

3. **`n_batch` / `n_ctx` 분리**
   둘을 같게 두면 60k 컨텍스트의 compute 버퍼가 폭주한다.

4. **`prefill_max`**
   P3 placeholder `32`가 생산에 그대로 나가면 3k 프롬프트가 **14.8 tok/s / 211초**.
   `n_parallel==1` 이면 `prefill_max = n_batch` (5800). 수정 후 4k 스모크 103 tok/s.

5. **`--jinja --reasoning-format deepseek`**
   P1 ChatML 하드코딩은 DSV4에서 `<|im_end|>` 누수.
   GGUF jinja + `<think>` 를 `reasoning_content`로 분리해야 Open WebUI와 맞는다.

6. **`--n-rs-seq 1` + prefill 체크포인트**
   DSV4는 긴 suffix `seq_rm`이 실패한다. HCA 스냅샷을 수백 MiB로 키울 수 없어,
   prefill 끝 체크포인트(~17 MiB / 3k) + 마지막 1토큰만 자른다. 이슈 #28 / P5.

7. **HIP 환경**
   `HSA_OVERRIDE_GFX_VERSION=9.0.6`, `ROCBLAS_USE_HIPBLASLT=0` 없으면 gfx906 경로가 깨지거나 hipBLASLt가 실패한다.

8. **`Conflicts=llama-server.service`**
   같은 카드·같은 포트·같은 85 GiB 가중치를 두 프로세스가 동시에 가질 수 없다.

---

## 3. 속도 — 오늘 실측

서버가 돌려준 `timings` (llama-server 키 호환: `prompt_per_second` / `predicted_per_second` / `cache_n`).

### 3.1 전량 prefill (prompt eval)

| 케이스 | prompt_n | cache_n | golbang tok/s | llama tok/s | golbang wall | llama wall |
|--------|--------:|--------:|--------------:|------------:|-------------:|-----------:|
| short-1 | 8 / 7 | 0 / 1 | 11.60 | 10.78 | 3.59 s | 3.53 s |
| medium | 664 / 663 | 0 / 1 | **62.99** | **63.95** | 14.49 s | 14.28 s |
| **long** | 2541 / 2540 | 0 / 1 | **84.39** | **126.09** | **34.20 s** | **24.15 s** |

짧은 프롬프트 tok/s는 고정 오버헤드 때문에 낮고, 비교 대상이 아니다.
**~660토큰은 동급. ~2540토큰은 llama-server가 약 1.5배.**
원인은 커널 언어가 아니라 생산 설정의 **물리 ubatch**(1024 vs 5800)와 컴퓨트 SHA 차이다.
llama-server 저널 중간값도 같은 방향을 가리킨다: 2536토큰 처리 중 **162.9 tok/s** 순간치, 최종 126.1 tok/s.

### 3.2 decode (eval, 체감 생성 속도)

| 케이스 | golbang tok/s | llama tok/s |
|--------|--------------:|------------:|
| short-1 | 7.95 | 8.37 |
| short-2 | 8.26 | 8.50 |
| medium | 8.13 | 8.19 |
| long | 7.87 | 8.00 |
| turn1 | 8.09 | 8.20 |
| turn2 | 8.35 | 8.34 |
| 골방 `/metrics` 수명 | **8.16** | — |

**오차 범위의 동률.** 약 8 tok/s, 토큰당 ~120–127 ms.
P6 rocprof가 이미 말한 천장이다. decode wall의 ~68%는 호스트/CPU expert + `graph splits=66`이고, 1위 GPU 커널은 wall의 6.3%다.
골방이 Rust라는 사실과 llama.cpp가 C++라는 사실은 이 숫자에 거의 안 나타난다.

### 3.3 prefix / 다턴

| 케이스 | golbang | llama-server |
|--------|---------|--------------|
| turn1 (같은 long 재요청) | cache_n=**2540**, prefill 1토큰, wall **4.10 s** | cache_n=**2537**, prefill 4토큰, wall **4.30 s** |
| turn2 (후속 질문) | cache_n=**2540**, suffix 22토큰 / 2.04 s, wall **5.90 s** | cache_n=**2537**, suffix 25토큰 / 2.49 s, wall **6.34 s** |
| short-2 (같은 `1+1=`) | cache_n=7 | cache_n=4 |

양측 모두 슬롯 prefix를 재사용한다.
골방은 P5 경로: `prefix restored from checkpoint reuse_len=2540`.
llama.cpp.new도 자체 슬롯 캐시로 같은 주문을 2.5k → 수 토큰으로 줄인다.

체감 의미: 3k대 대화를 턴마다 다시 먹으면 TTFT 30–40초. 캐시가 살면 **4–6초**.
이 이득은 “골방만의 속도”가 아니라 **양쪽 다 켜 둔 기능**이다.
골방 쪽에서 특별히 한 일은, 핀 SHA의 DSV4가 긴 `seq_rm`을 거부하는  constr를 **체크포인트 + `n_rs_seq=1`로 뚫은 것**이다. 그 전에는 생산 대화 3턴이 전부 `cache_n=0`이었다.

### 3.4 과부하 (동시 4요청)

생산 설정은 슬롯 1개다.

| | golbang (`queue-size=1`) | llama-server |
|--|--------------------------|--------------|
| HTTP | **200, 200, 503, 503** | 200, 200, 200, 200 |
| 503 지연 | **1.6 ms / 1.6 ms**, `Retry-After: 1` | 없음 |
| 요청별 wall | 3.22 / 6.39 / 0.002 / 0.002 s | 5.78 / 2.90 / 8.64 / **11.54** s |
| 스위트 wall | 6.39 s | 11.54 s |

골방은 큐가 가득 차면 핸들러가 decode를 기다리지 않고 즉시 거절한다.
llama-server는 네 건을 슬롯 하나에 직렬로 넣는다. 마지막 클라이언트는 11.5초를 모른다 채 기다린다.

이 차이가 ADR이 처음부터 “llama.cpp보다 개선된 동시성”이라고 부른 실체다.
GPU를 더 빠르게 돌리는 것이 아니라, **HTTP 제어면이 `llama_decode`에 묶이지 않는 것**이다.

---

## 4. 아키텍처 — 무엇이 달라졌나

### 4.1 결정 (ADR, 2026-08-13 accepted)

요구는 다섯 개였다. ① Rust ② GGUF ③ llama.cpp보다 개선된 동시성 ④ gfx906 ⑤ 서빙+컴퓨트 모두 Rust.

④와 ⑤가 충돌한다. candle/mistral.rs는 HIP가 없고, hipfire는 RDNA, rocm-rs gfx906 코드젠은 미검증이다.
그래서 **하이브리드**로 확정했다.

```
┌──────────────────────────────────────────────────────────┐
│  golbang  (단일 Rust 바이너리, tokio 1 + axum 0.8)         │
│                                                          │
│   HTTP API          Scheduler           Continuous       │
│   (axum)     →      (async loop)  →     Batcher          │
│   /v1/chat/…        SchedulePolicy      slots + budget   │
│   /metrics          bounded mpsc                         │
│         CancellationToken · 즉시 503                      │
│                            │                             │
│                            │ spawn_blocking + C ABI      │
│                            ▼                             │
│   golbang-sys  →  llama.h @ 5b474eb69                    │
│                →  libllama + libggml-hip (gfx906)        │
└──────────────────────────────────────────────────────────┘
```

- 서빙 / 스케줄 / 배칭 / 템플릿 / 샘플링 / API = **100% Rust**
- GPU 수학 = 이미 gfx906으로 빌드된 HIP 커널을 Rust가 호출
- 산출물은 파이썬/Go 런타임이 없는 **단일 바이너리** (`golbang-server` 5.3 MiB + 같은 `.so`)

llama-server는 같은 수학을 **C++ 프로세스 안**에서 돌린다.
HTTP는 cpp-httplib 스레드, 스케줄은 `update_slots()`가 `llama_decode`에 동기 결합.

### 4.2 크레이트 경계

| 크레이트 | 역할 | llama-server 대응 |
|----------|------|-------------------|
| `golbang-sys` | `llama.h` bindgen, SHA 핀, `.so` 링크. unsafe만 여기 | llama.cpp 내부 |
| `golbang-core` | Model/Engine, Scheduler, Slot, Policy, prefix, jinja, reasoning | `server.cpp` 슬롯 루프 + 템플릿 |
| `golbang-server` | axum 라우트, SSE, `/metrics`, API 키 | cpp-httplib + 내장 UI |

Rust 소스(생성 bindings 제외) 약 **6.1k 줄**.
스케줄러 한 파일(`scheduler.rs`)이 1.3k 줄로, llama-server의 거대 `server.cpp`에서 **채팅 서빙에 필요한 루프만** 떼어 온 형태다.

### 4.3 스케줄 — “개선된 동시성”의 정확한 의미

llama-server도 continuous batching이 **기본**이다.
“CB가 없다”, “HTTP가 decode에 막힌다”는 사실이 아니다.

실제 한계는 다음이었다.

| llama-server | golbang |
|--------------|---------|
| 스케줄 루프가 `llama_decode`에 동기 결합. 정책 교체가 C++ 내부 | `SchedulePolicy` trait (`join` / `evict` / `rank`). 기본 FIFO |
| 과부하 시 내부 큐에 쌓임 | bounded `mpsc::try_send` → **즉시 503 + Retry-After** |
| 취소는 decode 경계까지 대기 | 핸들러는 즉시 `CancellationToken`. 슬롯 회수는 같은 하한(decode 1회) |
| 임베딩/rerank/스펙큘/슬롯 UI까지 한 프로세스 | 채팅 스트리밍 + 메트릭 + 최소 스모크 UI(`--webui`) |

join의 정의도 같다: **이번 `llama_decode`가 돌아온 직후** 빈 슬롯에 넣는다.
골방 로그: `join after decode boundary`. GPU 워커만 `spawn_blocking`.
그래서 decode 중에도 수신·503·취소 토큰이 살아 있다.
P2에서 Qwen 0.6B `np=2` 후발 TTFT는 132.5 → 50.1 ms로, 같은 조건 llama-server 49.8 ms와 동률이었다.

오늘 생산은 `n_parallel=1`이라 그 TTFT 이득은 나타나지 않는다.
나타난 것은 큐 프로브의 **1.6 ms 503**이다.

### 4.4 메모리 장부와 그래프

양쪽 다 같은 패킹이다.

| 항목 | 크기 |
|------|-----:|
| 가중치 전체 | 84.68 GiB |
| CPU_Mapped expert 0–31 | ~62.8 GiB |
| GPU 가중치 | ~26.9 GiB |
| KV 60k | ~2.9 GiB |
| compute (골방 ubatch 1024) | ~2.2 GiB |
| VRAM 합 | ~31.6 / 32.0 GiB |

런타임에 가중치를 매 토큰 PCIe로 보내지 않는다. 활성 expert 6개의 GEMV가 **CPU에서** 돈다.
층마다 GPU attention ↔ CPU expert 경계에서 그래프가 끊긴다 (`splits=66` at bs=1).
그래서 커널을 Rust로 다시 쓰거나 HIP 한 개를 고쳐도 decode는 안 움직인다. P6(#29)가 그 프로파일이다.

골방이 여기서 한 일:

- `LoadParams.n_cpu_moe` 로 llama.cpp와 같은 정규식 오버라이드를 Rust에서 켬
- `n_batch` / `n_ubatch` / `n_ctx` / `n_rs_seq` 를 CLI로 분리
- `IterationBudget::for_context` 로 생산 prefill 청크를 llama-server 단일 슬롯과 맞춤

못 한 일:

- 핀 SHA에서 ubatch 5800을 32 GiB에 넣는 것. 그래서 오늘 long prefill이 진다.

### 4.5 다턴 prefix (P3 축소안 → P5)

```
P3: 전역 PrefixStore 도입 시도
    → llama_memory_seq_cp 스파이크는 가능, 그러나 전역 복사는 안 씀
    → 슬롯 로컬 LCP만. 그런데 finish_slot 이 매번 clear_seq
    → 생산 대화 3턴 cache_n=0, 턴당 prefill 41–44초

P5: Stop/Length 는 KV를 남긴다
    생성 토큰 ID를 슬롯 캐시에 붙인다
    DSV4: prefill 끝 PARTIAL_ONLY 체크포인트 + n_rs_seq=1
    다음 bind: LCP → suffix rm 실패 → 체크포인트 복원 → 1토큰 rm → suffix만 prefill
```

오늘 측정에서 그 경로가 살아 있다.
`prefix restored from checkpoint reuse_len=2540`. turn2 wall 5.90 s.
llama.cpp.new도 같은 주문을 캐시하므로, **사용자 체감의 다턴 이득은 이제 양쪽 다 있다.**
차이는 구현 위치다. 골방은 그 로직이 Rust에 있고, DSV4 제약을 위키·지시서·테스트로 고정했다.

### 4.6 API 표면

| | golbang | llama-server |
|--|---------|--------------|
| `POST /v1/chat/completions` | OpenAI SSE + `stream:false` | 동일 + 더 많은 필드 |
| `timings` | llama 키 호환 | 원조 |
| `reasoning_content` | Rust `ReasoningFormat::Deepseek` | `--reasoning-format deepseek` |
| jinja | minijinja, GGUF `tokenizer.chat_template` | 내장 jinja |
| `/metrics` | Prometheus. prefix 제외 수명 tok/s, TTFT/ITL 히스토그램, 503 카운터 | 있음 (`endpoint_metrics`) |
| `/health` | **없음 (404)** | `{"status":"ok"}` |
| `/props`, 슬롯 UI, 임베딩, rerank | 없음 | 있음 |
| 인증 | Bearer / `X-Api-Key` | `--api-key` |
| 빈 messages | HTTP 400 | (구현 의존) |

골방은 **채팅 서빙에 필요한 면만** 남겼다.
빠진 `/health` 는 열등이지, 철학이 아니다. 넣으면 된다.

### 4.7 로드맵이 남긴 것

| 단계 | 이슈 | 상태 | 아키텍처에 남긴 것 |
|------|------|------|-------------------|
| P0 | #23 | done | SHA 핀 FFI. gfx906 Hello decode |
| P1 | #24 | done | OpenAI SSE, 당시 ChatML |
| P2 | #25 | done | Policy + 즉시 503 + spawn_blocking |
| P3 | #26 | done | slot-local prefix, chunked prefill, `/metrics` |
| P5 | #28 | done | evict 후 KV 생존, DSV4 체크포인트 |
| P6 | #29 | done | rocprof. 커널 지목 실패. HIP 패치 0 |
| P4 | #27 | todo | 순수 Rust 커널. P6 gate 실패로 열지 않음 |

P4를 열지 않은 이유가 오늘 숫자와 같다.
언어를 바꿔도, 특정 HIP 커널 하나를 고쳐도, decode 8 tok/s는 안 바뀐다.

---

## 5. 그럼 골방이 나은 점은 뭔가

처리량 표만 보면 llama-server가 long prefill에서 이기고 decode는 동률이다.
그래도 이 레포를 만든 이유는 아래다.

### 5.1 제어면이 우리 코드다

과부하를 **관측 가능한 거절**로 바꿀 수 있다.
오늘: 동시 4요청 중 2건이 1.6 ms 만에 503.
llama-server 클라이언트는 11.5초를 슬롯 뒤에서 기다린다.
정책을 바꾸려면 골방은 `SchedulePolicy` 구현체를 교체하면 되고, llama-server는 `server.cpp`를 고친다.

### 5.2 HTTP가 decode에 묶이지 않는다

tokio 런타임 + `spawn_blocking` 한 줄이 GPU 구간이다.
수신·인증·503·취소가 추론 스레드를 공유하지 않는다.
llama-server도 스레드를 나누지만, 스케줄 루프 자체는 decode와 한 몸이다.

### 5.3 DSV4를 32 GiB에서 켜는 경로를 우리가 소유한다

옵션 여덟 개가 전부 코드와 유닛 파일에 이름이 있다.
`n_cpu_moe`, `n_ubatch`, `n_rs_seq`, jinja, reasoning, Conflicts.
llama.cpp 플래그를 “외워서 켠” 것이 아니라, OOM·ChatML 누수·`cache_n=0`·`prefill_max=32` 를 하나씩 깨고 Rust 쪽에 고정했다.
다음 사람이 같은 삽질을 반복하지 않는다.

### 5.4 다턴 prefix를 DSV4 제약에 맞춰 재구현했다

llama.cpp.new도 캐시가 산다. 그래도 핀 SHA + DSV4 `seq_rm` 실패는 **우리 쪽에서 먼저 막혀 있었고**, P5가 그걸 체크포인트로 풀었다.
로직이 `scheduler.rs` / `prefix_cache.rs` / `slot.rs`에 있고 테스트가 있다.
“의존 라이브러리가 알아서 해 주길” 기다리지 않는다.

### 5.5 관측성이 서빙 레이어에 붙어 있다

요청이 끝나면 llama와 같은 `prompt eval` / `eval time` 로그.
응답 JSON에 `timings`.
`/metrics`에 수명 평균과 히스토그램.
오늘 골방 수명 decode **8.16 tok/s**, prefill(캐시 제외) **68.9 tok/s**.

### 5.6 표면이 얇다

임베딩·rerank·스펙큘·내장 UI를 안 싣는다.
gfx906 + 이 모델 + OpenAI 채팅만 보면 고장면이 적다.
단일 언어(Rust)라 스케줄·템플릿·메트릭을 한 저장소에서 리뷰한다.

### 5.7 안전 핀

`golbang-sys/build.rs`는 HEAD가 `5b474eb69`가 아니면 빌드를 거절한다.
같은 디렉터리의 `llama.cpp.new` / `-furnace` / `-prefetch`를 실수로 링크하지 못한다.
오늘 비교가 “다른 SHA의 다른 ubatch”인 줄 아는 것도 이 핀 덕분이다.

---

## 6. llama-server가 나은 점 · 동률인 점

정직하게 적는다.

| 항목 | 승 |
|------|----|
| ~2.5k 전량 prefill | **llama-server** (126 vs 84 tok/s). `-ub 5800` + 신 SHA |
| decode tok/s | **동률** (~8). 차이는 노이즈 |
| 다턴 prefix 체감 | **동률** (양쪽 cache_n ≈ 2540, 2턴 wall 6초 전후) |
| 기능 폭 | **llama-server** (`/props`, 임베딩, 샘플러 옵션). 골방의 내장 UI는 API 키·속도 통계·스모크 채팅뿐 |
| `/health` | **llama-server** |
| 신 트리 추적 | **llama-server** (`e700bfb37`). 골방은 고의로 핀 |
| GPU 커널 | **동일 계열** ggml-hip. 골방은 더 오래된 SHA |
| 32 GiB에서 모델이 뜨는가 | **동률**. 같은 `-ncmoe 32` |

골방 long prefill이 느린 것은 회귀로 포장하지 않는다.
32 GiB + 핀 SHA의 compute 버퍼가 5800을 거부한 결과다.
ubatch를 올리려면 VRAM 장부를 바꿔야 한다 (`n-cpu-moe-vram`).

---

## 7. 결론

골방은 llama.cpp를 **교체한 GPU 엔진이 아니다.**
같은 HIP 커널을 Rust가 오케스트레이션하는 **서빙 엔진**이다.

오늘 숫자로 말하면:

- 생성 속도: **같다** (8 tok/s, `n_cpu_moe=32` 천장)
- 긴 프롬프트 첫 토큰: **llama-server가 빠르다** (24 s vs 34 s @ 2.5k)
- 같은 대화의 다음 턴: **둘 다 6초 전후** (prefix 생존)
- 슬롯이 바쁠 때: **골방만 1.6 ms 만에 503**

그래서 “llama.cpp로 돌리는 것보다 골방이 나은 점”은 벤치 표의 tok/s가 아니라 다음이다.

1. **거절할 수 있는 서버** — 과부하가 대기 시간으로 숨지 않는다
2. **정책을 우리 언어로 바꾸는 서버** — join/evict/예산이 Rust trait
3. **이 카드·이 모델을 켜는 지식이 코드에 남은 서버** — 옵션 여덟 개가 재현 가능하다
4. **DSV4 다턴이 우리 스케줄러에서 다시 산 서버** — `cache_n=0` 40초를 6초로 줄인 경로가 저장소에 있다
5. **커널 환상에 속지 않는 서버** — P6가 “고칠 커널 없음”을 숫자로 닫았다

다음 레버는 여전히 커널이 아니다. VRAM을 늘리거나 `n_cpu_moe`를 내릴 수 있을 때 decode가 움직인다.
그 전까지 골방의 할 일은 처리량 경쟁이 아니라, 이 천장 위에서 **제어면과 다턴과 관측을 더 정확하게 만드는 것**이다.

---

## 부록 A. 재현

```bash
# 골방 (생산 유닛이 이미 떠 있으면 그대로)
python3 scripts/compare_golbang_llama.py \
  --name golbang-deepseek \
  --url http://127.0.0.1:8080/v1/chat/completions \
  --out docs/bench/raw/golbang-$(date +%F).json

# 교체
sudo systemctl stop golbang-deepseek.service
sudo systemctl start llama-server.service
# 모델 로드 ~9초 (페이지 캐시)

python3 scripts/compare_golbang_llama.py \
  --name llama-server \
  --url http://127.0.0.1:8080/v1/chat/completions \
  --out docs/bench/raw/llama-server-$(date +%F).json

sudo systemctl stop llama-server.service
sudo systemctl start golbang-deepseek.service
```

## 부록 B. 관련 기록

- 위키: `architecture-decision`, `dsv4-run-notes`, `n-cpu-moe-vram`, `p5-prefix-cache-notes`, `p6-rocprof-notes`
- 벤치: `docs/bench/p2.md`, `p3.md`, `p5.md`, `p6.md`
- 지시서: `docs/work-orders/`
- 이전 비통제 비교: 작업 #299 (다른 날 대화 집계)
