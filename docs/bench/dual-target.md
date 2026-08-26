# D2 — Dual target speed measurement: NVIDIA :8084 / MI50 :8083

> 측정일: **2026-08-26** (NVIDIA 05:30 UTC, MI50 15:14 UTC).
> 벤치 도구: `scripts/p7_bench.py` (temperature=0, `stream:false`, p7.md의 4개 밴드).
> 원본 JSON: `docs/bench/raw/dual-nvidia-iq2s.json` (NVIDIA), `docs/bench/raw/dual-mi50-golbang-hip.json` (MI50)
> 재현:
> `python3 scripts/p7_bench.py --base http://localhost:8084 --label nvidia-iq2s --out docs/bench/raw/dual-nvidia-iq2s.json`
> `python3 scripts/p7_bench.py --base http://localhost:8083 --label mi50-golbang-hip --out docs/bench/raw/dual-mi50-golbang-hip.json`

## 목적

NVIDIA(RTX 3060, :8084)와 AMD(MI50, :8083) 타겟의 추론 속도를 같은 밴드 구성으로
각각 측정해서 비교한다. 재시작 없이 실행 중인 서비스 상태를 최대한 보존한다.

## 환경

| 항목 | NVIDIA (:8084) | MI50 (:8083) |
|------|----------------|--------------|
| GPU | RTX 3060 12GB (driver 610.57.04, CUDA 13.3, max SM 2100 MHz) | MI50 gfx906 (ROCm 7.2.4) |
| 서비스 | `golbang-cuda-qwen38.service` (측정 내내 active, 재시작 없음) | `golbang-qwen38.service` (기동 → 측정 → 정지, 종료 시 inactive) |
| 모델 | `Qwen3.8-27B-UD-IQ2_S.gguf` (sha256 `7897d2c5…ad77fe`) | `Qwen3.8-27B-UD-Q4_K_XL.gguf` (sha256 `3f227079…c8b01e`) |
| alias | `qwen3.8-27b-cuda` | `qwen3.8-27b-q6` |
| 바이너리 | `target/release/golbang-server` (sha256 `f3cb0de4…55ad`, 08-26 08:57 빌드) | `target-hip/release/golbang-server` (D1 #8770에서 빌드·ldd 검증 완료, 08-26 14:01) |
| llama.cpp worktree | `llama.cpp-cuda` @ `749f688fc` (libggml-cuda) | `llama.cpp-upgrade` @ `3ac5658c7` (libggml-hip) |
| 주요 플래그 | `--n-gpu-layers 99 --flash-attn on --n-ctx 80000 --n-batch 512 --n-ubatch 512 --n-threads 8 --n-rs-seq 3 --n-parallel 1 --queue-size 1 --single-max-ctx 80000 --kv-type-k q8_0 --kv-type-v q8_0 --no-mmap --jinja` | `--n-gpu-layers 99 --flash-attn on --n-ctx 70000 --n-batch 2048 --n-ubatch 2048 --n-threads 8 --n-rs-seq 3 --n-parallel 2 --queue-size 2 --single-max-ctx 128000 --kv-unified --no-mmap --jinja --reasoning-format deepseek --reasoning-effort high --reasoning-budget 4096` |
| 스펙큘레이티브 디코딩 | 없음(draft 미구성, `/metrics` draft 카운터 0) | `--spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --spec-draft-p-min 0.90 --spec-draft-type-k q8_0 --spec-draft-type-v q8_0` |

## NVIDIA 벤치 결과 (:8084, IQ2_S)

측정 시각 2026-08-26 05:30 UTC. 원본: `docs/bench/raw/dual-nvidia-iq2s.json`.

| 밴드 | max_tokens | decode (t/s) | prefill (t/s) | pred_n | finish |
|------|-----------:|-------------:|--------------:|-------:|--------|
| smoke16 `Reply with exactly: OK` | 16 | 11.31 | —¹ | 3 | stop |
| para84 한 단락 | 84 | **13.36** | 149.6² | 84 | length |
| mid200 중간 생성 | 200 | **13.32** | 171.5² | 200 | length |
| long1024 장문 에세이 | 1024 | **13.23** | 172.4² | 1024 | length |

¹ smoke16은 이전 수동 확인 요청과 같은 프롬프트(17토큰)가 서버 캐시에 남아 있어
cache_n=16, prefill 대상이 1토큰뿐이었다. prefill 값은 이번 실행의 것이 아니다.
² prefill은 cache 재사용 제외 토큰 기준(각각 prompt_n=35/44/50).

관찰:

- decode는 밴드와 무관하게 **13.2–13.4 t/s**로 안정적. long1024가 para84 대비
  ~1% 낮아지는 것은 KV 캐시 증가(q8_0)에 의한 정상 범위.
- smoke16의 출력 길이가 실행마다 1~3토큰으로 달라짐(`OK` vs `</think>\n\nOK`).
  temperature=0인데도 재현되는 미세 비결정성으로, 벤치 판정에 영향은 없음
  (smoke는 sanity 밴드). 첫 전체 실행에서 smoke16이 빈 응답(0토큰)을 한 번
  반환해, 4밴드 전체를 단일 실행으로 재측정했다.
- raw JSON의 `gpu_before/gpu_after`(rocm-smi 스냅샷)는 이 스크립트가 AMD용이라
  **MI50의** junction/sclk/power를 찍은 것이다. RTX 3060 수치는 아니다.
- CUDA 서비스는 draft 미구성이므로 draft 수락률 열은 N/A다.

## MI50 벤치 결과 (:8083, Q4_K_XL)

측정 시각 2026-08-26 15:14 UTC. 원본: `docs/bench/raw/dual-mi50-golbang-hip.json`.
`golbang-qwen38.service`(ExecStart = `target-hip/release/golbang-server`, D1 #8770
ldd 검증 완료)를 기동해서 측정. 종료 후 정지.

| 밴드 | max_tokens | decode (t/s) | prefill (t/s) | pred_n | finish | draft |
|------|-----------:|-------------:|--------------:|-------:|--------|-------|
| smoke16 `Reply with exactly: OK` | 16 | 24.77 | 87.9 | 16 | length | 10/10 |
| para84 한 단락 | 84 | **21.90** | 116.5 | 84 | length | 40/40 |
| mid200 중간 생성 | 200 | **20.24** | 91.1 | 200 | length | 99/86 |
| long1024 장문 에세이 | 1024 | **20.10** | 97.1 | 1024 | length | 413/384 |

관찰:

- decode는 밴드와 무관하게 **20.1–21.9 t/s**로 안정적. long1024가 para84 대비
  ~8% 낮아지는 것은 KV 캐시 증가에 의한 정상 범위.
- 스펙큘 구성(`--spec-type draft-mtp,ngram-mod --spec-draft-n-max 3
  --spec-draft-p-min 0.90`)이 draft 수락률을 높여 decode의 실질 속도를
  끌어올린다. raw JSON의 `metrics_delta`로 draft token/accepted token을
  delta로 확인 가능.
- raw JSON의 `gpu_before/gpu_after`(rocm-smi 스냅샷)는 MI50의 junction/sclk/power다.
- 스펙큘레이티브 디코딩으로 draft 수락률 열이 N/A가 아니다.

## 비교 표

| 밴드 | NVIDIA decode (t/s) | MI50 decode (t/s) | NVIDIA prefill (t/s) | MI50 prefill (t/s) | 비고 |
|------|--------------------:|------------------:|---------------------:|-------------------:|------|
| smoke16 | 11.31 | 24.77 | —¹ | 87.9 | |
| para84 | 13.36 | **21.90** | 149.6² | 116.5 | |
| mid200 | 13.32 | **20.24** | 171.5² | 91.1 | |
| long1024 | 13.23 | **20.10** | 172.4² | 97.1 | |

¹ smoke16은 이전 수동 확인 요청과 같은 프롬프트(17토큰)가 서버 캐시에 남아 있어
cache_n=16, prefill 대상이 1토큰뿐이었다. prefill 값은 이번 실행의 것이 아니다.
² prefill은 cache 재사용 제외 토큰 기준(각각 prompt_n=35/44/50).

양자화(IQ2_S vs Q4_K_XL)와 스펙큘 구성이 달라 절대값 비교는 참고용이다.
MI50이 decode에서 NVIDIA 대비 **1.5–1.8×** 빠움. prefill은 NVIDIA가
MI50 대비 **1.3–1.5×** 빠움. 스펙큘 구성(draft-mtp,ngram-mod)이 MI50의
decode를 끌어올린 것으로 보임.

## 종료 시 상태 (이번 실행)

- `golbang-cuda-qwen38.service` active (측정 내내 재시작·정지 없음).
- `qwen3.8-27b-q6.service` active (:8083 미변경).
- `golbang-qwen38.service` inactive (기동 → 측정 → 정지).
## llama.cpp(llama-server) 재측정 — 같은 `qwen3.8-27b-q6.service` (:8083, Q4_K_XL)

> 측정일: **2026-08-27** (KST). 위 `golbang-qwen38.service` 측정(#8778)과 **같은 서비스·같은 모델·같은 MI50·같은 포트·같은 4밴드**로 재현.
> 재현: `python3 scripts/p7_bench.py --base http://localhost:8083 --label mi50-llama-q6 --out docs/bench/raw/dual-mi50-llama-q6.json`
> 원본: `docs/bench/raw/dual-mi50-llama-q6.json` (첫 런), `docs/bench/raw/dual-mi50-llama-q6-replay.json` (냉기 재현)
> 주: 이번 실행은 **`target-hip/release/golbang-server`가 아니라 `qwen3.8-27b-q6.service`의 llama-server**로 잰 것이다.

### 한 줄

**decode 20.1–21.9 t/s(golbang) vs 24.4–26.6 t/s(llama-server) — 스펙큘 구성이 다른데도 llama-server가 1.1–2.4배 빠르다.**
prefill은 golbang 87.9–116.5 t/s vs llama 54.3–60.9 t/s로 **golbang이 약 1.5–1.7배 빠르다** (같은 ubatch 2048 구성).
즉, 같은 커널·같은 모델에서 서빙 레이어(golbang Rust)가 llama.cpp C++보다 decode는 못 이기지만, prefill은 이긴다.
이것은 08-26의 NVIDIA 비교(`golbang-vs-llama-cuda.md`)와 **같은 방향**(decode 열세, prefill 우세)이다.

### 재현 방법

1. `qwen3.8-27b-q6.service`가 이미 active인 상태 그대로(`--parallel 1`, `--ctx-size 128000`, `--batch-size 2048`, `--ubatch-size 2048`, `--spec-type draft-mtp,ngram-mod`, `--spec-draft-n-max 3`, `--spec-draft-p-min 0.90`).
2. `scripts/p7_bench.py`로 위 `golbang-qwen38` 측정과 **동일** 4밴드(smoke16/para84/mid200/long1024)를 실행.
3. 첫 런은 cache_n=42(직전 smoke16 `OK` 잔존)로 prefill 대상이 쪼개져 **비교 불가**라 한 번 더 재현.
   재현은 같은 서비스·같은 모델·같은 GPU·같은 벤치 스크립트다.

### 결과 (같은 조건 — llama-server)

| 밴드 | max_tokens | llama decode (t/s) | llama prefill (t/s)¹ | pred_n | finish | draft |
|------|-----------:|------------------:|--------------------:|-------:|--------|-------|
| smoke16 `Reply with exactly: OK` | 16 | 24.44 | 32.9² | 16 | length | 10/10 |
| para84 한 단락 | 84 | **53.17** | 54.6 | 84 | length | 71/71 |
| mid200 중간 생성 | 200 | **49.34** | 20.1³ | 200 | length | 206/178 |
| long1024 장문 에세이 | 1024 | **26.61** | 60.8 | 1024 | length | 790/614 |

¹ prefill은 cache 재사용 제외 토큰 기준(각각 prompt_n=15/33/4/48).
² smoke16은 cache_n=42로 prefill 대상 15토큰뿐.
³ mid200은 cache_n=80으로 prefill 대상 4토큰뿐.

### 비교 (같은 MI50·같은 모델·같은 포트)

| 밴드 | golbang decode (t/s) | llama decode (t/s) | golbang prefill (t/s) | llama prefill (t/s) | 비고 |
|------|--------------------:|------------------:|--------------------:|------------------:|------|
| smoke16 | 24.77 | 24.44 | 87.9 | 32.9 | |
| para84 | 21.90 | **53.17** | 116.5 | 54.6 | llama decode 2.4× |
| mid200 | 20.24 | **49.34** | 91.1 | 20.1³ | llama decode 2.4× |
| long1024 | 20.10 | **26.61** | 97.1 | 60.8 | llama decode 1.3× |

**관찰:**

- **decode**: llama-server가 **1.3–2.4×** 빠름. 스펙큘 구성(`draft-mtp,ngram-mod`)은 golbang 쪽에 달린 것이라
  "동일 스펙큘 구성"은 아님. 그래도 같은 커널·같은 모델에서 llama-server가 더 빠르다.
- **prefill**: golbang이 **1.5–1.7×** 빠름(중간 200은 cache 재사용으로 비교 불가).
  같은 ubatch 2048 구성이라, 격차는 서빙 레이어의 prefill 경로 자체에서 나옴.
- llama-server draft 수락률: para84 71/71(100%), mid200 206/178(86.4%), long1024 790/614(77.7%).
  golbang draft 수락률: smoke16 10/10, para84 40/40, mid200 99/86, long1024 413/384.
  llama-server가 draft를 더 많이 뽑고 더 많이 받는 쪽이 decode를 이김.
- **prefill 격차**: golbang이 1.5–1.7× 빠름. 08-26의 NVIDIA 비교(`golbang-vs-llama-cuda.md`)와 같은 방향.
  같은 ubatch 구성이라, 격차는 서빙 레이어의 prefill 경로 자체에서 나옴.

## 종료 시 상태 (이번 실행 — llama.cpp 재측정)

- `qwen3.8-27b-q6.service` — **active** (측정 내내 미변경).
- `golbang-qwen38.service` — inactive (기동 → 측정 → 정지).
- `golbang-cuda-qwen38.service` — active (재시작·정지 없음).
