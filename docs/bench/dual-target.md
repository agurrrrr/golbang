# D2 — Dual target speed measurement: NVIDIA :8084 / MI50 :8083

> 측정일: **2026-08-26** (NVIDIA). MI50 측정은 보류(아래 "MI50" 절 참조).
> 벤치 도구: `scripts/p7_bench.py` (temperature=0, `stream:false`, p7.md의 4개 밴드).
> 원본 JSON: `docs/bench/raw/dual-nvidia-iq2s.json`
> 재현: `python3 scripts/p7_bench.py --base http://localhost:8084 --label nvidia-iq2s --out docs/bench/raw/dual-nvidia-iq2s.json`

## 목적

NVIDIA(RTX 3060, :8084)와 AMD(MI50, :8083) 타겟의 추론 속도를 같은 밴드 구성으로
각각 측정해서 비교한다. 재시작 없이 실행 중인 서비스 상태를 최대한 보존한다.

## 환경

| 항목 | NVIDIA (:8084) | MI50 (:8083, 보류) |
|------|----------------|--------------------|
| GPU | RTX 3060 12GB (driver 610.57.04, CUDA 13.3, max SM 2100 MHz) | MI50 gfx906 (ROCm 7.2.4) |
| 서비스 | `golbang-cuda-qwen38.service` (측정 내내 active, 재시작 없음) | `golbang-qwen38.service` (inactive, 기동 예정) |
| 모델 | `Qwen3.8-27B-UD-IQ2_S.gguf` (sha256 `7897d2c5…ad77fe`) | `Qwen3.8-27B-UD-Q4_K_XL.gguf` (sha256 `3f227079…c8b01e`) |
| alias | `qwen3.8-27b-cuda` | `qwen3.8-27b-q6` |
| 바이너리 | `target/release/golbang-server` (sha256 `f3cb0de4…55ad`, 08-26 08:57 빌드) | `target-hip/release/golbang-server` (D1 #8770에서 빌드·ldd 검증 완료) |
| llama.cpp worktree | `llama.cpp-cuda` @ `749f688fc` (libggml-cuda) | `llama.cpp-upgrade` @ `3ac5658c7` (libggml-hip) |
| 주요 플래그 | `--n-gpu-layers 99 --flash-attn on --n-ctx 80000 --n-batch 512 --n-ubatch 512 --n-threads 8 --n-rs-seq 3 --n-parallel 1 --queue-size 1 --single-max-ctx 40000 --kv-type-k q8_0 --kv-type-v q8_0 --no-mmap --jinja` | (기동 시 확인 예정) |
| 스펙큘레이티브 디코딩 | 없음(draft 미구성, `/metrics` draft 카운터 0) | p7.md 관례상 ngram+MTP 구성 |

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

## MI50 벤치 (:8083, Q4_K_XL) — 보류

사용자 지시로 이번 실행에서는 :8083을 건드리지 않는다. MI50 측정은
`qwen3.8-27b-q6.service`(현재 active)를 일시 정지해야 하므로, 별도 실행에서
이슈 #74의 2단계 순서(정지 → 기동 → 측정 → 원복)대로 진행할 예정이다.

진행 전 확인 사항:

- `golbang-qwen38.service`의 ExecStart가 아직 `target/release/golbang-server`
  (CUDA 링크 바이너리)를 가리키고 있다. D1의 `target-hip/release/golbang-server`
  로 변경된 뒤 기동해야 한다(경로 변경은 D3 서비스 경로 수정 작업의 범위).
- 대조 llama-server `qwen3.8-27b-q6.service`는 :8083을 점유하고 있으므로
  측정 전 `systemctl stop`, 종료 후 반드시 `systemctl start`로 원복한다.

## 비교 표 (MI50 측정 후 채움)

| 밴드 | NVIDIA decode (t/s) | MI50 decode (t/s) | 비고 |
|------|--------------------:|------------------:|------|
| smoke16 | 11.31 | — | |
| para84 | 13.36 | — | |
| mid200 | 13.32 | — | |
| long1024 | 13.23 | — | |

양자화(IQ2_S vs Q4_K_XL)와 스펙큘 구성이 달라 절대값 비교는 참고용이다.

## 종료 시 상태 (이번 실행)

- `golbang-cuda-qwen38.service` active (측정 내내 재시작·정지 없음).
- `qwen3.8-27b-q6.service` active (:8083 미변경).
- `golbang-qwen38.service` inactive (기동하지 않음).
