# golbang vs llama.cpp(llama-server) — NVIDIA RTX 3060 A/B 비교

> 측정일: **2026-08-26** (KST 15:00–15:10). 같은 호스트, 같은 GPU(RTX 3060), **같은 모델 파일**, 같은 포트(:8084)에서
> 한쪽씩 기동해서 잰 통제 A/B. `scripts/p7_bench.py`의 4밴드 구성(temperature=0, `stream:false`).
> 원본 JSON: `docs/bench/raw/dual-golbang-iq2s-rerun.json`(golbang), `docs/bench/raw/dual-llama-iq2s.json`(llama-server)
> 재현:
> - golbang: `systemctl start golbang-cuda-qwen38` → `python3 scripts/p7_bench.py --base http://localhost:8084 --model qwen3.8-27b-cuda --label cuda-golbang-rerun --out docs/bench/raw/dual-golbang-iq2s-rerun.json`
> - llama-server: `systemctl start qwen3.8-q2` → `python3 scripts/p7_bench.py --base http://localhost:8084 --model qwen3.8-q2 --label llama-q2 --out docs/bench/raw/dual-llama-iq2s.json`
> 결과는 있는 그대로 적는다.
>
> **수정 (15:50–16:00 KST): `qwen3.8-q2`의 `--parallel 4` → `--parallel 1`로 바꿔 동일 벤치를 재측정**
> (`docs/bench/raw/llama-q2-par1.json`). 결과: **prefill 수치가 그대로** — parallel(슬롯 분할)이 prefill 격차의
> 원인이 아니라는 게 확인됨. §3.2·§4·§5에 반영.

## 한 줄

**decode 속도는 동률(13.2–13.4 t/s), prefill은 golbang이 1.4–1.7배 빠르다.**
같은 CUDA 커널 위에서 Rust 서빙 레이어(golbang)가 C++ llama-server보다 decode를 못 거는 일은 없었고,
긴 프롬프트 prefill에서는 golbang의 ubatch 512 구성이 llama-server의 slot 단위 파이프라인보다 유리했다.
**`--parallel 4`→`1` 재측정(§3.2)에서도 prefill 격차는 그대로였으므로, 격차 원인은 슬롯 수(컨텍스트 분할)이 아니라
서빙 레이어의 prefill 경로 자체이다.**

---

## 1. 무엇을, 어떻게 쟀나

두 systemd 유닛은 같은 :8084 포트를 공유한다(`Conflicts=` 상호 등록).
순서: `golbang-cuda-qwen38` 정지 → `qwen3.8-q2` 기동 → 벤치 → `qwen3.8-q2` 정지 →
`golbang-cuda-qwen38` 재기동 → **같은 세션에서** 벤치 재측정.

두 측정은 같은 날 10분 간격, 같은 GPU 클럭 정책(sclk level 5, 1386 MHz)에서 수행됐다.
llama-server 측정이 먼저(15:02), golbang 재측정이 나중(15:07)이다.
참고: golbang의 이전 측정(`docs/bench/raw/dual-nvidia-iq2s.json`, 14:30 KST)과 재측정 값이
0.1% 이내로 일치해서 세션 차이는 무시할 수준이다.

| 밴드 | 프롬프트 | max_tokens |
|------|----------|-----------:|
| smoke16 | `Reply with exactly: OK` | 16 |
| para84 | 스펙큘레이티브 디코딩 한 단락 설명 | 84 |
| mid200 | flash attention 기술 문단 (~180단어) | 200 |
| long1024 | 장문 에세이 (800단어+) | 1024 |

## 2. 환경 — 두 유닛의 차이

모델 파일은 **동일**하다(`Qwen3.8-27B-UD-IQ2_S.gguf`, sha256 `7897d2c5…ad77fe`).
llama.cpp 워크트리도 동일하다(`llama.cpp-cuda` @ `749f688fc`) — 즉 CUDA 커널 언어가 같다.

| | golbang-cuda-qwen38 | qwen3.8-q2 (llama-server) |
|--|---------------------|---------------------------|
| 바이너리 | `target/release/golbang-server` (Rust 서빙 레이어) | `llama.cpp-cuda/build/bin/llama-server` (C++) |
| llama.cpp 커널 | `749f688fc` (libggml-cuda) | 동일 |
| 모델 | `Qwen3.8-27B-UD-IQ2_S.gguf` (sha256 `7897d2c5…`) | 동일 파일 |
| alias | `qwen3.8-27b-cuda` | `qwen3.8-q2` |
| 오프로드 | `--n-gpu-layers 99 --flash-attn on` | `-ngl 99 -fa on` |
| 컨텍스트 | `--n-ctx 80000 --single-max-ctx 40000` (슬롯당) | `--ctx-size 80000`, **슬롯당 20224** (`n_slots=4`) |
| 병렬 | `--n-parallel 1 --queue-size 1` | `--parallel 4` (슬롯 4개) |
| batch | `--n-batch 512 --n-ubatch 512` | `--batch-size 512 --ubatch-size 512` |
| KV 캐시 | `--kv-type-k q8_0 --kv-type-v q8_0 --no-mmap` | `--cache-type-k q8_0 --cache-type-v q8_0` |
| 추론(reasoning) | 템플릿 기본값 → **reasoning off** (`thinking=false`) | `--reasoning on --reasoning-effort medium --reasoning-budget 2048` |
| 스펙큘레이티브 디코딩 | 없음 (draft 미구성) | `--spec-type ngram-mod` (실측에서 효과 0, 아래 §4) |
| VRAM (벤치 직후) | 11689 / 12288 MiB | 미기록 (journal verbosity 부족) |

## 3. 속도 — 실측

서버가 돌려준 `timings` (llama-server 로그의 `print_timing`과 일치 확인).

### 3.1 decode (eval, 체감 생성 속도)

| 밴드 | golbang t/s | llama-server t/s | 비고 |
|------|------------:|-----------------:|------|
| smoke16 | 11.07 (3토큰)¹ | 13.22 (16토큰)¹ | ¹샘플 수 차이, 판정 대상 아님 |
| para84 | **13.35** (84토큰) | **13.39** (84토큰) | 동률 |
| mid200 | **13.31** (200토큰) | **13.41** (200토큰) | 동률 |
| long1024 | **13.23** (1024토큰) | **13.36** (1024토큰) | 동률 (llama +1%) |

**오차 범위 내 동률.** 토큰당 ~74.6–75.6 ms.
`--spec-type ngram-mod`가 llama-server 쪽에 달려 있음에도 decode가 golbang과 같다는 것은
ngram 스펙큘이 이 워크로드에서는 사실상 작동하지 않았거나(§4) 효과가 0에 수렴했다는 뜻이다.

### 3.2 prefill (prompt eval)

| 밴드 | prompt_n | golbang t/s | llama-server t/s (par 4) | llama-server t/s (par 1)¹ | 배수(par 1 vs gol) |
|------|---------:|------------:|-------------------------:|--------------------------:|-----:|
| smoke16 | 1²/15² | —² | 33.9² | 34.7² | —² |
| para84 | 35 / 33 | **148.9** | 92.1 | 91.8 | **1.62×** |
| mid200 | 44 / 42 | **170.5** | 111.4 | 111.3 | **1.53×** |
| long1024 | 50 / 48 | **172.0** | 120.4 | 121.0 | **1.42×** |

¹ `--parallel 1`로 기동해 재측정한 값(`llama-q2-par1.json`, KST 16:00).
² smoke16은 golbang 측에서 이전 수동 스모크 요청과 같은 프롬프트가 서버 캐시에 남아 있어
cache_n=16, prefill 대상이 1토큰뿐이었다 — 비교 불가. llama-server는 첫 요청이라 15토큰 전량 prefill.

**단일 요청 기준 prefill은 golbang이 일관되게 1.4–1.7배 빠르다.**
`--parallel 4` → `1`로 바꿔 슬롯당 컨텍스트를 20224에서 풀 80128로 늘려도
llama-server의 prefill t/s는 **±0.6% 이내로 변동이 없다**(para84: 92.07→91.77, mid200: 111.41→111.29, long1024: 120.36→120.97).
즉, 이 격차의 원인은 **슬롯 수·컨텍스트 분할이 아니라 서빙 레이어의 prefill 경로 자체**다.

두 쪽 모두 물리 ubatch가 512인데도 차이가 나는 주된 후보 요인(수정됨):

- ~~llama-server는 `--parallel 4`라 컨텍스트를 슬롯 4개로 나눈다(슬롯당 20224) — **반증됨**.
  parallel 1(슬롯당 80128)에서도 prefill t/s가 동일했다. 짧은 프롬프트는 양쪽 모두 ubatch 한 번에
  처리되므로, slot 스케줄 오버헤드나 컨텍스트 분할이 t/s에 영향을 주지 않는다는 뜻이다.~~
- 남은 후보: llama-server의 prefill 요청 처리 경로(슬롯 상태 머신 → `process_batch` 호출 방식,
  batch 구성·정렬, `--spec-type ngram-mod`의 prefill 단계 개입 여부)이 golbang의
  `prefill_max = n_batch(512)` 직결 경로보다 토큰당 오버헤드가 크다.
  짧은 프롬프트일수록(33–50토큰) 이 고정 오버헤드가 t/s에 크게 반영된다 —
  그래서 격차가 프롬프트가 짧을수록 큰 것이다(아래 §4-4).

### 3.3 wall time (전체 요청 소요시간)

| 밴드 | golbang wall | llama-server wall |
|------|-------------:|------------------:|
| para84 | 6.53 s | 6.62 s |
| mid200 | 15.28 s | 15.29 s |
| long1024 | 77.69 s | 77.05 s |

decode가 지배하는 구간이라 wall도 동률이다.

## 4. 관찰

1. **reasoning 동작이 달랐다.** llama-server는 `--reasoning on`이라 모든 응답이
   reasoning 토큰으로 채워졌다(para84: reasoning 418 + content 0, long1024: reasoning 2439 + content 2516).
   golbang은 템플릿 기본값(reasoning off)이라 content만 생성했다.
   decode t/s는 토큰 단위라 이 차이와 무관하지만, **사용자가 받는 출력의 성격이 다르다** —
   llama-server 쪽은 max_tokens 84/200을 reasoning에 다 써서 본문이 비어 있었다.
2. **`--spec-type ngram-mod` 효과 미관측.** llama.cpp 로그에 spec 관련 줄이 없고,
   decode t/s가 golbang(스펙큘 없음)과 동일하다. 이 모델·워크로드에서는
   ngram 스펙큘이 draft 수락을 못 하거나(수락률 0) 경로가 안 걸린 것으로 보인다.
3. **smoke16의 golbang 출력 길이가 1~3토큰으로 흔들린다** (`OK` vs `</think>\n\nOK`).
   temperature=0에서도 재현되는 미세 비결정성(이전 측정 때도 기록됨). 판정 영향 없음.
4. **prefill 격차는 프롬프트가 짧을수록 크다**(1.62× → 1.43×).
   긴 프롬프트로 갈수록 ubatch 512 블록 효율이 양쪽 모두 살아나 격차가 좁혀진다.
   대화급(수천 토큰) prefill에서는 격차가 더 줄어들 가능성이 있다.
   parallel 1 재측정(§3.2)으로 보정하면, 밴드별 prefill **절대 시간 차이**가
   para84 +125 ms / mid200 +119 ms / long1024 +106 ms로 **프롬프트 길이와 무관하게 ~110–125 ms 수준**이다.
   이는 요청당 고정 오버헤드(슬롯 스케줄링·batch 구성·API 경로)가 원인이라는 가설과 일치한다.

## 5. 결론

- **decode: 동률.** 같은 CUDA 커널·같은 양자화에서 Rust 서빙 레이어의 오버헤드는
  체감 속도에 안 나타난다(13.2–13.4 t/s, 토큰당 ~75 ms).
- **prefill: golbang 우위 (1.4–1.7×, 단일 요청·짧은 프롬프트 기준).**
  ubatch 512가 같은데도 격차가 나는 원인은 **llama-server의 서빙 레이어 prefill 경로**로
  좁혀졌다 — `--parallel 4`→`1` 재측정(§3.2)에서 슬롯당 컨텍스트를 풀 80128로 늘려도
  prefill t/s가 ±0.6% 이내라 슬롯 분할 가설은 반증됐다. 밴드별 절대 시간 차이는
  ~106–125 ms로 고정 오버헤드 성격을 보인다.
- **동시 처리: llama-server 우위(설계상).** `--parallel 4`로 슬롯 4개 동시 서비스 가능,
  golbang은 `n_parallel=1 + queue_size=1`(거절). 이번 벤치는 단일 요청이므로 측정되지 않음.
- **reasoning 제어: 유닛 설정 의존.** llama-server 쪽에 `--reasoning on`이 걸려 있어
   짧은 max_tokens에서는 reasoning만 가득 채우고 본문이 비어 있었다. golbang은 템플릿 기본값(off).
  비교할 때 이 차이를 감안해야 한다.

## 종료 시 상태

- 1차 측정 후 (15:04): `golbang-cuda-qwen38.service` active, `qwen3.8-q2.service` inactive
- **parallel 1 재측정 후 (16:10): 원복 완료** — `qwen3.8-q2.service` 정지 후 `--parallel 1` → `4`로
  서비스 파일을 원복(테스트용 임시 수정), `golbang-cuda-qwen38.service` 재기동(active, 스모크 응답 확인).
  `qwen3.8-q2.service`는 다시 inactive.
