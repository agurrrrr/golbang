# P3 — 성능 & 안정화: prefix cache, chunked prefill, 메트릭

> **이슈:** #26 · **선행:** P2 (동시성 코어) · **후행:** P4 (선택)
> **성격:** 동작하는 스케줄 루프 위에서 **성능 최적화 + 관측성 + 재현성**을 확보한다.
> llama-server와 **동일 조건** 벤치로 위치를 객관화한다.

---

## 1. 목표

1. **prefix caching** — 공통 프롬프트의 KV를 재사용해 TTFT 단축
2. **chunked prefill** — 긴 프롬프트가 decode를 굶기지 않게 분할 처리
3. **메트릭** — `/metrics`로 관측성 확보
4. **재현성** — llama.cpp 소스 빌드(P0에서 미뤄둔 (B)안)로 `.so` 의존 제거
5. **벤치 비교** — llama-server 대비 위치 측정

**완료 기준:** llama-server와 **동일 GGUF · 동일 `-c/-np/-ctk/-ctv` · 동일 GPU 클럭**으로
벤치 비교를 수행하고 결과를 기록한다.

---

## 2. 배경 / 근거

- P2에서 정책 교체 가능한 스케줄 루프와 P1 대비 TTFT 회귀를 입증했다.
- P3는 그 루프 위의 최적화다:
  - 시스템 프롬프트처럼 **반복되는 접두사**의 KV를 재계산하지 않으면 TTFT가 줄어든다.
  - 긴 prefill이 한 iteration을 오래 점유하면 다른 슬롯의 decode가 밀린다. 이를 쪼갠다.
- llama-server도 ubatch/logical batch로 chunked prefill을 이미 지원한다. "도입"이 아니라 **우리 정책으로 제어**하는 것이 목적이다.
- vllm-gfx906 포크에 gfx906 attention tiling 패치가 있으므로, 필요 시 이식을 검토한다.

---

## 3. 산출물

| 산출물 | 위치 | 설명 |
|--------|------|------|
| prefix cache | `golbang-core/src/prefix_cache.rs` | 공통 접두사 KV 재사용 (또는 축소안) |
| chunked prefill | `golbang-core/src/scheduler.rs` + policy | prefill을 청크로 분할 |
| 메트릭 | `golbang-server/src/metrics.rs` | `/metrics` 엔드포인트 |
| 재현성 빌드 | `golbang-sys/build.rs` | cmake+hipcc 소스 빌드 ((B)안) |
| 벤치 결과 | `docs/bench/p3.md` | 공정 조건 + 측정표 |

---

## 4. 단계별 작업

### 4.0 prefix cache API 스파이크 (체크리스트 맨 앞)

구현에 들어가기 전에 **복사 가능 여부부터** 확인한다.

로컬 `llama.h`(SHA `5b474eb69`) 기준 이름은 `llama_kv_cache_seq_*`가 아니라
**`llama_memory_seq_*`** 이다.

- [ ] 스파이크: `llama_memory_seq_cp` / `llama_memory_seq_rm` / `llama_memory_seq_keep`로
      seq 간 KV 복사가 되는지 최소 프로그램으로 확인.
- [ ] 가능하면 공통 접두사 KV 재사용으로 구현.
- [ ] **불가하면 축소:** 슬롯 안에서 시스템 프롬프트 KV만 재사용. 전역 prefix store는 하지 않는다.
- [ ] 스파이크 결과(가능/불가 + 사용한 심볼)를 `docs/bench/p3.md` 맨 위에 기록한 뒤에야 4.1로 간다.

### 4.1 prefix caching

- [ ] 4.0이 "가능"일 때만: 프롬프트 토큰의 최장 공통 접두사를 탐색(해시/트라이 등).
- [ ] 일치 접두사 KV를 재사용하고 suffix만 prefill.
- [ ] 시스템 프롬프트가 고정된 워크로드에서 TTFT 개선 측정.

### 4.2 chunked prefill

- [ ] 긴 프롬프트를 N토큰 청크로 쪼개 여러 iteration에 나눠 prefill.
- [ ] prefill 청크 사이에 다른 슬롯의 decode를 끼워 넣어 ITL 저하 방지.
- [ ] 스케줄 **정책**(P2 trait)에 "이번 iteration은 prefill 청크 X개 + decode Y개" 예산을 넣는다.
- [ ] llama.cpp batch 분할과 충돌하면 무리하지 말고 P3 범위에서 제외하고 기록.

### 4.3 메트릭 / 관측성

- [ ] `/metrics` (Prometheus 텍스트 포맷):
  - `tokens_generated_total`, `prompt_tokens_total`
  - TTFT / ITL 히스토그램
  - 슬롯 점유율(active/total), 큐 깊이, 거절(503) 횟수
- [ ] 요청별 tracing span.

### 4.4 재현성 빌드 (선택, ADR과 맞춤)

P0는 **(A) 기존 `.so` + SHA 고정**이다. ADR의 submodule+cmake 서술은 **이 절(B)** 의 이야기다.

> **선택 여부 명시:** P3 DoD에서 4.4는 **"(선택) 소스 빌드"로 표기된 선택 항목**이다.
> 필수는 아니다 — (A)로도 벤치·메트릭은 완료할 수 있다. 다만 ADR 리스크에
> "P0 구현자가 (B)를 먼저 하지 않는다"고 명시되어 있으므로, **P0에서는 절대 (B)를 하지 않는다.**

- [ ] `build.rs`가 cmake+hipcc로 llama.cpp(ggml-hip, gfx906)를 직접 빌드.
- [ ] llama.cpp를 git submodule 또는 **고정 커밋**으로 고정.
- [ ] (A)/(B)를 feature 또는 환경변수로 토글.

### 4.5 gfx906 커널 튜닝 (선택)

- [ ] `vllm-gfx906` 포크의 gfx906 attention tiling 패치 이식 가능성 검토.
- [ ] llama.cpp `GGML_HIP_MMQ_MFMA` 등 gfx906 최적화 플래그 재확인.

### 4.6 벤치 비교 — 공정성 조건을 먼저 박는다

아래를 맞추지 않은 숫자는 기록하지 않는다.

| 고정 항목 | 값 |
|-----------|-----|
| GGUF | 양측 동일 파일 (경로·SHA256) |
| 컨텍스트 | 동일 `-c` |
| 병렬 | 동일 `-np` / `n_parallel` |
| KV 타입 | 동일 `-ctk` / `-ctv` |
| GPU 클럭 | `set_gpu_clocks.sh`로 고정 후 측정 |
| 프롬프트 | 동일 세트 (단일 / 동시 N / prefix 반복) |

- [ ] 단일 요청 tok/s
- [ ] 동시 N요청 TTFT / ITL / 총 latency / VRAM
- [ ] prefix cache on/off (4.0이 가능일 때만)
- [ ] 결과를 `docs/bench/p3.md`에 표로 기록. 사후 목표치를 끼워 맞추지 않는다.

---

## 5. 완료 기준 (Definition of Done)

- [ ] 4.0 스파이크 결과가 문서화됨 (가능 → 구현, 불가 → 축소안)
- [ ] prefix 경로가 선택된 범위에서 TTFT 개선을 측정했거나, 축소 사유가 기록됨
- [ ] chunked prefill이 동작하거나, 제외 사유가 기록됨
- [ ] `/metrics`에서 TTFT/ITL/슬롯 점유율 조회 가능
- [ ] **공정 조건**을 명시한 llama-server 벤치가 `docs/bench/p3.md`에 있음
- [ ] **(선택)** 소스 빌드 (B)로 재현성 확보 — 선택 항목. (A)만으로도 벤치·메트릭은 완료 가능
- [ ] 산출물 커밋

---

## 6. 리스크 & 완화

| 리스크 | 영향 | 완화 |
|--------|------|------|
| `llama_memory_seq_*`로 prefix 재사용 불가 | 전역 cache 포기 | 4.0을 맨 앞에 둔 이유. 슬롯 내 재사용으로 축소 |
| chunked prefill이 batch 모델과 충돌 | 구현 난항 | llama.cpp가 지원하는 분할에 맞춤. 무리하면 제외하고 기록 |
| 소스 빌드(cmake+hipcc) 시간/환경 | 빌드 지연 | (A)를 기본 유지. (B)는 선택 |
| 벤치 공정성 | 비교 신뢰도 ↓ | 위 표(GGUF/`-c`/`-np`/`-ctk`/`-ctv`/클럭)를 어기면 숫자를 올리지 않음 |

---

## 7. 예상 공수

**2 ~ 3일.** prefix cache / chunked prefill은 API 지원 범위에 따라 변동이 크다.

---

## 8. 다음 단계로 넘기는 것

- P3까지 끝나면 실사용 가능한 golbang 서버가 된다.
- P4는 선택. 아래를 **모두** 만족할 때만 착수한다 (P4 지시서 gate).
