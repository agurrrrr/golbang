# P3 — 성능 & 안정화: prefix cache, chunked prefill, 메트릭

> **이슈:** #20 · **선행:** P2 (동시성 코어) · **후행:** P4 (선택)
> **성격:** 동작하는 동시성 코어 위에서 **성능 최적화 + 관측성 + 재현성**을 확보한다.
> llama.cpp `llama-server`와 동일 조건 벤치로 위치를 객관화한다.

---

## 1. 목표

1. **prefix caching** — 공통 프롬프트의 KV를 재사용해 TTFT 단축
2. **chunked prefill** — 긴 프롬프트가 decode를 굶기지 않게 분할 처리
3. **메트릭** — `/metrics`로 관측성 확보
4. **재현성** — llama.cpp 소스 빌드(P0에서 미뤄둔 (B)안)로 `.so` 의존 제거
5. **벤치 비교** — llama-server 대비 위치 측정

**완료 기준:** llama.cpp `llama-server`와 **동일 모델·동일 하드웨어**로 벤치 비교를 수행하고 결과를 기록한다.

---

## 2. 배경 / 근거

- P2에서 continuous batching으로 TTFT/지연 개선을 입증했다.
- P3는 "개별 최적화"로 한 단계 더 간다:
  - 시스템 프롬프트처럼 **반복되는 접두사**의 KV를 재계산하지 않으면 TTFT가 크게 줄어든다.
  - 긴 prefill이 한 iteration을 오래 점유하면 다른 슬롯의 decode가 밀린다(ITL 악화). 이를 쪼갠다.
- vllm-gfx906 포크에 gfx906 attention tiling 패치가 있으므로, 필요 시 이식을 검토한다.

---

## 3. 산출물

| 산출물 | 위치 | 설명 |
|--------|------|------|
| prefix cache | `golbang-core/src/prefix_cache.rs` | 공통 접두사 KV 재사용 |
| chunked prefill | `golbang-core/src/scheduler.rs` | prefill을 청크로 분할 |
| 메트릭 | `golbang-server/src/metrics.rs` | `/metrics` 엔드포인트 |
| 재현성 빌드 | `golbang-sys/build.rs` | cmake+hipcc 소스 빌드 ((B)안) |
| 벤치 결과 | `docs/bench/` | llama-server 비교 데이터 |

---

## 4. 단계별 작업

### 4.1 prefix caching
- [ ] 프롬프트 토큰 시퀀스의 **최장 공통 접두사**를 탐색(해시/트라이 등).
- [ ] 일치하는 접두사의 KV를 재사용하고, 나머지 suffix만 prefill.
- [ ] 시스템 프롬프트가 고정된 워크로드에서 TTFT 개선 측정.
- [ ] ⚠️ llama.cpp KV 캐시 API(`llama_kv_cache_seq_*`)로 seq 복사/이동이 가능한지 확인 후 구현.

### 4.2 chunked prefill
- [ ] 긴 프롬프트를 N토큰 청크로 쪼개 여러 iteration에 나눠 prefill.
- [ ] prefill 청크 사이에 다른 슬롯의 decode를 끼워 넣어 ITL 저하 방지.
- [ ] 스케줄러가 "이번 iteration은 prefill 청크 X개 + decode Y개" 같은 예산(budget) 정책을 갖도록.

### 4.3 메트릭 / 관측성
- [ ] `/metrics` (Prometheus 텍스트 포맷):
  - `tokens_generated_total`, `prompt_tokens_total`
  - TTFT / ITL 히스토그램
  - 슬롯 점유율(active/total), 큐 깊이, 거절(503) 횟수
- [ ] 요청별 tracing span.

### 4.4 재현성 빌드 (선택)
- [ ] P0에서 미뤄둔 (B)안: `build.rs`가 cmake+hipcc로 llama.cpp(ggml-hip, gfx906)를 직접 빌드.
- [ ] llama.cpp를 git submodule 또는 고정 커밋으로 고정해 재현 가능하게.
- [ ] 기존 `.so` 재사용((A)안)과 토글 가능하게 feature/환경변수 분기.

### 4.5 gfx906 커널 튜닝 (선택)
- [ ] `vllm-gfx906` 포크의 gfx906 attention tiling 패치 이식 가능성 검토.
- [ ] llama.cpp `GGML_HIP_MMQ_MFMA` 등 gfx906 최적화 플래그 재확인.

### 4.6 벤치 비교
- [ ] 동일 GGUF, 동일 머신에서 `llama-server` vs `golbang`:
  - 단일 요청 처리량(tok/s)
  - 동시 N요청 TTFT/ITL/총 latency
  - prefix cache on/off 효과
- [ ] 결과를 `docs/bench/`에 표로 기록.

---

## 5. 완료 기준 (Definition of Done)

- [ ] prefix cache로 반복 프롬프트 TTFT가 개선됨을 측정
- [ ] chunked prefill로 긴 프롬프트 중에도 다른 슬롯 ITL이 안정적
- [ ] `/metrics`에서 TTFT/ITL/슬롯 점유율 조회 가능
- [ ] **llama-server와 동일 조건 벤치 비교 결과를 docs에 기록**
- [ ] (선택) 소스 빌드로 재현성 확보
- [ ] 산출물 커밋

---

## 6. 리스크 & 완화

| 리스크 | 영향 | 완화 |
|--------|------|------|
| llama.cpp KV API로 prefix 재사용이 제한적 | prefix cache 구현 불가/부분 구현 | API 지원 범위를 먼저 실험. 안 되면 슬롯 내 연속 프롬프트 재사용 등 축소안 적용 |
| chunked prefill이 llama.cpp batch 모델과 충돌 | 구현 난항 | llama.cpp가 지원하는 batch 분할 방식에 맞춰 설계. 무리하면 P3 범위에서 제외하고 기록 |
| 소스 빌드(cmake+hipcc) 시간/환경 문제 | 빌드 지연 | (A)안 재사용을 기본으로 유지, (B)는 선택. CI가 아닌 로컬 검증 우선 |
| 벤치 공정성 | 비교 신뢰도 ↓ | 동일 모델/양자화/컨텍스트 길이/GPU 클럭 고정. `set_gpu_clocks.sh` 활용 |

---

## 7. 예상 공수

**2 ~ 3일.** (prefix cache / chunked prefill은 llama.cpp API 지원 범위에 따라 변동 큼. 메트릭·벤치는 비교적 일정.)

---

## 8. 다음 단계로 넘기는 것

- P3까지 끝나면 **실사용 가능한 golbang 서버**가 된다.
- P4(순수 Rust 커널)는 선택 사항으로, hot-path 프로파일링 결과가 있을 때 착수 여부를 결정한다.
