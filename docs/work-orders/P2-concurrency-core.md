# P2 — 동시성 코어: continuous batching 스케줄러

> **이슈:** #19 · **선행:** P1 (단일 요청 E2E) · **후행:** P3
> **성격:** **프로젝트의 핵심 차별화.** llama.cpp의 동시성 한계를 실제로 개선하는 단계다.
> 하이브리드여도 동시성을 결정하는 오케스트레이션은 100% 우리 코드이므로, 개선 목표는 온전히 달성된다.

---

## 1. 목표

여러 클라이언트 요청이 동시에 들어와도:
1. decode 중에 **새 요청이 배치에 join**할 수 있고
2. 연결 끊김/타임아웃 시 **슬롯이 즉시 회수**되며
3. 과부하 시 **즉시 503**으로 백프레셔를 건다.

**완료 기준:** `n_parallel`개 동시 요청 시 개별 TTFT/지연이 **단일 동기 루프 대비 개선됨을 측정**으로 입증한다.

---

## 2. 배경 / 근거 (왜 개선이 되는가)

### llama.cpp(llama-server)의 한계
llama-server는 모든 활성 슬롯을 **하나의 통합 배치로 묶어 `llama_decode()`를 동기 호출**한다.
- decode가 반환되기 전까지 **새 요청이 배치에 끼어들지 못함** (head-of-line blocking).
- 취소도 decode 경계까지 기다려야 해서 **자원 회수가 늦음**.
- HTTP 수신/스케줄링/GPU 추론이 사실상 한 루프에 얽혀 있어 **서로 블로킹**.

### golbang의 개선 포인트
| 개선 | 내용 |
|------|------|
| **비동기 분리** | HTTP 수신(axum) / 스케줄링 / GPU 추론을 별도 tokio 태스크로 격리 |
| **iteration-level continuous batching** | 매 iteration마다 슬롯 join/evict — decode 중에도 신규 요청 삽입 |
| **선제 취소** | `CancellationToken`으로 연결 끊김 즉시 슬롯 회수 |
| **백프레셔** | bounded mpsc 큐. 가득 차면 즉시 503 (무한 대기 방지) |

> **중요:** GPU 커널(ggml-hip)은 재사용하지만, 위 4가지는 전부 Rust 오케스트레이션 영역이다.
> 따라서 하이브리드 구조에서도 동시성 개선은 온전히 달성된다.

---

## 3. 아키텍처 (설계 골격)

```
            ┌──────────────────────────────────────────────┐
 HTTP req → │ axum handlers (tokio, async)                 │
            │   └─ 요청을 Job으로 변환, oneshot/mpsc로 결과 수신 │
            └───────────────┬──────────────────────────────┘
                            │ bounded mpsc (백프레셔)
                            ▼
            ┌──────────────────────────────────────────────┐
            │ Scheduler 태스크 (async)                      │
            │   - 슬롯 풀 관리 (slot pool)                   │
            │   - 매 iteration: join / evict / batch 편성    │
            │   - 취소 토큰 감시 → 슬롯 회수                  │
            └───────────────┬──────────────────────────────┘
                            │ batch 작업 지시
                            ▼
            ┌──────────────────────────────────────────────┐
            │ GPU 추론 전담 (spawn_blocking / 전용 스레드)    │
            │   - llama_decode(통합 batch) 실행              │
            │   - logits → 각 슬롯별 샘플링 → 토큰 방출       │
            └──────────────────────────────────────────────┘
```

**핵심 분리:** GPU 추론은 blocking이므로 `tokio::task::spawn_blocking`(또는 전용 OS 스레드 + 채널)에 격리하고,
HTTP/스케줄링은 async로 유지해 **decode 중에도 수신·스케줄링이 멈추지 않게** 한다.

---

## 4. 산출물

| 산출물 | 위치 | 설명 |
|--------|------|------|
| 슬롯 풀 | `golbang-core/src/slot.rs` | N개 슬롯, 각 KV/상태 보유 |
| 스케줄러 | `golbang-core/src/scheduler.rs` | continuous batching 루프 |
| 배처 | `golbang-core/src/batch.rs` | 활성 슬롯 → 통합 batch 편성 |
| GPU 워커 | `golbang-core/src/engine.rs` | spawn_blocking 추론 전담 |
| 취소/백프레셔 | `golbang-core/src/` | CancellationToken, bounded mpsc |
| 서버 연동 | `golbang-server/src/` | handler → scheduler → SSE |

---

## 5. 단계별 작업

### 5.1 GPU 추론 격리
- [ ] `llama_decode` 호출을 `spawn_blocking`(또는 전용 스레드)로 이동. async 런타임을 블로킹하지 않음.
- [ ] 모델/컨텍스트는 단일 GPU이므로 **추론은 한 번에 하나의 batch만** 실행(뮤텍스/단일 워커). 동시성은 "여러 요청을 한 batch에 묶는 것"으로 달성.

### 5.2 슬롯 풀 + 스케줄러
- [ ] `n_parallel`(슬롯 수)만큼 슬롯 생성. 각 슬롯: 상태(Empty/Prefilling/Decoding), 생성 토큰 버퍼, 취소 토큰.
- [ ] **iteration 루프:**
  1. Empty 슬롯 + 대기 요청 매칭 → prefill 편성
  2. 활성 슬롯들을 통합 batch로 편성 (각자 다음 토큰 1개씩)
  3. GPU 워커에 batch 제출 → logits 수신
  4. 슬롯별 샘플링 → 토큰 방출(SSE로 흘려보냄)
  5. EOS/max_tokens/취소 슬롯 evict → Empty로 회수
- [ ] **decode 중 신규 요청 삽입:** iteration 경계마다 대기 큐를 확인해 빈 슬롯이 있으면 즉시 join.

### 5.3 선제 취소 / 타임아웃
- [ ] 각 요청에 `CancellationToken`. 클라이언트 연결 끊김(axum drop 감지) → 토큰 취소.
- [ ] 스케줄러가 매 iteration마다 취소 여부 확인 → 취소된 슬롯 즉시 evict + KV 해제.
- [ ] 타임아웃(요청당 최대 생성 시간) 지원.

### 5.4 백프레셔
- [ ] 요청 수신 큐를 **bounded mpsc**로. 가득 차면 핸들러가 즉시 `503 Service Unavailable` + `Retry-After`.
- [ ] 슬롯 가득 참 + 큐 가득 참 상태에서도 서버가 죽지 않고 부하를 거절.

### 5.5 서버 연동
- [ ] P1의 단일 `generate` 호출을 스케줄러 제출 방식으로 교체.
- [ ] 각 요청은 자기 슬롯의 토큰 스트림만 구독해 SSE로 흘려보냄.

### 5.6 측정 (완료 기준의 핵심)
- [ ] 벤치 스크립트: 동일 프롬프트 `n_parallel`개 동시 요청.
- [ ] 비교 대상: **(a) golbang 단일 루프(P1 방식)** vs **(b) golbang continuous batching(P2)**.
- [ ] 지표: 개별 요청 **TTFT**(첫 토큰까지), **ITL**(토큰 간 간격), 총 latency, 처리량(tok/s).
- [ ] P2에서 동시 요청의 TTFT/지연이 단일 루프 대비 개선됨을 수치로 기록.

---

## 6. 완료 기준 (Definition of Done)

- [ ] `n_parallel`개 동시 요청이 서로 블로킹 없이 각자 스트리밍됨
- [ ] decode 도중 새 요청이 배치에 join함을 로그/측정으로 확인
- [ ] 연결 끊음 시 슬롯이 즉시 회수됨을 확인
- [ ] 큐 초과 시 즉시 503
- [ ] **TTFT/지연 개선이 측정 데이터로 입증**됨 (docs에 결과 기록)
- [ ] 산출물 커밋

---

## 7. 리스크 & 완화

| 리스크 | 영향 | 완화 |
|--------|------|------|
| llama.cpp batch API가 다중 시퀀스를 기대와 다르게 처리 | continuous batching 구현 난항 | `llama_batch`에 여러 시퀀스/pos를 넣는 llama.cpp의 기존 방식(예: server의 슬롯 처리)을 참고해 맞춤 |
| KV 캐시 관리(슬롯별 독립 KV) | 메모리 부족/오염 | llama.cpp의 `llama_kv_cache_*`(seq_id) API로 슬롯별 seq 구분. 슬롯 수×컨텍스트 길이가 VRAM에 맞는지 검증 |
| 단일 GPU 직렬화로 인한 한계 | 동시성 이득이 예상보다 작음 | TTFT 개선(대기 감소)에 집중. 처리량 한계는 P3에서 chunked prefill 등으로 개선 |
| unsafe 상태를 여러 태스크가 공유 | 데이터 레이스/UB | GPU 접근은 단일 워커로 직렬화. async 쪽은 채널로만 통신(공유 메모리 최소화) |

---

## 8. 예상 공수

**2 ~ 4일.** 프로젝트에서 가장 공수가 크다.
llama.cpp의 batch/KV API와 스케줄러 설계를 맞추는 데 시간이 든다. 측정 인프라도 포함.

---

## 9. 다음 단계로 넘기는 것

- P2가 끝나면 동시성 코어가 완성된다. P3는 이 위에서 **성능**(prefix caching, chunked prefill)과 **관측성**(메트릭)을 더한다.
- P2에서 만든 슬롯/스케줄러 구조는 P3의 prefix cache(공통 프롬프트 KV 재사용)의 기반이 된다.
