# P2 — 동시성 코어: 스케줄 정책이 교체 가능한 배치 루프

> **이슈:** #25 · **선행:** P1 (단일 요청 E2E) · **후행:** P3
> **성격:** 오케스트레이션을 Rust로 소유해 **정책을 우리 코드로 바꾸게** 하는 단계다.
> llama-server도 이미 슬롯 + 통합 batch + `llama_decode` continuous batching이 **기본 내장**이다.
> 따라서 이 단계의 목표는 "CB를 도입하는 것"이 아니다.

---

## 1. 목표

여러 클라이언트 요청이 동시에 들어와도:

1. HTTP / 스케줄이 `llama_decode`에 **블로킹되지 않는다** (수신·503·취소 토큰이 살아 있다).
2. **iteration 경계**에서 join / evict / 취소가 **관측 가능**하다.
3. 스케줄 정책(join / evict / 우선순위 / 나중에 chunked prefill)이 **Rust 코드로 교체 가능**하다.

**완료 기준:**
- 회귀: 후발 요청 TTFT가 P1 순차 대기보다 낮다.
- 경쟁: 동일 모델·`n_parallel`·프롬프트로 llama-server와 TTFT/ITL/tok/s/VRAM을 **기록**한다.
  **llama-server 대비 처리량 우위는 필수 조건이 아니다.** 동률 + 제어면 개선이면 통과.

---

## 2. 배경 / 근거 (무엇을 고치고, 무엇을 주장하지 않는가)

### llama-server가 이미 하는 것

- `--cont-batching`이 **기본 켜짐**. 슬롯 + 통합 batch + `llama_decode`.
- HTTP는 cpp-httplib 스레드에서 받고, 추론 스레드는 큐를 비운 뒤 `update_slots()`한다.
- 신규 요청 join은 **이번 `llama_decode`가 반환된 다음**이다.

### 양쪽 공통 하한

golbang P2도 `spawn_blocking` 안의 `llama_decode`가 끝날 때까지 다음 iteration에 join/evict하지 못한다.
**취소 지연의 하한은 양쪽 모두 decode 1회다.**

따라서 아래는 llama-server 대비 차별화가 **아니다.** 지시서·커밋·이슈에 쓰지 않는다.

- "CB 자체 도입"
- "decode 도중에 신규 요청을 배치에 삽입"
- "HTTP가 decode에 막혀 있다" (llama-server HTTP는 이미 별도 스레드)
- "GPU 처리량이 llama-server보다 높다" (P2 성공 조건 아님)

### golbang이 실제로 가져가는 것

| 목표 | 내용 |
|------|------|
| **정책 제어** | join / evict / 우선순위 / (P3) chunked prefill을 Rust에서 직접 교체 |
| **단순한 제어면** | 연결 끊김 → 슬롯 회수 경로가 짧고 명시적 |
| **즉시 503** | bounded 큐. 과부하 시 대기 없이 거절 |
| **경량 표면** | 임베딩 / rerank / 스펙큘레이티브 없이 채팅 스트리밍 |

구현 수단은 비동기 분리 + iteration 경계 스케줄 + CancellationToken + bounded mpsc다.
수단을 목표와 바꾸지 않는다.

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
            │   - 정책 객체: 누가 join/evict 할지 결정        │
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
HTTP/스케줄링은 async로 유지해 **decode 중에도 수신·503·취소 토큰이 멈추지 않게** 한다.

join의 정확한 의미: **이번 `llama_decode` 반환 직후**, 빈 슬롯이 있으면 대기 요청을 넣는다.
"decode 실행 중에 배치를 고쳐 끼운다"가 아니다.

---

## 4. 산출물

| 산출물 | 위치 | 설명 |
|--------|------|------|
| 슬롯 풀 | `golbang-core/src/slot.rs` | N개 슬롯, 각 KV/상태 보유 |
| 스케줄 정책 | `golbang-core/src/policy.rs` | join/evict/우선순위 trait. 기본 FIFO |
| 스케줄러 | `golbang-core/src/scheduler.rs` | iteration 루프. 정책만 호출 |
| 배처 | `golbang-core/src/batch.rs` | 활성 슬롯 → 통합 batch 편성 |
| GPU 워커 | `golbang-core/src/engine.rs` | spawn_blocking 추론 전담 |
| 취소/백프레셔 | `golbang-core/src/` | CancellationToken, bounded mpsc |
| 서버 연동 | `golbang-server/src/` | handler → scheduler → SSE |
| 측정 | `docs/bench/p2.md` | P1 회귀 + llama-server 기록 |

---

## 5. 단계별 작업

### 5.1 GPU 추론 격리

- [ ] `llama_decode` 호출을 `spawn_blocking`(또는 전용 스레드)로 이동. async 런타임을 블로킹하지 않음.
- [ ] 모델/컨텍스트는 단일 GPU이므로 **추론은 한 번에 하나의 batch만** 실행(뮤텍스/단일 워커).
- [ ] decode가 돌아가는 동안 HTTP 수신, 503, 취소 토큰이 살아 있음을 테스트로 확인.

### 5.2 슬롯 풀 + 스케줄러 + 정책

- [ ] `n_parallel`(슬롯 수)만큼 슬롯 생성. 각 슬롯: 상태(Empty/Prefilling/Decoding), 생성 토큰 버퍼, 취소 토큰.
- [ ] 스케줄 정책을 trait으로 분리. 기본 구현은 FIFO. P3의 chunked prefill은 이 trait에 붙인다.
- [ ] **iteration 루프:**
  1. 정책이 Empty 슬롯 + 대기 요청을 매칭 → prefill 편성
  2. 활성 슬롯들을 통합 batch로 편성 (각자 다음 토큰 1개씩)
  3. GPU 워커에 batch 제출 → logits 수신 (**여기서 블로킹**)
  4. 슬롯별 샘플링 → 토큰 방출(SSE로 흘려보냄)
  5. EOS/max_tokens/취소 슬롯 evict → Empty로 회수
- [ ] **join 타이밍:** `llama_decode` **반환 직후** 대기 큐를 보고 빈 슬롯에 넣는다.

### 5.3 선제 취소 / 타임아웃

- [ ] 각 요청에 `CancellationToken`. 클라이언트 연결 끊김(axum drop 감지) → 토큰 취소.
- [ ] 스케줄러가 **매 iteration 시작**에 취소 여부를 확인 → evict + KV 해제.
- [ ] 취소가 HTTP 쪽에서 즉시 접수되는 것과, 슬롯 회수가 **다음 decode 경계**인 것을 구분해서 로그/테스트에 적는다. 하한은 decode 1회다.
- [ ] 타임아웃(요청당 최대 생성 시간) 지원.

### 5.4 백프레셔

- [ ] 요청 수신 큐를 **bounded mpsc**로. 가득 차면 핸들러가 즉시 `503 Service Unavailable` + `Retry-After`.
- [ ] decode 중에도 503이 나와야 한다. decode가 끝날 때까지 핸들러가 멈추면 실패.

### 5.5 서버 연동

- [ ] P1의 단일 `generate` 호출을 스케줄러 제출 방식으로 교체.
- [ ] 각 요청은 자기 슬롯의 토큰 스트림만 구독해 SSE로 흘려보냄.

### 5.6 측정 (완료 기준의 핵심) — 비교를 둘로 나눈다

#### (a) 회귀 — 자기 자신보다 나은지

- [ ] 비교: **golbang P1 단일 루프** vs **golbang P2 스케줄러**.
- [ ] 동일 프롬프트 `n_parallel`개 동시 요청.
- [ ] 기대: 후발 요청 TTFT가 P1 순차 대기보다 **낮음**.
- [ ] 지표: 개별 TTFT, ITL, 총 latency, tok/s.

#### (b) 경쟁 — llama-server와 같은 조건으로 기록

- [ ] 동일 GGUF, 동일 `n_parallel`, 동일 프롬프트, 동일 GPU 클럭.
- [ ] 지표: TTFT / ITL / tok/s / VRAM.
- [ ] **처리량 우위는 필수 조건이 아니다.** 동률 + 제어면(정책 교체 / 즉시 503 / 짧은 취소 경로)이면 통과.
- [ ] 결과를 `docs/bench/p2.md`에 표로 남긴다. 기대 수치를 사후에 맞추지 말고, 측정값을 있는 그대로 적는다.

---

## 6. 완료 기준 (Definition of Done)

- [ ] decode 중에도 HTTP 수신·503·취소 토큰이 살아 있음 (테스트 또는 재현 로그)
- [ ] `llama_decode` 반환 직후 빈 슬롯에 join됨을 로그/측정으로 확인
- [ ] 연결 끊김이 다음 iteration에서 슬롯 회수로 이어짐
- [ ] 큐 초과 시 **즉시** 503 (decode 대기가 아님)
- [ ] 스케줄 정책이 코드상 교체 가능한 경계(trait/모듈)로 분리됨
- [ ] (a) P1 대비 후발 TTFT 개선이 숫자로 기록됨
- [ ] (b) llama-server와 동일 조건 측정이 `docs/bench/p2.md`에 기록됨
- [ ] 산출물 커밋

---

## 7. 리스크 & 완화

| 리스크 | 영향 | 완화 |
|--------|------|------|
| llama.cpp batch API가 다중 시퀀스를 기대와 다르게 처리 | 스케줄러 구현 난항 | llama-server 슬롯 처리(`llama_batch` + seq/pos)를 참고 |
| KV 캐시 관리(슬롯별 독립 KV) | 메모리 부족/오염 | 이 SHA의 API는 `llama_memory_seq_*`. 슬롯 수×컨텍스트가 VRAM에 맞는지 검증 |
| 단일 GPU 직렬화로 처리량이 llama-server와 같음 | "이겨야 한다"는 착각 | 처리량 승리는 DoD가 아님. 제어면·TTFT 회귀만 필수 |
| unsafe 상태를 여러 태스크가 공유 | 데이터 레이스/UB | GPU 접근은 단일 워커. async는 채널만 |

---

## 8. 예상 공수

**2 ~ 4일.** 프로젝트에서 가장 공수가 크다.
llama.cpp의 batch/KV API와 스케줄러 설계를 맞추는 데 시간이 든다. 측정 인프라도 포함.

---

## 9. 다음 단계로 넘기는 것

- P2가 끝나면 정책이 교체 가능한 루프가 있다. P3는 이 위에 prefix cache, chunked prefill, 메트릭을 얹는다.
- P2에서 만든 슬롯/정책 구조는 P3의 prefix cache 기반이 된다.
- GPU 커널 튜닝과 llama-server 처리량 추월은 P2가 아니다.
