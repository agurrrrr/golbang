# P1 — 단일 요청 E2E: OpenAI 호환 SSE 스트리밍

> **이슈:** #24 · **선행:** P0 (FFI 추론 1회 성공) · **후행:** P2
> **성격:** P0의 unsafe FFI를 **안전한 Rust API**로 감싸고, HTTP 서버로 단일 요청을 끝까지 처리한다.
> 이 단계까진 **동시성 없음** — 요청 1개를 올바르게 스트리밍하는 데 집중한다.

---

## 1. 목표

OpenAI 호환 `POST /v1/chat/completions` 요청 1건을 받아:
1. 메시지를 프롬프트로 변환하고 토크나이즈
2. prefill + autoregressive decode를 반복하며
3. 생성되는 토큰을 **SSE(Server-Sent Events)** 로 토큰 단위 스트리밍 응답한다.

**완료 기준:** `curl` SSE에서 `choices[].delta.content`가 토큰 단위로 오고,
EOS/`max_tokens` 시 `data: [DONE]`, 빈 `messages`는 4xx.

---

## 2. 배경 / 근거

- P0에서 "로드 → 토크나이즈 → decode 1회 → logits"가 검증됐다.
- P1은 이를 반복 루프(autoregressive generation)로 확장하고, tokio/axum HTTP 경계와 연결한다.
- 아키텍처 책임 분리:
  - `golbang-core` — 모델/토크나이저/샘플러의 **안전한 Rust API** (unsafe는 `golbang-sys`에 격리)
  - `golbang-server` — axum 라우터, 요청 검증, SSE 인코딩

### chat template 결정 (2026-08-13 고정)

로컬 `llama.h`(SHA `5b474eb69`)의 `llama_chat_apply_template`는
**jinja 파서가 아니다.** 사전 정의 템플릿 목록만 지원한다.
Qwen GGUF의 `tokenizer.chat_template`(jinja)를 이 API만으로 처리할 수 없다.
`common`의 minja는 `llama.h`에 없어 추가 바인딩이 필요하다.

| 단계 | 결정 |
|------|------|
| **P1** | Qwen **ChatML을 Rust에 하드코딩**. 적용 실패 시 **raw prompt 폴백 + 경고 로그**. |
| **후속** | GGUF `tokenizer.chat_template`을 `minijinja` 등으로 적용. |

"단순 템플릿 후 개선"으로 열어 두지 않는다. 위 두 줄이 결정이다.

---

## 3. 산출물

| 산출물 | 위치 | 설명 |
|--------|------|------|
| 안전한 모델 래퍼 | `golbang-core/src/model.rs` | `Model::load`, `generate` 등 safe API |
| 토크나이저 | `golbang-core/src/tokenizer.rs` | encode/decode 래퍼 |
| 샘플러 | `golbang-core/src/sampler.rs` | temperature / top-p / top-k |
| 생성 루프 | `golbang-core/src/generate.rs` | 토큰 스트림(yield) 반환 |
| ChatML | `golbang-core/src/chat.rs` | Qwen ChatML 하드코딩 + raw 폴백 |
| HTTP 라우터 | `golbang-server/src/main.rs`, `routes.rs` | axum 엔드포인트 |
| OpenAI 스키마 | `golbang-server/src/types.rs` | request/response + SSE chunk |
| SSE 스트림 | `golbang-server/src/sse.rs` | `Sse<...>` 스트림 변환 |

---

## 4. 단계별 작업

### 4.1 golbang-core — 안전한 래퍼

- [x] `Model::load(path, params)` — `golbang-sys` unsafe 호출을 감싼 safe 생성자. RAII로 `Drop`에서 자원 해제.
- [x] 내부 상태(모델/컨텍스트/vocab)를 하나의 구조체로 캡슐화. `Send`는 되되, GPU 동시 접근은 이 단계에선 고려하지 않음(P2에서).
- [x] `encode(text) -> Vec<Token>`, `decode(tokens) -> String`.
- [x] `Sampler` — temperature, top-p, top-k 적용. logits 슬라이스를 받아 토큰 1개 반환.
- [x] `generate(prompt, params) -> impl Stream<Item = Token>`:
  - prefill(프롬프트 전체 decode 1회)
  - 반복: 마지막 토큰 decode → logits → sample → EOS면 종료, 아니면 토큰 방출
  - `max_tokens`, stop 조건 처리
  - **토큰 1개 단위로 쪼갤 것.** P2 스케줄러가 이 단위로 join/evict한다.

### 4.2 chat template (결정된 경로)

- [x] Qwen ChatML을 Rust로 하드코딩.
  - 예: `<|im_start|>system\n…<|im_end|>\n<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n`
- [x] 적용 실패(빈 메시지·알 수 없는 role 등) → 메시지를 이어 붙인 **raw prompt 폴백** + `tracing` 경고.
- [x] `llama_chat_apply_template`에 의존하지 않는다 (Qwen jinja 불가).
- [x] GGUF `tokenizer.chat_template` + minijinja는 **이 단계 범위 밖**. 이슈/후속 지시서에만 남긴다.

### 4.3 golbang-server — HTTP 계층

- [x] `axum` 라우터: `POST /v1/chat/completions`.
- [x] 요청 스키마(OpenAI 호환): `model`, `messages[]`, `temperature`, `top_p`, `max_tokens`, `stream`.
- [x] `stream: true` → SSE. `stream: false` → **비스트리밍 전체 JSON 응답으로 P1 범위에서 확정 지원** (선택이 아니라 기본 동작).
  - OpenAI 호환 서버는 비스트리밍 응답도 기본 지원해야 하므로 P2로 미루지 않는다.
  - SSE와 동일한 `choices[].message.content` 스키마를 단일 JSON으로 반환.
- [x] SSE chunk 포맷: OpenAI `chat.completion.chunk` (`choices[].delta.content`). 종료 시 `data: [DONE]`.
- [x] 에러 응답: 모델 미로드 / 검증 실패 시 OpenAI 스타일 에러 JSON.
- [x] `messages`가 비어 있으면 **4xx** (본문에 이유를 명시).

### 4.4 실행 진입점

- [x] `main.rs`: `--model` 경로, `--host/--port` 인자(clap 또는 env). 서버 기동 시 모델 1회 로드.
- [x] `tracing-subscriber`로 로깅. 요청/생성 토큰 수 로그.

### 4.5 검증

- [x] 서버 기동 후
  `curl -N -X POST localhost:PORT/v1/chat/completions -d '{"model":"...","messages":[{"role":"user","content":"안녕"}],"stream":true}'`
- [x] 각 SSE 이벤트의 `choices[].delta.content`가 **토큰 단위**로 오는지 확인.
- [x] EOS 또는 `max_tokens`에서 `data: [DONE]`으로 종료.
- [x] `{"messages":[]}` → 4xx.
- [x] `stream:false`로 전체 JSON 응답 확인 (P1 기본 지원).

---

## 5. 완료 기준 (Definition of Done)

- [x] `cargo build` 전체 크레이트 통과
- [x] `curl` SSE에서 `choices[].delta.content`가 **토큰 단위**로 출력
- [x] EOS/`max_tokens`에서 정상 종료 + `data: [DONE]`
- [x] 빈 `messages` → 4xx
- [x] `stream:false` 비스트리밍 JSON 응답도 정상 동작
- [x] ChatML 하드코딩 경로가 기본. 실패 시 raw 폴백 + 경고 로그가 남음
- [x] 산출물 커밋

---

## 6. 리스크 & 완화

| 리스크 | 영향 | 완화 |
|--------|------|------|
| ChatML이 해당 GGUF와 안 맞음 | 출력 품질 저하 | 폴백+경고. jinja는 후속. P1 DoD는 프로토콜이지 품질이 아님 |
| unsafe→safe 경계 누수 | UB/메모리 문제 | unsafe는 `golbang-sys`에만. core는 safe API만 노출. RAII Drop 철저 |
| 생성 루프 무한/미종료 | 서버 행 | `max_tokens` 상한 + EOS 감지를 테스트로 고정 |
| SSE 백프레셔 미비 | 느린 클라이언트에 버퍼 팽창 | P1은 단일 요청이라 영향 작음. 본격 백프레셔는 P2 |

---

## 7. 예상 공수

**1 ~ 2일.** P0에서 추론이 이미 검증됐다면, 대부분은 safe 래퍼 설계와 OpenAI 스키마/SSE 포맷 맞추기.

---

## 8. 다음 단계로 넘기는 것

- P1이 끝나면 "모델 1개 + 생성 루프"가 있다. P2는 이 생성 루프를 **여러 요청이 동시에** 쓰도록 스케줄러/슬롯 풀로 재구성한다.
- P1의 `generate`는 **토큰 1개 단위**여야 P2가 수월하다. 한 요청을 끝까지 도는 루프를 통째로 스케줄러에 넣지 말 것.
- jinja chat template, prefix cache, 동시 요청은 P1 범위가 아니다.
