# golbang 로드맵

> 각 단계(P0~P6)는 **독립적으로 검증 가능**하다. 앞 단계가 끝나야 다음 단계로 진행.
> 상세 실행은 `docs/work-orders/`를 따른다. P4는 선택이며 P5·P6보다 뒤다.

## 고정 결정 (2026-08-13, 작업 #272 합의)

- P0 링크: **(A) 기존 `.so` + SHA 고정**. 드리프트 시 (B) 재빌드.
- P1 템플릿: **Qwen ChatML 하드코딩**. jinja는 후속. `llama_chat_apply_template`만으로는 Qwen을 처리하지 못한다.
- P2 성공: P1 대비 후발 TTFT 개선 + llama-server와 동일 조건 **기록**. 처리량 승리는 필수가 아니다.
- P4: 선택. P0~P3 완료 + gfx906 커널 매크로 스모크 + 프로파일이 특정 커널을 지목할 때만.

## P0 — FFI 연결 (가장 리스크 큰 구간, 먼저 검증)

**목표:** gfx906에서 Rust가 `llama.h`를 통해 모델을 로드하고 토큰 1개를 디코딩한다.
**목적:** 연결 검증. 속도 하한 없음.

- [x] 링크 대상 SHA 고정: `367ebbc20` / `llama.h` 1638줄 / `llama.cpp-glm5next` (2026-09-01, G2 이슈 #98. origin/master `f8dbcd618` + glm5next PR #27754 + DPP + MMQ I=64 + GCN repack). 이전 핀은 `3ac5658c7` / 1629줄 / `llama.cpp-upgrade` (롤백 경로로 유지), 그 이전은 `5b474eb69` / 1611줄.
- [x] `golbang-sys/build.rs`: 그 SHA의 `llama.h`를 bindgen
- [x] (A) 기존 `.so` 링크 (`libllama`, `libggml-hip`, ROCm 런타임)
- [x] 현행 API: `llama_model_load_from_file` / `llama_init_from_model` / `llama_model_free`
- [x] 검증: `GOLBANG_TEST_MODEL` + 프롬프트 `"Hello"` → `llama_decode` 1회 → argmax ∈ `[0, n_vocab)` + HIP/gfx906 로그
- [x] **완료 기준:** `cargo test -p golbang-sys` 통과 (속도 하한 없음)

## P1 — 단일 요청 E2E

**목표:** OpenAI 호환 `/v1/chat/completions` 1개 요청을 SSE 스트리밍으로 응답한다.

- [x] `golbang-core`: 모델 래퍼, 토크나이저, 샘플러. `generate`는 토큰 1개 단위
- [x] Qwen ChatML 하드코딩. 실패 시 raw prompt 폴백 + 경고
- [x] `golbang-server`: axum + SSE
- [x] **완료 기준:** `choices[].delta.content` 토큰 단위, EOS/`max_tokens` 시 `data: [DONE]`, 빈 messages는 4xx

## P2 — 동시성 코어: 스케줄 정책이 교체 가능한 배치 루프

**목표:** HTTP/스케줄이 decode에 막히지 않고, iteration 경계에서 join/evict/취소가 관측되며, 정책을 Rust에서 교체할 수 있다.

llama-server도 이미 CB가 기본이다. "CB 도입"이나 "decode 중 삽입"은 목표가 아니다.
join은 **이번 `llama_decode` 반환 직후** 빈 슬롯에 넣는 것이다.

- [x] GPU 추론 전담 (`spawn_blocking`) + tokio HTTP 분리
- [x] 슬롯 풀 + 정책 trait + iteration 루프
- [x] `CancellationToken` (회수 하한은 decode 1회)
- [x] bounded mpsc — decode 중에도 즉시 503
- [x] **완료 기준:** (a) P1 대비 후발 TTFT 개선 (b) llama-server와 동일 조건 기록. 처리량 승리 필수 아님

## P3 — 성능 & 안정화

- [x] **먼저** `llama_memory_seq_*` 복사 스파이크. 불가 → 슬롯 내 재사용으로 축소 (#26)
- [x] prefix caching (slot-local만. 생산 다턴 `cache_n=0`은 P5)
- [x] chunked prefill (정책에 예산으로 추가)
- [x] `/metrics`
- [ ] (선택) (B) cmake+hipcc 소스 빌드 — ADR의 재현성 빌드는 여기
- [x] **완료 기준:** 동일 GGUF / `-c` / `-np` / `-ctk` / `-ctv` / GPU 클럭으로 llama-server 벤치 기록
      (`docs/bench/p3.md`. 생산 실측은 위키 `dsv4-run-notes`)

## P5 — 다턴 prefix cache (커널 0줄)

**목표:** 같은 대화의 다음 턴에서 prefix KV가 슬롯에 남아 suffix만 prefill한다.

- [x] Stop/Length evict 후 `clear_seq` 하지 않음
- [x] 생성 토큰 ID를 슬롯 캐시에 누적
- [x] bind 시 LCP만큼 `n_past`, suffix KV만 `llama_memory_seq_rm` (DSV4는 prefill 체크포인트 + `n_rs_seq=1`)
- [x] **완료 기준:** 2턴째 `cache_n > 0`, 3k대 TTFT가 suffix만큼. 이슈 #28

## P6 — rocprof → 지목 커널만 HIP C++

**목표:** P3 §4.5를 실제로 한다. Rust 커널이 아니다.

- [x] gfx906 rocprof, prefill/decode 순위표 (`docs/bench/p6.md`)
- [x] 지목 안 됨 — 중단 사유만 (decode/suffix 1위는 CPU MoE `n_cpu_moe=32`. HIP 패치 0)
- [x] **완료 기준:** 순위표 + 중단 사유. 커널 diff 0. 이슈 #29

## P7 — Qwen3.8-27B decode ≥ llama-server (local-llm #364)

**목표:** 같은 MI50 · 같은 GGUF · 같은 spec 플래그 · SHA `3ac5658c7`에서
`golbang-qwen38` decode가 생산 llama-server(#364: 실사용 17–23 t/s, 짧은 25 t/s)를
맞추거나 넘긴다. DSV4 ~8 t/s 동률 A/B와 다른 문제.

- [ ] 고치기 전 통제 A/B (`docs/bench/p7.md`)
- [ ] 병목 분류: spec 미발화 / 검증 비용 / 호스트 오버헤드. 클럭·커널 추측 금지
- [ ] **완료 기준:** 16 / ~84 / 100–230 / 장문 밴드에서 golbang ≥ llama. 16토큰만으로 완료 금지. 이슈 #31

## P4 — (장기·선택) 순수 Rust 커널

**목표:** hot-path 커널을 Rust+HIP로 점진 재작성해 FFI 의존을 줄인다.

착수 전 gate (전부 필수):

- [x] P0~P3 완료
- [x] P6(#29) rocprof가 병목을 특정 커널로 지목 — **실패. 열지 않음** (`docs/bench/p6.md`)
- [ ] gfx906에서 rocm-rs 커널 매크로 스모크 (no-op / vector add)
- [ ] **완료 기준:** 특정 커널 Rust 치환 + 성능 회귀 없음

---

## 마일스톤 요약

| 단계 | 산출물 | 검증 |
|------|--------|------|
| P0 | FFI 바인딩 + gfx906 추론 1회 | `cargo test` 통과, 속도 하한 없음 |
| P1 | OpenAI 호환 단일 스트리밍 | curl SSE + 4xx |
| P2 | 동시성 코어 — 교체 가능한 스케줄 루프 | P1 대비 TTFT + llama-server 기록 |
| P3 | 성능 최적화 | 공정 조건 벤치 |
| P5 | 다턴 prefix KV 생존 | 2턴째 `cache_n>0`, TTFT=suffix |
| P6 | rocprof + HIP C++ 1커널 | 순위표, 지목 시에만 패치 |
| P7 | Qwen3.8 decode ≥ llama-server #364 | 밴드별 A/B, 16토큰만으로 완료 금지 |
| P4 | Rust 커널 일부 (선택) | P6 이후 gate, 회귀 없음 |
