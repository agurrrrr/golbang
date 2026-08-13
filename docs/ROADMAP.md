# golbang 로드맵

> 각 단계(P0~P4)는 **독립적으로 검증 가능**하다. 앞 단계가 끝나야 다음 단계로 진행.

## P0 — 스캐폴드 & FFI 연결 (가장 리스크 큰 구간, 먼저 검증)

**목표:** gfx906에서 Rust가 `llama.h`를 통해 모델을 로드하고 토큰 1개를 디코딩한다.

- [ ] `golbang-sys/build.rs`: bindgen으로 `llama.h` → Rust FFI 생성
- [ ] `build.rs`에서 cmake+hipcc로 llama.cpp(ggml-hip, gfx906) 빌드 or 기존 `.so` 링크
- [ ] 링크: `libllama`, `libggml-hip`, ROCm 런타임(`amdhip64`, `rocblas`, `hipblas`)
- [ ] 검증 테스트: GGUF 로드 → `llama_tokenize` → `llama_decode` 1회 → `llama_get_logits`
- [ ] **완료 기준:** `cargo test -p golbang-sys` 가 gfx906에서 실제 추론 1회 통과

## P1 — 단일 요청 E2E

**목표:** OpenAI 호환 `/v1/chat/completions` 1개 요청을 SSE 스트리밍으로 응답한다.

- [ ] `golbang-core`: 모델 래퍼(안전한 Rust API), 토크나이저, 샘플러(temperature/top-p)
- [ ] `golbang-server`: axum 라우터, 요청 검증, SSE 스트림
- [ ] **완료 기준:** `curl`로 채팅 1건이 토큰 단위 스트리밍으로 출력됨

## P2 — 동시성 코어 (핵심 차별화)

**목표:** llama.cpp보다 나은 동시성 구조를 실제로 구현한다.

- [ ] GPU 추론 전담 스레드(`spawn_blocking`) + tokio async HTTP 분리
- [ ] 슬롯 풀(slot pool) + iteration-level continuous batching 스케줄러
- [ ] 매 iteration마다 슬롯 join/evict, decode 중 신규 요청 삽입
- [ ] `CancellationToken` 기반 선제 취소 / 타임아웃
- [ ] bounded mpsc 백프레셔 (과부하 시 즉시 503)
- [ ] **완료 기준:** `n_parallel`개 동시 요청 시 개별 TTFT/지연이 단일 루프 대비 개선 측정

## P3 — 성능 & 안정화

- [ ] prefix caching (공통 프롬프트 KV 재사용)
- [ ] chunked prefill (긴 프롬프트가 decode를 굶기지 않게)
- [ ] 메트릭(토큰/s, TTFT, ITL, 슬롯 점유율) — `/metrics`
- [ ] vllm-gfx906 포크의 gfx906 attention tiling 패치 이식 검토
- [ ] **완료 기준:** llama.cpp `llama-server`와 동일 모델·동일 하드웨어 벤치 비교

## P4 — (장기·선택) 순수 Rust 커널

**목표:** hot-path 커널을 Rust+HIP로 점진 재작성해 FFI 의존을 줄인다.

- [ ] 대상: GEMM / attention / sampling hot path
- [ ] `hip-sys`/`rocm-rs` 성숙도 재평가 후 착수 여부 결정
- [ ] **완료 기준:** 특정 커널이 Rust 구현으로 치환되고 성능 회귀 없음

---

## 마일스톤 요약

| 단계 | 산출물 | 검증 |
|------|--------|------|
| P0 | FFI 바인딩 + gfx906 추론 1회 | `cargo test` 통과 |
| P1 | OpenAI 호환 단일 스트리밍 | curl E2E |
| P2 | continuous batching 동시성 | 동시 요청 지연 개선 측정 |
| P3 | 성능 최적화 | llama-server 대비 벤치 |
| P4 | Rust 커널 일부 | 회귀 없음 |
