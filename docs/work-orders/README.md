# golbang 작업 지시서 (Work Orders)

> **방향 확정:** 하이브리드 아키텍처 — 서빙·스케줄링·배칭은 100% Rust, GPU 연산은 gfx906 빌드된 `ggml-hip`을 C ABI FFI로 호출.
> 최종 산출물은 파이썬/Go 런타임이 없는 **단일 Rust 바이너리**.
>
> 이 디렉토리는 각 단계(P0~P4)를 **독립적으로 실행 가능한 작업 지시서**로 쪼갠 것이다.
> 상위 로드맵은 `docs/ROADMAP.md`, 아키텍처 결정 근거는 위키 `architecture-decision` 참조.

## 구현 전 고정 결정 (2026-08-13)

작업 #272 검토 합의. 구현 중에 이 네 줄을 뒤집지 않는다.

1. **P0 링크:** (A) 기존 `.so` + SHA `5b474eb69` 고정. 드리프트 시 (B).
2. **P1 템플릿:** Qwen ChatML 하드코딩. 실패 시 raw 폴백 + 경고. jinja는 후속.
3. **P2 성공:** P1 대비 후발 TTFT 개선 + llama-server와 동일 조건 기록. 처리량 승리는 필수 아님.
4. **P4:** 선택. P0~P3 완료 + gfx906 커널 매크로 스모크 + 프로파일이 특정 커널을 지목할 때만.

## 하이브리드의 역할 (요약)

llama-server도 이미 continuous batching(슬롯 + 통합 batch + `llama_decode`)이 **기본 내장**되어 있다.
따라서 golbang의 목표는 "CB 자체를 도입하는 것"이 아니다.
실제 한계는 (a) 스케줄 루프가 `llama_decode`에 동기 결합되어 **정책(join/evict/chunked prefill/우선순위)을 바꾸기 어렵고**,
(b) 취소가 decode 경계까지 기다리며, (c) 기능이 무겁다는 점이다.

커널(ggml-hip)을 재사용하고 오케스트레이션만 Rust로 다시 작성하면
동시성을 결정하는 부분(스케줄 정책·취소·백프레셔)을 전부 우리가 제어할 수 있다.
**GPU 처리량 우위는 목표가 아니고, 정책 교체 가능 + 제어면 단순화 + 경량 표면이 목표다.**

## 환경 (검증 완료)

| 항목 | 상태 |
|---|---|
| GPU | gfx906 (AMD MI50), ROCm 7.2 |
| Rust | 1.97.1 (rustup, `~/.cargo/bin` PATH 설정 필요) |
| HIP 커널 | llama.cpp `ggml-hip`이 gfx906으로 빌드됨 (`GGML_HIP=ON`, `CMAKE_HIP_ARCHITECTURES=gfx906`) |
| FFI 진입점 | `llama.h` C API 존재 (`llama_decode`, `llama_tokenize` 등) |
| 기존 자산 | `/home/agurrrrr/code/local-llm/llama.cpp` (빌드 산출물 `.so` 존재) |
| GGUF 모델 | `/home/agurrrrr/code/local-llm/models` (Qwen 계열 GGUF) |

## 단계별 지시서

| 단계 | 파일 | 목표 | 완료 기준 | 이슈 |
|------|------|------|-----------|------|
| **P0** | [P0-ffi-binding.md](P0-ffi-binding.md) | Rust가 `llama.h` FFI로 GGUF 로드 → gfx906 추론 1회 | `cargo test -p golbang-sys` 추론 1회 통과 (속도 하한 없음, 연결 검증) | #23 |
| **P1** | [P1-single-request-e2e.md](P1-single-request-e2e.md) | OpenAI 호환 `/v1/chat/completions` 1건을 SSE 스트리밍으로 응답 | `choices[].delta.content` 토큰 단위 + `[DONE]` + 빈 messages 4xx | #24 |
| **P2** | [P2-concurrency-core.md](P2-concurrency-core.md) | 정책 교체 가능한 스케줄 루프 + 제어면 | P1 대비 후발 TTFT 개선 + llama-server 기록 (처리량 승리 필수 아님) | #25 |
| **P3** | [P3-performance.md](P3-performance.md) | prefix cache, chunked prefill, 메트릭 | 동일 GGUF/`-c`/`-np`/`-ctk`/`-ctv`/클럭 벤치 | #26 |
| **P4** | [P4-rust-kernels.md](P4-rust-kernels.md) | (장기·선택) hot-path 커널 Rust+HIP 점진 재작성 | 특정 커널 Rust 치환 + 성능 회귀 없음 | #27 |

## 진행 규칙

1. **순차 진행** — P0가 끝나야 P1, P1이 끝나야 P2. 앞 단계의 완료 기준을 만족하지 못하면 다음 단계로 넘어가지 않는다.
2. **각 단계는 독립 검증 가능** — 모든 단계에 "완료 기준"이 있고, 실제 명령으로 확인한다.
3. **가장 리스크 큰 것부터** — P0(FFI + gfx906 추론)가 전체의 성패를 가른다. 여기서 막히면 방향 재검토.
4. **커밋** — 각 단계 완료 시 해당 단계 산출물을 커밋한다.
