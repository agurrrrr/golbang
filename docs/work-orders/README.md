# golbang 작업 지시서 (Work Orders)

> **방향 확정:** 하이브리드 아키텍처 — 서빙·스케줄링·배칭은 100% Rust, GPU 연산은 gfx906 빌드된 `ggml-hip`을 C ABI FFI로 호출.
> 최종 산출물은 파이썬/Go 런타임이 없는 **단일 Rust 바이너리**.
>
> 이 디렉토리는 각 단계(P0~P4)를 **독립적으로 실행 가능한 작업 지시서**로 쪼갠 것이다.
> 상위 로드맵은 `docs/ROADMAP.md`, 아키텍처 결정 근거는 위키 `architecture-decision` 참조.

## 하이브리드가 동시성 개선에 유효한 이유 (요약)

llama-server의 병목은 GPU 커널이 아니라 **서빙 오케스트레이션**에 있다.
모든 활성 슬롯을 하나의 통합 배치로 묶어 `llama_decode()`를 동기 호출하기 때문에,
decode가 반환되기 전까지 새 요청·취소가 끼어들지 못하는 **head-of-line blocking**이 발생한다.

커널(ggml-hip)을 재사용하고 오케스트레이션만 Rust로 다시 작성해도,
동시성을 결정하는 부분(스케줄링·배칭·취소·백프레셔)은 전부 우리가 제어하므로
**동시성 개선 목표는 그대로 달성된다.** 즉 하이브리드여도 동시성 개선은 유효하다.

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
| **P0** | [P0-ffi-binding.md](P0-ffi-binding.md) | Rust가 `llama.h` FFI로 GGUF 로드 → gfx906 추론 1회 | `cargo test -p golbang-sys` 추론 1회 통과 | #17 |
| **P1** | [P1-single-request-e2e.md](P1-single-request-e2e.md) | OpenAI 호환 `/v1/chat/completions` 1건을 SSE 스트리밍으로 응답 | `curl` 채팅 1건 토큰 단위 스트리밍 출력 | #18 |
| **P2** | [P2-concurrency-core.md](P2-concurrency-core.md) | continuous batching 스케줄러 (핵심 차별화) | 동시 요청 TTFT/지연이 단일 루프 대비 개선 측정 | #19 |
| **P3** | [P3-performance.md](P3-performance.md) | prefix cache, chunked prefill, 메트릭 | llama-server와 동일 조건 벤치 비교 | #20 |
| **P4** | [P4-rust-kernels.md](P4-rust-kernels.md) | (장기·선택) hot-path 커널 Rust+HIP 점진 재작성 | 특정 커널 Rust 치환 + 성능 회귀 없음 | #21 |

## 진행 규칙

1. **순차 진행** — P0가 끝나야 P1, P1이 끝나야 P2. 앞 단계의 완료 기준을 만족하지 못하면 다음 단계로 넘어가지 않는다.
2. **각 단계는 독립 검증 가능** — 모든 단계에 "완료 기준"이 있고, 실제 명령으로 확인한다.
3. **가장 리스크 큰 것부터** — P0(FFI + gfx906 추론)가 전체의 성패를 가른다. 여기서 막히면 방향 재검토.
4. **커밋** — 각 단계 완료 시 해당 단계 산출물을 커밋한다.
