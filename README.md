# golbang

> **Rust로 만든, gfx906(AMD MI50)용 GGUF LLM 서빙 엔진.**
> llama.cpp보다 개선된 비동기 동시성 아키텍처를 목표로 한다.

## 목표 (요구사항)

| # | 요구 | 구현 |
|---|------|------|
| 1 | Rust | 서빙·스케줄링·배칭·토크나이즈·API 전부 Rust |
| 2 | GGUF 서빙 | llama.cpp `llama.h` C API로 GGUF 로드/추론 |
| 3 | llama.cpp보다 개선된 동시성 | tokio 비동기 분리 + iteration-level continuous batching |
| 4 | gfx906(MI50) 호환 | 이미 gfx906으로 빌드된 `ggml-hip` 커널 재사용 |
| 5 | 서빙+컴퓨트 모두 Rust | 단일 Rust 바이너리. GPU 연산만 C ABI FFI로 호출 |

## 왜 하이브리드인가

순수 Rust로 gfx906 GPU 커널을 작성하는 길은 **검증되지 않았다**.
rocm-rs에 `#[amdgpu_global]` 커널 매크로는 있으나 gfx906 코드젠·rocBLAS급 최적화 커널은 미검증이고,
candle/mistral.rs는 HIP 백엔드가 없으며 hipfire는 RDNA 전용이다.
따라서 **서빙·스케줄링·배칭은 100% Rust**로 작성하되,
GPU 수학 연산은 **이미 gfx906으로 빌드된 ggml-hip 커널을 Rust가 C ABI로 호출**한다(P0~P3).
순수 Rust 커널 경로는 P4에서 재평가한다.
최종 산출물은 파이썬/Go 런타임이 없는 **단일 Rust 바이너리**다.

상세 결정 근거는 위키 `architecture-decision` 참조.

## 구조

```
golbang-server   # 실행 바이너리: axum HTTP API, OpenAI 호환 엔드포인트
golbang-core     # 스케줄러, continuous batcher, 슬롯 풀, 샘플링, 스트리밍
golbang-sys      # llama.cpp / ggml-hip 저수준 FFI 바인딩 (unsafe, bindgen)
```

## "개선된 동시성"의 의미

llama-server도 이미 continuous batching(슬롯+통합 batch)이 기본 내장되어 있다.
실제 한계는 (a) 스케줄 루프가 `llama_decode`에 동기 결합되어 정책(join/evict/chunked prefill/우선순위)을 바꾸기 어렵고,
(b) 취소가 decode 경계까지 기다려야 하며, (c) 기능이 무겁다는 점이다.

golbang이 차별화되는 지점은:
- **정책 제어** — 스케줄 정책(join/evict/우선순위)을 Rust 코드로 직접 교체 가능
- **단순한 제어면** — 연결 끊김 → 슬롯 회수의 경로가 짧고 명시적
- **즉시 503** — bounded 큐로 과부하 시 대기 없이 거절
- **경량 표면** — 임베딩/rerank/스펙큘레이티브 없이 채팅 스트리밍에 집중

구현 수단은:
- **비동기 분리** — HTTP 수신 / 스케줄링 / GPU 추론을 별도 tokio 태스크로 격리
- **iteration 경계 join/evict** — `llama_decode` 반환 직후 빈 슬롯에 신규 요청 삽입
- **선제 취소** — `CancellationToken`으로 연결 끊김 즉시 슬롯 회수
- **백프레셔** — bounded 큐로 과부하 시 즉시 503

> GPU 처리량 우위는 P2 성공 조건이 아니다. llama-server 대비 동률 + 위 제어면 개선이면 통과.

## 빌드 (예정)

```bash
export PATH="$HOME/.cargo/bin:$PATH"
# llama.cpp ggml-hip (gfx906) 링크 필요 — golbang-sys/build.rs가 처리
cargo build --release
```

## 로드맵

`docs/ROADMAP.md` 참조.

## 작업 지시서

각 단계(P0~P4)의 상세 실행 지시서는 `docs/work-orders/` 참조.
방향: **하이브리드 확정** (Rust 오케스트레이션 + gfx906 빌드된 ggml-hip FFI).
