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

순수 Rust로 gfx906 GPU 커널을 작성하는 것은 현재 불가능에 가깝다
(candle/mistral.rs는 HIP 백엔드 없음, hipfire는 RDNA 전용).
따라서 **서빙·스케줄링·배칭은 100% Rust**로 작성하되,
GPU 수학 연산은 **이미 gfx906으로 빌드된 ggml-hip 커널을 Rust가 C ABI로 호출**한다.
최종 산출물은 파이썬/Go 런타임이 없는 **단일 Rust 바이너리**다.

상세 결정 근거는 위키 `architecture-decision` 참조.

## 구조

```
golbang-server   # 실행 바이너리: axum HTTP API, OpenAI 호환 엔드포인트
golbang-core     # 스케줄러, continuous batcher, 슬롯 풀, 샘플링, 스트리밍
golbang-sys      # llama.cpp / ggml-hip 저수준 FFI 바인딩 (unsafe, bindgen)
```

## "개선된 동시성"의 의미

llama-server는 모든 활성 슬롯을 하나의 통합 배치로 묶어 `llama_decode()`를
동기 호출한다. decode가 반환될 때까지 새 요청·취소가 배치에 끼어들지 못해
head-of-line blocking이 생긴다.

golbang은:
- **비동기 분리** — HTTP 수신 / 스케줄링 / GPU 추론을 별도 tokio 태스크로 격리
- **iteration-level continuous batching** — 매 iteration마다 슬롯 join/evict
- **선제 취소** — `CancellationToken`으로 연결 끊김 즉시 슬롯 회수
- **백프레셔** — bounded 큐로 과부하 시 즉시 503

## 빌드 (예정)

```bash
export PATH="$HOME/.cargo/bin:$PATH"
# llama.cpp ggml-hip (gfx906) 링크 필요 — golbang-sys/build.rs가 처리
cargo build --release
```

## 로드맵

`docs/ROADMAP.md` 참조.
