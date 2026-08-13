# P0 — golbang-sys: llama.h Rust FFI 바인딩 + gfx906 추론 1회

> **이슈:** #23 · **선행:** 없음 · **후행:** P1
> **왜 먼저인가:** 전체 프로젝트에서 가장 리스크가 큰 구간이다.
> Rust → llama.cpp(ggml-hip) FFI 연결과 gfx906 실제 추론이 되는지를 **가장 먼저, 가장 작은 범위로** 검증한다.
> 여기서 막히면 하이브리드 방향 자체를 재검토해야 한다.
>
> **이 단계의 목적:** 연결 검증. **속도 하한은 두지 않는다.**

---

## 1. 목표

Rust 바이너리가 `llama.h` C API를 통해:
1. GGUF 모델을 로드하고
2. 프롬프트를 토크나이즈하고
3. gfx906 GPU에서 `llama_decode`를 1회 실행하고
4. logits을 읽어 다음 토큰 1개를 샘플링한다.

**완료 기준:** `GOLBANG_TEST_MODEL`로 지정한 소형 GGUF, 프롬프트 `"Hello"`,
`llama_decode` 1회 후 argmax 토큰이 `[0, n_vocab)` 이고, 로그에 HIP/gfx906이 보이며
CPU 폴백이 아니다. `cargo test -p golbang-sys`가 이를 통과한다. **속도 하한 없음.**

---

## 2. 배경 / 근거

- 이미 `/home/agurrrrr/code/local-llm/llama.cpp`에 `GGML_HIP=ON`, `CMAKE_HIP_ARCHITECTURES=gfx906`로 빌드된 산출물(`.so`)이 있다.
  - `libllama.so`, `libggml.so`, `libggml-base.so`, `libggml-cpu.so`, `libggml-hip.so` 등.
- `llama.h`는 C API를 노출하므로 Rust `bindgen`으로 FFI 바인딩 생성이 가능하다.
- 이 단계에서는 **안전한 래퍼를 만들지 않는다.** unsafe 저수준 FFI가 "연결되고 1회 도는지"만 본다. 래퍼는 P1에서 만든다.

---

## 3. 산출물

| 산출물 | 위치 | 설명 |
|--------|------|------|
| build 스크립트 | `golbang-sys/build.rs` | bindgen 생성 + 링크 설정 |
| FFI 바인딩 | `golbang-sys/src/lib.rs` (+ 생성된 bindings) | `llama.h` 심볼을 Rust로 노출 |
| 래퍼 최소골격 | `golbang-sys/src/` | 로드/토크나이즈/디코드 호출 시퀀스 |
| 검증 테스트 | `golbang-sys/tests/` or `src/lib.rs` 내 `#[cfg(test)]` | gfx906 추론 1회 |
| 크레이트 의존성 | `golbang-sys/Cargo.toml` | `bindgen` (build), `libc` 등 |

---

## 4. 단계별 작업

### 4.0 링크 대상 검증 (첫 태스크, 다른 작업보다 먼저)

구현 착수 전에 아래 **고정 값**을 다시 확인하고, 값이 바뀌었으면 이 표를 고친 뒤 진행한다.

| 항목 | 고정 값 (2026-08-13 확인) |
|------|---------------------------|
| 트리 | `/home/agurrrrr/code/local-llm/llama.cpp` |
| `git rev-parse HEAD` | `5b474eb69dac2d7c26ba8855310d3b60e02a5c4f` (`5b474eb69`) |
| **커밋 시각** (git 기록 기준) | 2026-08-06 18:56:34 +0900 |
| **`libggml-hip.so.0` 빌드 시각** (.so mtime 기준) | 2026-08-06 19:01:11 +0900 |
| **`libllama.so.0` 빌드 시각** (.so mtime 기준) | 2026-08-06 19:01:35 +0900 |
| gfx906 코드 | `strings build/bin/libggml-hip.so \| grep gfx906` — 포함 확인됨 |
| bindgen 입력 | **이 SHA의** `include/llama.h` (1611줄). 라이브 최신 헤더 금지 |
| 현행 로드 API | `llama_model_load_from_file` / `llama_init_from_model` / `llama_model_free` |
| DEPRECATED | `llama_load_model_from_file` / `llama_new_context_with_model` / `llama_free_model` |

> **라벨 구분 (2026-08-13 리뷰 반영):** "커밋 시각"(git 기록)과 ".so 빌드 시각"(.so mtime)은 **서로 다른 값**이다.
> ADR/README에는 간결히 "빌드 2026-08-06 19:01"로 쓰되, P0 표에서는 위처럼 두 시각을 명시적으로 구분해 둔다.

같은 디렉터리의 다른 HEAD (섞지 말 것):

| 경로 | HEAD |
|------|------|
| `llama.cpp` | `5b474eb69` ← **이 트리만 사용** |
| `llama.cpp.new` | `e700bfb37` |
| `llama.cpp-furnace` | `5013b9f91` |
| `llama.cpp-prefetch` | `6e3d2ef73` |

- [ ] 구현 당일 `rev-parse HEAD`와 `.so` mtime이 위 표와 같은지 재확인.
- [ ] bindgen 입력을 **그 SHA의 `llama.h`**로 고정 (vendor 복사 또는 절대경로). 라이브 트리 헤더와 옛 `.so`를 섞지 말 것.
- [ ] **심볼/SHA가 바뀌면 (A)를 버리고 (B) 재빌드(P3)로 전환한다.** ADR의 submodule+cmake 서술은 (B)용이며, P0 기본 경로가 아니다.

### 4.1 링크 전략 결정 (택 1, 먼저 결정할 것)

- **(A) 기존 `.so` 재사용 (권장, 빠름):** 이미 gfx906으로 빌드된 `.so`를 `cargo:rustc-link-search`로 링크.
  - 장점: P0 리스크를 "FFI 연결"에만 집중. llama.cpp 재빌드 시간 제거.
  - `build.rs`에서 `$GOLBANG_LLAMA_DIR/build/bin`(또는 실제 `.so` 위치)를 탐색.
  - 전제: 4.0의 SHA 고정이 지켜질 때만.
- **(B) cmake+hipcc 재빌드:** `build.rs`가 llama.cpp를 직접 빌드.
  - 장점: 재현성. 단점: P0에서 변수가 많아져 디버깅이 어려움.
  - **P0에서는 A로 검증하고, B는 P3(안정화)에서 재현성 확보용으로 이관.**

> **결정 (2026-08-13 고정):** P0는 **(A) + SHA 고정**. 심볼/SHA 드리프트 발생 시 (B)로 전환.

### 4.2 build.rs 작성

- [ ] llama.cpp 소스/빌드 경로를 환경변수(`GOLBANG_LLAMA_DIR`)로 주입받되, 기본값은 `/home/agurrrrr/code/local-llm/llama.cpp`.
- [ ] `bindgen::Builder`로 **4.0에서 고정한 그 `llama.h`** 파싱 → `OUT_DIR/bindings.rs` 생성.
  - allowlist: `llama_*` 함수와 필요한 타입만 (불필요한 심볼 폭증 방지).
- [ ] 링크 지시 출력:
  - `cargo:rustc-link-search=native=<llama.cpp build lib 경로>`
  - `cargo:rustc-link-lib=dylib=llama`, `ggml`, `ggml-base`, `ggml-hip`
  - ROCm 런타임: `amdhip64`, 필요시 `rocblas`, `hipblas` (`/opt/rocm/lib` 검색 경로 추가)
- [ ] `cargo:rustc-link-arg`로 rpath 설정(실행 시 `LD_LIBRARY_PATH` 없이 .so 탐색 가능하게) — 선택.

### 4.3 FFI 바인딩 노출

- [ ] `golbang-sys/src/lib.rs`에서 `include!(concat!(env!("OUT_DIR"), "/bindings.rs"))`.
- [ ] 스켈레톤 `add()` 제거. 외부로 노출할 최소 심볼 정리:
  - 백엔드: `llama_backend_init`, `llama_backend_free`
  - 모델: `llama_model_default_params`, **`llama_model_load_from_file`**, **`llama_model_free`**
  - 컨텍스트: `llama_context_default_params`, **`llama_init_from_model`**, `llama_free`
  - 토크나이즈: `llama_tokenize`, `llama_vocab_*`
  - 추론: `llama_decode`, `llama_get_logits`, batch 유틸(`llama_batch_init` 등)

> ⚠️ **API 이름 고정 (2026-08-13 확인):** 로컬 `llama.h`(SHA `5b474eb69`, 1611줄) 기준
> `llama_load_model_from_file` / `llama_new_context_with_model` / `llama_free_model`은 **DEPRECATED**이고
> 현재 이름은 **`llama_model_load_from_file` / `llama_init_from_model` / `llama_model_free`** 이다.
> bindgen allowlist도 현행 이름으로 맞출 것. 추측으로 쓰지 않는다.

### 4.4 검증 테스트 작성

- [ ] 테스트용 소형 GGUF 경로를 환경변수(`GOLBANG_TEST_MODEL`)로 주입.
- [ ] 테스트 시퀀스:
  1. `llama_backend_init`
  2. 모델 로드 (GPU 오프로드 파라미터 설정 — `n_gpu_layers`)
  3. 컨텍스트 생성
  4. 짧은 프롬프트(예: "Hello") 토크나이즈
  5. batch 구성 → `llama_decode` 1회
  6. `llama_get_logits`에서 argmax로 다음 토큰 1개 확인
  7. 토큰이 유효 범위 내인지 assert
  8. 자원 해제
- [ ] **실행 확인:** `cargo test -p golbang-sys -- --nocapture`가 gfx906에서 panic/segfault 없이 통과.
- [ ] **CPU 폴백 검증 (구체):**
  - `llama_backend_init` 로그에서 `backend: HIP`(및 장치명에 gfx906/0x66a1)가 보이는지 확인.
  - `llama_model_load_from_file` 호출 후 **오프로드된 레이어 수**를 assert한다
    (`llama_model_n_layers` 대비 `n_gpu_layers` 설정이 실제로 반영됐는지).
  - `n_gpu_layers` 명시만으로 "CPU 폴백 아님"을 단정하지 않는다.

### 4.5 안전성 / 디버깅

- [ ] segfault 발생 시: 링크 누락(rocblas/hipblas), GPU 레이어 설정, batch 메모리 정렬 여부를 우선 의심.
- [ ] `LLAMA_LOG` 계열 콜백으로 llama.cpp 내부 로그를 Rust `tracing`으로 브릿지(선택, 디버깅에 유용).

---

## 5. 완료 기준 (Definition of Done)

연결 검증이 목적이다. **속도 하한·토큰 품질 기준은 없다.**

- [ ] `cargo build -p golbang-sys` 링크 에러 없음
- [ ] `GOLBANG_TEST_MODEL`로 소형 GGUF를 지정해 `cargo test -p golbang-sys -- --nocapture` 통과
- [ ] 테스트 프롬프트는 `"Hello"`
- [ ] `llama_decode` 1회 후 argmax 토큰이 `[0, n_vocab)`
- [ ] 로그에 HIP / gfx906이 보이고 CPU 폴백이 아님:
  - `llama_backend_init` 로그에서 `backend: HIP` 확인
  - 오프로드 레이어 수 assert (`n_gpu_layers` 설정이 실제 반영)
- [ ] (선택) `LLAMA_LOG` 브릿지로 내부 로그도 `tracing`에 남김
- [ ] 산출물 커밋
- [ ] 이 문서 4.0 표의 SHA·빌드 시각이 구현에 쓴 값과 일치

---

## 6. 리스크 & 완화

| 리스크 | 영향 | 완화 |
|--------|------|------|
| llama.cpp API 버전 불일치 | 함수명/시그니처 미스매치로 컴파일 실패 | 실제 헤더를 직접 읽고 bindgen allowlist를 그에 맞춤 |
| 링크 누락(rocblas/hipblas/amdhip64) | 런타임 `.so` 로드 실패 | `ldd libggml-hip.so`로 의존성을 미리 나열해 모두 링크 |
| CPU 폴백으로 조용히 실행 | "GPU에서 도는 줄" 착각 | `n_gpu_layers` 명시 + 로그로 HIP 백엔드 사용 확인 |
| FFI 경계 UB/segfault | 디버깅 난이도 ↑ | 최소 시퀀스만 먼저, 로그 브릿지, 필요 시 `LLAMA_DEBUG` |

---

## 7. 예상 공수

**0.5 ~ 1.5일.** (A) 기존 `.so` 재사용 + 헤더 API만 정확히 맞추면 빠르다.
대부분의 시간은 API 명칭 매칭과 링크 의존성 해결에 쓰일 것으로 예상.

---

## 8. 다음 단계로 넘기는 것

- P0 성공 시: "어떤 함수로 로드/디코드/샘플링이 도는지"가 확정되므로, P1에서 이를 **안전한 Rust 래퍼(`golbang-core`)**로 감싼다.
- P0 실패 시: 원인(FFI? 커널? GPU?)을 기록하고 하이브리드 방향 재검토.
