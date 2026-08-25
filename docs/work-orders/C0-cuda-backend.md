# CUDA(RTX 3060) 백엔드 지원 작업 계획

> **이슈:** 신규 · **선행:** 없음 · **후행:** C1~C4
> **목표:** golbang이 AMD MI50(gfx906/HIP)뿐 아니라 NVIDIA RTX 3060(CUDA)에서도
> 추론 서버로 동작하도록 만든다. 참고 서비스는 `qwen3.8-q2.service`
> (llama-server, RTX 3060 CUDA, port 8084, active running).
>
> **왜 필요한가:** 현재 golbang은 `golbang-sys/build.rs`가 SHA 고정 + `gfx906` 바이트
> 검사 + `libggml-hip.so` 링크로 **HIP 전용 하드코딩**되어 있다. CUDA 트리(`llama.cpp-cuda`,
> `libggml-cuda.so`)가 이미 빌드되어 있지만 golbang이 이를 쓰지 못한다.

---

## 1. 현재 상태 요약 (조사 결과)

| 항목 | 값 |
|------|----|
| golbang GPU 백엔드 | **HIP 전용** (gfx906 MI50) |
| golbang-sys build.rs | SHA `3ac5658c7` 고정 + `gfx906` 바이트 검사 + `libggml-hip.so` 링크 |
| HIP 트리 | `/home/agurrrrr/code/local-llm/llama.cpp-upgrade` @ `3ac5658c7` |
| CUDA 트리 | `/home/agurrrrr/code/local-llm/llama.cpp-cuda` @ `749f688f` |
| CUDA `.so` | `libggml-cuda.so`, `libllama.so` 등이 `build/bin`에 존재 |
| 참고 서비스 | `qwen3.8-q2.service` (llama-server, CUDA0, port 8084, `-ngl 99`) |
| RTX 3060 | 12 GiB VRAM, `nvidia-smi` 상 11649MiB 사용 중 (llama-server가 점유) |
| 모델 | `/home/agurrrrr/models/qwen3.8/Qwen3.8-27B-UD-IQ2_S.gguf` |

**핵심 쟁점 2가지:**
1. **빌드 타임 선택**: `golbang-sys/build.rs`가 하나의 트리/SHA/백엔드에 하드코딩.
   → `GOLBANG_GPU=cuda|hip` 환경변수로 트리·SHA·링크 라이브러리를 선택하도록 일반화.
2. **런타임 백엔드 로드**: `golbang-core/src/model.rs` `init_backend()`가
   `golbang_sys::LLAMA_BIN_DIR`(컴파일 타임 env)을 그대로 씀 → CUDA 빌드 시
   해당 `build/bin`을 가리키면 됨. 별도 런타임 선택 불필요(빌드 타임에 결정).

---

## 2. 단계별 작업 (C1~C4, 순차)

### C1 — golbang-sys CUDA 빌드 지원 (build.rs 일반화)
- `build.rs`를 `GOLBANG_GPU` 환경변수(`hip` 기본, `cuda` 선택)로 분기.
- CUDA 모드: 트리 `llama.cpp-cuda` @ `749f688f`, `libggml-cuda.so` 링크,
  `gfx906` 바이트 검사를 CUDA 심볼 검사로 대체.
- 링크 라이브러리: `hip` 모드 → `amdhip64/hipblas/rocblas`; `cuda` 모드 → `cudart/cublas`.
- **완료 기준:** `GOLBANG_GPU=cuda cargo build -p golbang-sys` 성공 + `cargo test -p golbang-sys` 통과.

### C2 — golbang-core/server 백엔드 로드 점검
- `init_backend()`가 CUDA 빌드에서 `llama.cpp-cuda/build/bin`을 로드하는지 확인.
- `libggml-cuda.so` 로드 시 CUDA 백엔드가 등록되는지 로그 확인.
- **완료 기준:** CUDA 빌드 서버가 부팅 시 `ggml-cuda` 백엔드를 인식하고 모델 로드.

### C3 — CUDA용 systemd 서비스 파일 작성
- `deploy/golbang-cuda-qwen38.service` 작성.
- `qwen3.8-q2.service` 파라미터를 참고하되 golbang 바이너리/플래그 사용:
  `--device` 대신 `--n-gpu-layers 99`, `CUDA_VISIBLE_DEVICES=0`, port 8084.
- `Conflicts`에 `qwen3.8-q2.service` 등 CUDA/llama 서비스 추가.
- **완료 기준:** 유닛 파일 작성 + `systemd-analyze verify` 통과.

### C4 — 빌드·스모크·서비스 기동
- CUDA 릴리즈 빌드, 스모크 요청(`/v1/chat/completions`) 검증.
- 서비스 등록(`systemctl daemon-reload` + `enable` + `start`) 후 저널 확인.
- **완료 기준:** CUDA 서비스가 active running, 스트리밍 응답 정상, 저널에 CUDA 로그.

---

## 3. 위험 / 주의

- **SHA 불일치**: CUDA 트리 HEAD `749f688f`는 HIP 트리 `3ac5658c7`와 다름.
  bindgen은 CUDA 트리의 `llama.h`를 써야 함. 헤더 줄 수가 다를 수 있음 → 하드코딩 검사 완화 필요.
- **VRAM**: RTX 3060 12 GiB. `qwen3.8-q2.service`가 `-ngl 99` + ctx 80k로 11649MiB 점유 중.
  golbang 기본값(`--n-ctx 70000 --n-batch 2048`)은 MI50 32 GiB 기준이라 3060에선 축소 필요.
- **동시 실행 금지**: `Conflicts`로 llama-server/CUDA 서비스와 겹치지 않게.
- **모델**: `Qwen3.8-27B-UD-IQ2_S.gguf` (qwen3.8-q2가 쓰는 모델) 사용 권장.

---

## 4. 완료 정의

- `GOLBANG_GPU=cuda`로 빌드한 golbang-server가 RTX 3060에서
  `/v1/chat/completions` 스트리밍 응답을 정상 수행.
- `golbang-cuda-qwen38.service`가 active running이며 저널에 CUDA/RTX 3060 로그.
- 기존 MI50(HIP) 서비스(`golbang-deepseek.service` 등)가 여전히 동작(회귀 없음).

## 5. 문서

- 위키 `golbang-cuda-backend`에 진행 상황 기록.