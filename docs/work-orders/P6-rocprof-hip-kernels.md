# P6 — rocprof로 지목된 커널만 HIP C++로 고치기

> **이슈:** #29 · **선행:** P5 (#28, 권장) · **후행:** P4 (#27, 선택)
> **성격:** P3 §4.5가 선택으로 빠져 한 번도 안 한 일. **Rust GPU 커널이 아니다.**
> 프로파일이 이름을 붙인 커널 하나만 HIP C++로 고친다.

---

## 1. 목표

1. gfx906(MI50)에서 golbang 추론을 `rocprof`로 찍어 커널 순위를 남긴다.
2. 그 순위가 **특정 GPU 커널**을 가리킬 때만, 그 커널 하나를 HIP C++로 고친다.
3. 정확도 일치 + 성능 회귀 없음을 측정한다.

**완료 기준:** `docs/bench/p6.md`에 커널 순위표가 있다.
지목되면 패치 1개 + 전후 벤치. 지목되지 않으면 중단 사유만 남기고 커널 diff는 0.

---

## 2. 배경 / 근거

P3 지시서 §4.5 (선택):

- `vllm-gfx906` attention tiling 이식 검토
- `GGML_HIP_MMQ_MFMA` 재확인

P3 DoD에 필수가 아니라 #26은 메특·축소 prefix·벤치 골격만 하고 닫혔다.
rocprof 트레이스는 저장소·위키 어디에도 없다.

작업 #300 / ADR:

- golbang과 llama.cpp는 **같은** `libggml-hip.so` (SHA `5b474eb69`).
- 3k prefill ~77 tok/s, decode ~7.8 tok/s, GPU 99–100%.
- 언어만 Rust로 바꾸면 ISA가 같다. P4 DoD도 "더 빠름"이 아니라 회귀 없음.
- 지금 체감 40초는 커널이 아니라 `cache_n=0` (P5).
- 그다음이 **rocprof → 지목 커널만 HIP C++**.

이 이슈는 P4가 아니다. P4(#27)는 `rocm-rs` `#[amdgpu_global]` 치환이고
gate에 "프로파일이 특정 커널을 지목"이 있다. **그 프로파일이 여기 산출물이다.**

---

## 3. 산출물

| 산출물 | 위치 | 설명 |
|--------|------|------|
| rocprof 순위 | `docs/bench/p6.md` | prefill / decode 구간별 커널 시간 |
| (조건부) HIP 패치 | 핀 llama.cpp 또는 문서화된 overlay | 지목된 커널 1개 |
| 재빌드 `.so` | `libggml-hip.so` | gfx906, 기존 플래그 유지 |
| 전후 벤치 | `docs/bench/p6.md` | 동일 GGUF/`-c`/`-np`/클럭 |

---

## 4. 단계별 작업

### 4.0 선행 확인

- [ ] P5(#28)가 생산에서 `cache_n > 0` 인지 확인. 아니면 트레이스가
      3k 전량 prefill GEMM에 먹혀 순위가 왜곡된다.
- [ ] P5를 못 기다리면, 트레이스에 "전량 prefill / prefix hit"를 **따로** 찍고
      표에 구분한다. 기본은 P5 이후.

### 4.1 프로파일

- [ ] 대상: `golbang-deepseek`와 같은 모델·옵션
      (DSV4 IQ2_M, `--n-gpu-layers 99 --n-cpu-moe 32 --flash-attn on`,
      `--n-ctx 60000 --n-batch 5800 --n-ubatch 1024`).
- [ ] GPU 클럭 고정 (`set_gpu_clocks.sh`).
- [ ] `rocprof` / `rocprofv3`로 커널 디스패치 시간. 호스트 구간과 섞지 않는다.
- [ ] 최소 두 구간:
      1. prefill (긴 프롬프트, prefix hit 후 suffix)
      2. decode (생성 ≥50토큰)
- [ ] 상위 커널 이름·호출 수·총 시간·비율을 `docs/bench/p6.md`에 표로 남긴다.

### 4.2 지목 또는 중단

아래를 **하나라도** 만족하면 커널 패치를 하지 않고 기록만 하고 끝낸다.

- 1위가 GPU 커널이 아니다 (스케줄러, 토크나이즈, 샘플링).
- 1위가 CPU MoE / PCIe (`n_cpu_moe=32` expert 이동).
- 상위 커널 사이 차이가 측정 오차 수준이고, 하나 고쳐도 tok/s가 안 움직일 것으로 본다.

지목 조건: 한 커널(또는 명확한 한 패밀리, 예: 특정 attention kernel)이
해당 구간 GPU 시간의 큰 몫을 차지하고, 알고리즘/타일 개선 후보가 있다.

후보 (프로파일이 이름을 붙일 때만):

- P3 §4.5 `vllm-gfx906` attention tiling
- `GGML_HIP_MMQ_MFMA` 경로가 실제로 도는지 재확인
- flash-attn / DSV4 HC / Lightning Indexer / GDN 중 프로파일이 찍은 것

### 4.3 HIP C++ 패치 (지목된 경우만)

- [ ] 핀 트리 `/home/agurrrrr/code/local-llm/llama.cpp` @ `5b474eb69`
      또는 그 SHA에서 분기한 overlay. `llama.cpp.new` 등 다른 HEAD 금지.
- [ ] **HIP C++만.** Rust `#[amdgpu_global]` / `golbang-kernel` 신설 금지.
- [ ] 커널 **하나**. 전면 재작성 금지.
- [ ] `GGML_HIP=ON`, `CMAKE_HIP_ARCHITECTURES=gfx906`,
      `GGML_HIP_MMQ_MFMA=ON`으로 `libggml-hip.so` 재빌드.
- [ ] golbang은 그 `.so`를 링크 (P0 (A) 경로). SHA 드리프트 시 위키에 기록.

### 4.4 회귀

- [ ] `temperature=0` 동일 프롬프트: 토큰 시퀀스 일치 (또는 허용 오차를 문서화).
- [ ] 동일 조건 전후: prefill tok/s, decode tok/s, VRAM.
- [ ] 회귀면 패치 되돌리고 채택하지 않음. 측정은 남긴다.

---

## 5. 완료 기준 (Definition of Done)

- [ ] `docs/bench/p6.md`에 prefill/decode 커널 순위
- [ ] 지목됨 → HIP 패치 1개 + 정확도 + 회귀 없는 전후 벤치
- [ ] 지목 안 됨 → 중단 사유가 같은 파일에 있고 커널 diff 0
- [ ] Rust GPU 커널 0줄
- [ ] P4(#27) gate 문장("프로파일이 특정 커널을 지목")을 이 결과로 갱신
- [ ] 산출물 커밋

---

## 6. 리스크 & 완화

| 리스크 | 영향 | 완화 |
|--------|------|------|
| prefix miss 트레이스 | GEMM이 전부 1위 | P5 이후, 또는 구간 분리 |
| gfx906 rocprof 심볼 부실 | 커널 이름 없음 | 디스패치 그리드/모듈로라도 구분. 이름 없으면 추측 패치 금지 |
| 다른 llama.cpp HEAD에 패치 | 핀 `.so`와 불일치 | SHA `5b474eb69`만 |
| P4로 미끄러짐 | 수개월 Rust 커널 | 이 지시서 제목 그대로 HIP C++. P4 이슈를 열지 않음 |
| MoE CPU가 진짜 병목 | 커널 패치 무효 | §4.2 중단 조건 |

---

## 7. 예상 공수

- 프로파일 + 기록: **0.5일**
- 지목 후 HIP 패치 1개: **1 ~ 5일** (커널에 따라)
- 중단이면 프로파일만.

---

## 8. P4와의 관계

P4는 이 이슈가 끝난 뒤에만 다시 본다.

- 여기 순위가 커널을 안 가리키면 P4 gate 실패. 열지 않는다.
- 가리키고 HIP로 고쳤으면, Rust 재작성은 **소유권** 이유일 때만. 속도 이유가 아님.
- gfx906 `#[amdgpu_global]` 스모크는 여전히 P4 몫이다.
