# DSV4.1 Q2_K AVX2 패널 커널: 마이크로벤치와 비채택 근거

기록: 2026-09-14. 이슈 #229(DSV41-OPT-A2), 상위 #224. 선행 결과 위키: `dsv41-opt-a-avx2-repack-negative`, `shi3z-a100-dsv41-techniques-for-golbang` 5절 A. 실험 도구: `tools/q2k-panel-bench/`. 원시 출력: `docs/bench/raw/dsv41-q2k-panel.txt`.

## 요약

이슈 #225에서 AVX2 Q2_Kx8 리팩 커널이 Zen2에서 generic 경로보다 느리다는 것을 확인했습니다. 이 이슈는 그 후속으로, 업스트림 iqp(IQ panel) 방식의 패널 커널을 Q2_K로 확장해 설계하고 판정했습니다. 후보 커널은 수치적으로 정확하지만(전체 K에서 최대 상대오차 1.9e-6), **속도는 generic보다 느립니다.** decode에 해당하는 n_tok=1에서 0.40배, prefill에 해당하는 n_tok=64에서 0.60배였습니다. 따라서 중단 기준에 따라 즉시 비채택으로 종료하고, 전체 모델 A/B는 수행하지 않았습니다.

## 1. 후보 설계

업스트림 iqp는 IQ2/IQ3/IQ1/IQ4_XS 같은 grid 기반 타입을 8행 단위 int8 패널로 디코드한 뒤 정수 GEMM을 돌립니다. Q2_K는 grid 타입이 아니지만 다음 두 가지 이유로 같은 패턴을 적용할 수 있습니다.

1. Q2_K의 2비트 q는 0~3이라 int8에 그대로 들어가고, 16값 sub-block scale sc는 0~15라 `q * sc`가 최대 45로 int8에 들어갑니다. 따라서 **scale을 미리 곱한 패널**을 만들 수 있습니다. 이러면 내부 정수 점곱에서 sub-block scale 곱셈(`madd`)과 scale 셔플이 사라집니다.
2. min 항(`-dmin * m`)은 q8_K의 `bsums`(16값 합)로 분리해 super-block당 한 번만 처리합니다.

패널 레이아웃은 iqp와 동일하게 `w[sb*128 + g*32 + row*4 + k] = q * sc`로 두고, 대응하는 활성 바이트는 `a->qs[sb*16 + g*4 + k]`입니다. 커널은 `_mm256_maddubs_epi16(w, y)` + `_mm256_madd_epi16(ones, ·)`로 8행을 한 번에 누적하고, super-block마다 `dfac * a.d`와 `dmin * a.d`만 fp32로 곱합니다.

기준선은 실제 mul_mat의 nrc==1 경로와 동일한 호출 방식으로, `ggml_vec_dot_q2_K_q8_K`를 (행, 열) 쌍마다 호출했습니다.

## 2. 실측 조건

| 항목 | 값 |
|--|--|
| 트리 | `llama.cpp-ds41` @ `24032ea2b` |
| 호스트 | AMD EPYC 7452 (Zen2, AVX2, AVX-512 없음), 64스레드 |
| 형태 | K=5120, M=2304, n_as=64, n_ids=6 |
| 스레드 | 1 (커널 자체의 효율 비교) |
| 데이터 | Q2_K 블록·q8_K 블록을 난수로 생성, 시행 3회의 최솟값 |
| 패널 빌드 | 1021 ms / 64 expert, 2.77 us / (패널, super-block) |

## 3. 수치 정확성

패널 커널의 결과를 generic `ggml_vec_dot_q2_K_q8_K`와 비교했습니다. 빌더의 (q, sc, m) 매핑은 참조 dequant와 전 위치에서 일치했습니다.

| 검사 | 최대 절대오차 | 최대 상대오차 | 판정 |
|--|--|--|--|
| 단일 super-block (8행) | 3.815e-06 | 4.799e-06 | PASS |
| 전체 K (8행, nb=20) | 3.624e-05 | 1.944e-06 | PASS |

커널은 정확합니다. 문제는 속도입니다.

## 4. 성능

`panel (prebuilt)`는 패널을 미리 만들어 둔 순수 커널 시간이고, `panel (incl.build)`는 디코드 비용까지 매번 지불한 값입니다. generic 대비 배속은 클수록 빠릅니다.

### 4.1 n_tok=1 (decode, expert 6개 × 1열)

| 경로 | ms | GFLOP/s | 배속 |
|--|--:|--:|--:|
| generic | 2.498 | 56.7 | 1.00x |
| panel (prebuilt) | 6.262 | 22.6 | 0.40x |
| panel (incl.build) | 66.781 | 2.1 | 0.04x |

### 4.2 n_tok=64 (prefill, expert 64개 × 6열)

| 경로 | ms | GFLOP/s | 배속 |
|--|--:|--:|--:|
| generic | 147.643 | 61.4 | 1.00x |
| panel (prebuilt) | 246.672 | 36.7 | 0.60x |
| panel (incl.build) | 881.221 | 10.3 | 0.17x |

패널은 scale 미리 곱셈으로 generic의 scale 셔플·곱셈을 제거했는데도 느립니다. 원인은 Zen2 AVX2에서 int8 dot이 이미 generic 커널에 잘 맞아 있고, 패널의 8행 인터리브가 활성 브로드캐스트와 `madd(ones)`를 추가하며, 디코드 오버헤드가 배치 이득을 상쇄하기 때문으로 보입니다. 이는 #225의 리팩 결과(디코드 0.72~0.84배, 프리필 0.48~0.82배)와 같은 방향입니다.

## 5. 두 번째 방향(pre-repacked GGUF) 판단

위 `panel (prebuilt)`는 패널이 이미 RAM에 있는 상태를 측정하므로, RAM 상주 문제를 완전히 배제하고 레이아웃 자체의 속도만 봅니다. 그 결과가 0.40~0.60배입니다. 따라서 리팩 레이아웃을 파일로 미리 구워 mmap으로 읽는 "pre-repacked GGUF"는 다음과 같습니다.

- RAM 상주 문제는 풀립니다.
- 그러나 느린 레이아웃을 그대로 쓰므로 속도 문제는 해결되지 않습니다.
- 리팩 레이아웃은 원본보다 커져 NAS 스트리밍을 악화시킵니다.

따라서 후순위였던 이 방향도 채택 근거가 없습니다.

## 6. Q4_K

Q4_K는 q가 0~15, sc가 0~63이어서 `q * sc`가 최대 945로 int8에 들어가지 않습니다. Q2_K처럼 scale을 미리 곱하는 패널을 만들 수 없어 전망이 더 나쁩니다. #225에서 Q4_K 리팩도 0.71~0.91배로 느렸으므로, Q4_K 패널은 별도로 프로토타입하지 않고 비채택으로 둡니다.

## 7. 결론과 권고

1. Q2_K AVX2 패널 커널은 정확하지만 generic보다 느리므로 **채택하지 않습니다.** n_tok=1에서 0.40배, n_tok=64에서 0.60배였습니다.
2. 중단 기준에 따라 전체 모델 A/B는 수행하지 않았습니다.
3. pre-repacked GGUF와 Q4_K 패널도 같은 이유로 채택하지 않습니다.
4. Q2_K expert의 CPU GEMV는 generic `ggml_vec_dot_q2_K_q8_K` 경로가 이 호스트에서 이미 최선입니다. CPU expert 시간을 줄이려면 커널 재배치가 아니라 (a) CPU expert를 GPU로 옮기는 VRAM 증설, (b) expert 수를 줄이는 소형 모델, (c) batch를 키우는 prefill 최적화처럼 다른 축을 봐야 합니다.

## 8. 산출물

- `tools/q2k-panel-bench/q2k-panel-bench.cpp`, `build.sh`: 패널 후보 커널과 마이크로벤치.
- `docs/bench/raw/dsv41-q2k-panel.txt`: 원시 출력.
- 이 문서: 정확성·성능 표와 채택 판단.
