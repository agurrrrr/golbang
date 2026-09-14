# q2k-panel-bench

Q2_K expert용 AVX2 패널 커널 후보의 마이크로벤치입니다. 이슈 #229(DSV41-OPT-A2)의 1단계 산출물이며, 결과 해석과 채택 판단은 `docs/bench/dsv41-q2k-panel-negative.md`에 있습니다.

## 하는 일

업스트림 iqp(IQ panel) 방식을 Q2_K로 확장한 후보 커널을 구현하고, generic `ggml_vec_dot_q2_K_q8_K` 경로와 비교합니다.

- 패널 구성: 8개 source 행을 super-block 단위 int8 패널로 디코드하고, 16값 sub-block scale을 미리 곱합니다(`q * sc`, Q2_K는 최대 45라 int8에 들어감).
- 정수 커널: `_mm256_maddubs_epi16` + `_mm256_madd_epi16`로 8행을 한 번에 누적하고, super-block마다 `d`·`dmin`만 fp32로 적용합니다.
- 정확성: 단일 super-block과 전체 K(8행)에서 generic 구현과의 최대 절대·상대오차를 확인합니다.
- 성능: `n_tok=1`(decode)과 `n_tok=64`(prefill)에서 generic 대비 배속을 출력합니다. `prebuilt`는 패널을 미리 만든 순수 커널 시간, `incl.build`는 디코드 비용까지 지불한 값입니다.

결과: 수치적으로 정확하지만 generic보다 느려 **비채택**입니다. 자세한 표는 위 문서를 보십시오.

## 빌드

```
./build.sh
```

llama.cpp-ds41 트리 위치는 `LLAMA_DS41_DIR`로, 빌드 디렉터리는 `BUILD`로 바꿀 수 있습니다(기본 `<tree>/build`). ggml CPU 라이브러리에 링크하므로 런타임 트리를 수정하지 않습니다.

## 실행

```
./q2k-panel-bench            # 정확성 + 성능
./q2k-panel-bench --no-acc   # 성능만
```

## 주의 사항

- **단일 스레드 측정**입니다. 두 커널의 효율 비율을 보는 것이 목적이며, ggml 백엔드의 32스레드 스케줄링은 포함하지 않습니다.
- **가짜 결과가 아님을 보장하는 장치**가 두 개 있습니다. (1) `ggml_cpu_init()`으로 fp16 변환 테이블을 초기화하지 않으면 generic 기준선이 0을 반환하므로 반드시 호출합니다. (2) 출력값을 `volatile` 싱크에 누적해 컴파일러가 커널 호출을 제거하지 못하게 합니다.
- min 항은 generic과 동일하게 `dmin * a.d * sum(m*bsums)`로 계산해야 합니다. `a.d`를 빠뜨리면 수치가 크게 어긋납니다.
