# moe-telemetry

DSV4.1(deepseek41) MoE 라우팅 텔레메트리 도구입니다. 이슈 #227(DSV41-OPT-C)의 1단계 산출물이며, 결과 해석과 배치 설계는 `docs/bench/dsv41-moe-hot-experts.md`에 있습니다.

## 하는 일

llama.cpp 컨텍스트 파라미터 `cb_eval`을 백엔드 스케줄러에 연결하여, 각 층의 `ffn_moe_topk-<layer>` 텐서(선택된 expert id, I32 `[n_expert_used, n_tokens]`)를 노드 실행 직후 읽습니다. 이를 층별·expert별로 집계해 다음을 출력합니다.

- `*-summary.txt`: 층별 상위 8개 expert와 상위 k개 커버리지, 전 층 평균 커버리지.
- `*-counts.tsv`: `layer<TAB>expert<TAB>count_prefill<TAB>count_decode`.

prefill과 decode를 별도 phase로 집계합니다. 런타임 트리는 수정하지 않습니다.

## 빌드

```
./build.sh cuda     # V100 sm_70 (build-cuda/)
./build.sh hip      # MI50 gfx906 (build/)
```

llama.cpp-ds41 트리 위치는 `LLAMA_DS41_DIR`로 바꿀 수 있습니다(기본 `/home/agurrrrr/code/local-llm/llama.cpp-ds41`).

## 실행

CUDA 예시입니다.

```
GOLBANG_MOE_TELEMETRY_OUT=/path/out \
CUDA_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=<tree>/build-cuda/bin:/opt/cuda-12.8/lib64 \
./moe-telemetry \
  -m /home/agurrrrr/models/DeepSeek-V4.1-Flash-Q2_K-ds41/DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf \
  -ngl 99 --n-cpu-moe 39 --tensor-split 16,16 \
  -c 16384 -b 4096 -ub 1024 -t 32 \
  --temp 0.8 -f prompt.txt -n 256
```

환경 변수는 다음과 같습니다.

- `GOLBANG_MOE_TELEMETRY_OUT`: 출력 파일 접두사(기본 `moe-telemetry`).
- `GOLBANG_MOE_TELEMETRY_MAX_TOKENS`: prefill 토큰 상한(기본 2048).
- `GOLBANG_MOE_TELEMETRY_DEBUG`: 설정하면 처음 6개 `ffn_moe_topk` 텐서의 `ne`/`nb`와 값 일부를 `stderr`에 출력합니다.

## 주의 사항

- **strided 뷰**: `selected_experts`는 전체 argsort `[384, n_tokens]`의 뷰입니다. 행 보폭은 `n_used*4`가 아니라 `nb[1]`(=1536바이트)입니다. 도구는 `nb[1]`을 사용하므로 multi-token prefill도 올바르게 집계합니다. 이 부분을 선형 복사로 처리하면 prefill에서 각 토큰의 argsort 384개를 통째로 읽어 가짜 균등 분포가 나옵니다.
- **속도 측정용이 아님**: 콜백이 층마다·스텝마다 텐서를 읽으므로 백엔드 동기화가 끼어들어 decode가 느려집니다. 라우팅 통계만 보십시오. 속도는 `llama-bench`나 서비스 유닛으로 측정합니다.
- `-n`은 생성 토큰 수, `--temp` 등 표준 sampling 인자를 그대로 받습니다.
