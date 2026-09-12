# 1Cat-vLLM 1.5.0 on 2×V100-16GB (QUASAR NVFP4) — target-only 실측

기록: 2026-09-12. 작업 #9396 후속. `qwen38-vllm-vs-golbang`의 "미확인" 항목(2×V100에서 1Cat-vLLM 실제 기동)을 채운 실측입니다.

## 결론

- 1Cat-vLLM 1.5.0 wheel은 2×V100-16GB에서 **타깃 단독(no spec)으로 기동·서빙된다.** QUASAR NVFP4 가중치 19.15 GiB가 TP2로 카드당 10.3 GiB에 들어간다.
- 디코드는 카드당 조건이 다른 4×V100 README 표보다 낮지만, 골방 CUDA(2×V100, MTP)와 비교하면 타깃 단독으로도 소폭 앞선다.
- **DFlash2 추측 디코딩은 2×16GB에서 들어가지 않는다.** 드래프터(`incoai/Qwen3.8-27B-DFlash2`, 3.85 GiB)가 TP2로 분할되지 않고 카드마다 복제되어, KV 캐시로 남는 메모리가 최소 요구량보다 약 1 GiB 부족하다. 1Cat의 지원 매트릭스(DFlash2 = 4×V100, 2×V100은 32GB 프로파일)와 일치한다.

## 환경

| 항목 | 값 |
|---|---|
| 런타임 | 1Cat-vLLM 1.5.0 wheel (`1cat_vllm-1.5.0-cp312...whl`) |
| Python | 3.12.14 (uv) / venv `~/venvs/1cat-vllm` |
| torch | 2.10.0+cu128 |
| CUDA | 12.8 (`/opt/cuda-12.8`), 드라이버 580.178.04 |
| GPU | Tesla V100-SXM2-16GB ×2 (TP2, NVLink 없음) |
| 타깃 | `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` (19.15 GiB, compressed-tensors W4A4 → SM70 TurboMind W4A16 dense) |
| KV | `fp8_e5m2` |

### 기동 전제 (중요)

1. tilelang GDN 커널이 nvcc를 기본 gcc로 호출한다. 시스템 기본 gcc가 16이라 `unsupported GNU version`으로 죽는다. **`NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-14"` + `CC/CXX/CUDAHOSTCXX=gcc-14`** 로 해결.
2. `--max-num-seqs`를 낮추지 않으면 `max_num_seqs (256) exceeds available Mamba cache blocks` 오류. 단일 요청 벤치이므로 `--max-num-seqs 1`.
3. vLLM은 cumem 충돌 때문에 `expandable_segments:True`를 스스로 끈다. `PYTORCH_CUDA_ALLOC_CONF` 조정은 효과 없음.

## 타깃 단독 서빙

```bash
CUDA_VISIBLE_DEVICES=0,1 CUDA_HOME=/opt/cuda-12.8 \
CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 CUDAHOSTCXX=/usr/bin/g++-14 \
NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-14" TORCH_CUDA_ARCH_LIST=7.0 \
vllm serve /home/agurrrrr/models/qwen3.8/Qwen3.8-27B-QUASAR-NVFP4 \
  --served-model-name qwen3.8-27b-nvfp4 --trust-remote-code \
  --tensor-parallel-size 2 --attention-backend FLASH_ATTN_V100 \
  --kv-cache-dtype fp8_e5m2 --max-model-len 32768 --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 --host 0.0.0.0 --port 8000
```

- 가중치 로드: 워커당 10.3 GiB, weights 6.05 s
- 엔진 초기화: 105 s (컴파일 70 s)

## 측정 (`vllm bench serve`, openai backend, concurrency 1, `--ignore-eos`)

| 워크로드 | 결과 |
|---|---:|
| 512 in / 256 out, 5 reqs | 출력 46.14 tok/s, TPOT 20.49 ms, TTFT 323 ms |
| 8192 in / 128 out, 3 reqs | TPOT 20.67 ms (≈48.4 tok/s 정상상태) |
| 16384 in / 4 out, 3 reqs | TTFT 6468 ms → 프리필 ≈2,533 tok/s |

디코드는 컨텍스트 512→8192에서 TPOT가 20.49→20.67 ms로 거의 평탄하다(하이브리드 어텐션 + E5M2 XQA).

## 골방 CUDA와 비교

| 런타임 | 디코드 | 프리필 |
|---|---:|---:|
| 1Cat-vLLM 1.5.0 타깃 단독 (2×V100, NVFP4) | ~46–48 tok/s | ~2,533 tok/s |
| golbang CUDA (`golbang-cuda-qwen38`, 2×V100, UD-Q4_K_XL + MTP) | 42–45 tok/s | 520–590 tok/s |

- 디코드는 1Cat 타깃 단독이 골방 MTP 대비 약 +5–10%.
- 프리필은 1Cat이 약 4.3–4.9배. 골방은 n-ubatch 512, 1Cat은 chunked prefill이라 조건 차이가 있으나 격차가 크다.
- 골방 수치는 짧은 컨텍스트, 1Cat의 8K/16K 수치도 단일 요청이라 대략 비교 가능하다.

## DFlash2 시도 (실패)

드래프터까지 포함하면 2×16GB에 들어가지 않는다.

- 드래프터 체크포인트에는 `lm_head`/`embed_tokens`가 없다(3.85 GiB = 5 layer + selector codebook). 로드 시 타깃 lm_head/embed와 공유된다.
- 그러나 드래프터 transformer가 TP2로 분할되지 않고 **카드마다 복제**된다. 타깃(9.9 GiB, vision 제외) + 드래프터(≈3.85 GiB) ≈ 13.7 GiB로, 15.77 GiB 카드에 KV 여유가 거의 없다.
- `gpu_memory_utilization`을 시작 가능한 최대(0.965)까지 올려도 KV 가용 0.51 GiB < 요구 1.48 GiB(2048 ctx) 로 약 1 GiB 부족.
- DFlash2를 쓰려면 4×V100(README 검증 구성) 또는 2×V100-32GB가 필요하다.

### DFlash2 관련 부수 관찰

- 드래프터 생성자에서 일시적으로 만드는 full-vocab `lm_head`가 OOM의 직접 원인이다(이후 타깃 lm_head로 교체됨). 이 일시 할당을 건너뛰도록 로컬 패치하면 OOM은 사라지지만, 그 다음 KV 캐시 부족(`No available memory for the cache blocks`)에 걸린다. 패치는 실험 후 원복했다.
- `draft_tensor_parallel_size:2`를 지정해도 드래프터 가중치는 분할되지 않았다.

## 관련

`qwen38-vllm-vs-golbang`, `qwen38-on-golbang`, `unsloth-q3-on-v100`, `dflash2-on-golbang`, `cuda-hip-separate-builds`
