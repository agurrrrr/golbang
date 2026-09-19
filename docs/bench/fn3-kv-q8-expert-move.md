# KV q8_0 재시험 + VRAM 회수/expert 층 이전 A/B — Qwen3.8-Flash-Next

> 기록: 2026-09-19. 이슈 #256(FN-SPEED-3), 상위 #253. CUDA 핀 `53b1389d0`
> (= upstream `911f6cdc8a` + PR #28243)에서 KV 캐시를 f16에서 q8_0으로
> 낮춰 `self_k_rot` assert 없이 동작하는지 재시험하고, 회수한 VRAM으로
> expert 층을 GPU로 더 옮겨 decode가 개선되는지 A/B했습니다. 관련:
> 위키 `qwen38-flashnext-speedup` §3.2, `qwen38-mtp`, `qwen38-cuda-pin-rebase-28896-28901`,
> `docs/bench/fn2-cuda-pin-rebase.md`.

## 한 줄

**q8_0 KV는 현재 핀 트리에서 안전하게 동작합니다.** 로드·서빙·80k 장문맥까지
크래시가 없고, 그리디 200토큰 출력은 f16과 byte-identical입니다. KV 크기는
100k 컨텍스트에서 약 2.6 GiB에서 약 1.4 GiB로 줄어 **약 1.1~1.2 GiB를
회수**했고, 이 회수분으로 expert 층을 1개 더 GPU로 옮기는 것(`--n-cpu-moe`
42→41)이 실제로 가능해졌습니다(f16 KV로는 GPU0 OOM). 그러나 **decode는
개선되지 않았습니다.** 교차 A/B에서 short는 사실상 동일하고, mid는 노이즈
범위이며, 80k는 오히려 소폭 낮았습니다(22.46 vs 23.36 t/s). 2층 이동
(`n-cpu-moe` 40)은 MTP 드래프트가 GPU1 OOM으로 비활성화되거나 mid에서
크래시했습니다. 따라서 **KV q8_0과 expert 층 이전은 모두 비채택**하고
프로덕션은 f16 KV·`n-cpu-moe` 42로 유지합니다.

## 1. 배경과 재시험 근거

위키 §3.2는 q8_0 KV로 약 1.3 GiB를 회수해 expert 층 1개를 더 옮길 수
있다고 봤습니다. 과거 관측은 qwen4exp에서 q8_0 KV 사용 시 `self_k_rot`
assert로 크래시한다는 것이었습니다. 현재 트리에는 PR #27742의 후속으로
`build_attn_qsa`가 rotation을 처리하도록 바뀌어 있습니다
(`src/models/qwen4exp.cpp:941-948`, `inp->self_k_rot`/`inp->self_v_rot`일 때
`llama_mul_mat_hadamard`를 q/k/v에 적용). 실제로 기동 로그에도
`llama_kv_cache: attn_rot_k = 1, n_embd_head_k_all = 256`과
`attn_rot_v = 1`이 출력되어 rotation 경로가 활성임을 확인했습니다.

## 2. 조건

- 2×V100-SXM2-16GB + EPYC 7452(32C/64T), mmap, 페이지 캐시 웜.
- 모든 런은 `deploy/golbang-cuda-flashnext.service`와 같은 인자에 KV 타입만
  바꿔 수동 기동했습니다: `--n-gpu-layers 99 --flash-attn on --n-ctx 100000
  --n-batch 4096 --n-ubatch 2048 --n-threads 32 --n-rs-seq 1 --n-parallel 1
  --queue-size 2 --jinja --reasoning-format deepseek --temperature 1.0
  --top-p 0.95 --top-k 20`, MTP draft `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`,
  `--spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.90`.
- 프롬프트는 `docs/bench/fn-mtp-nmax.md`와 같은 √2 무리수 증명 문제를, 긴
  구간에서는 기술 지문 뒤에 붙였습니다. 실제 토큰 수는 short 68, ~15k
  14,809, ~80k 78,169입니다.
- decode는 응답 `timings.predicted_per_second`, prefill은
  `prompt_per_second`입니다.
- **주의(분산 요인):** 측정 전 구간에 포트 8084의 HIP Qwen3.8-27B 서버가
  상시 동작하고 있었습니다. 두 설정에 공통으로 작용하므로 상대 비교는
  유효하지만, 절대값의 분산이 커졌습니다. 이 때문에 단발 표 대신 교차
  A/B(round1 A→B→C, round2 C→B→A)로 중앙값을 비교했습니다.

## 3. 로드·안전성 — `self_k_rot` assert는 재현되지 않음

| 설정 | 로드 | `self_k_rot` assert | 그리디 200토큰 |
|--|--|--|--|
| f16 KV, `n-cpu-moe` 42 | 성공 | – | 기준 |
| q8_0 KV, `n-cpu-moe` 42 | 성공 | 없음 | f16과 **byte-identical** |
| q8_0 KV, `n-cpu-moe` 41 | 성공 | 없음 | 수치적으로 다름(동급) |

q8_0 KV는 로드·첫 요청·80k 프리필·80k 디코드 전 구간에서 크래시하지
않았습니다. 과거 assert는 현재 트리에서 재현되지 않습니다.

출력 동일성은 NUMERIC 등급입니다. `n-cpu-moe` 42를 유지한 q8_0 KV는
그리디 200토큰이 f16과 완전히 같았지만(양자화된 KV가 attention 점수 순위를
바꾸지 않을 만큼 여유가 있음), expert 층을 1개 옮긴 `n-cpu-moe` 41에서는
동일성 등급이 낮아집니다(초반 문장 표현이 갈림). 이는 KV 양자화가 아니라
expert 연산 경로 변경에서 오는 차이로 봅니다.

## 4. VRAM — KV 약 1.1~1.2 GiB 회수

100k 컨텍스트의 KV 크기(기동 로그):

| 구성 요소 | f16 KV | q8_0 KV | 회수 |
|--|--:|--:|--:|
| full-attn 12층 K+V | 2346.00 MiB | 1246.31 MiB | 1099.69 MiB |
| indexer 12층 K | 293.25 MiB | 155.79 MiB | 137.46 MiB |
| draft head 1층 K+V | 103.86 MiB | 103.86 MiB | 0 (이미 q8_0) |
| 합계 | 약 2743 MiB | 약 1506 MiB | **약 1237 MiB** |

기동 직후 VRAM(수동 서버, MTP 드래프트 포함):

| 설정 | GPU0 used/free (MiB) | GPU1 used/free (MiB) |
|--|--:|--:|
| f16 KV, `n-cpu-moe` 42, split 11,1 | 14897 / 1248 | 14369 / 1777 |
| q8_0 KV, `n-cpu-moe` 42, split 11,1 | 13763 / 2382 | 14313 / 1833 |
| q8_0 KV, `n-cpu-moe` 41, split 11,1 | 15263 / 882 | 14313 / 1833 |

## 5. expert 층 이전 — 1층은 가능, 2층은 불가

### 5.1 1층 이동(`n-cpu-moe` 42→41)은 q8_0 KV가 있어야만 가능

같은 `--tensor-split 11,1`에서:

| 설정 | 결과 |
|--|--|
| f16 KV, `n-cpu-moe` 41 | **GPU0 OOM** (`failed to allocate compute pp buffers`, `llama_init_from_model returned null`) |
| q8_0 KV, `n-cpu-moe` 41 | 로드 성공, MTP 드래프트 포함, GPU0 여유 882 MiB |

즉 q8_0 KV가 회수한 약 1.1 GiB가 expert 1층(약 1.5 GiB 중 GPU0 분담분)을
GPU로 올리는 결정적 여유가 되었습니다. 이슈가 예상한 인과는 성립합니다.

### 5.2 2층 이동(`n-cpu-moe` 40)은 드래프트가 들어가지 않음

`n-cpu-moe` 40에서 `--tensor-split`을 여러 값으로 스윕했습니다.

| split | main model | MTP draft | 비고 |
|--|--|--|--|
| 11,1 | 로드 성공 | **비활성** | GPU0 부족 |
| 11,1.4~1.5 | 로드 성공 | 로드 성공 | GPU1 여유 235 MiB → mid에서 OOM 크래시 |
| 11,1.6~1.7 | 로드 성공 | **비활성** | MTP 컨텍스트 GPU1 OOM |
| 10,2 | 로드 성공 | 비활성 | GPU1 부족 |
| 11,2.0 | 로드 성공 | 비활성 | GPU1 부족 |

드래프트가 붙는 유일한 split(11,1.5)은 GPU1 여유가 235 MiB뿐이어서, mid
벤치 중 `RemoteDisconnected`로 서버가 죽었습니다. 따라서 **2층 이동은
현재 2×16 GiB 배치에서 안정적으로 불가능**합니다.

## 6. 속도 — 개선 없음

### 6.1 단발 3구간 (short×3, mid×2, long×1)

| 구간 | f16 / ncm42 | q8_0 / ncm42 | q8_0 / ncm41 |
|--|--:|--:|--:|
| short decode (t/s) | 32.16 | 31.08 | 32.47 |
| ~15k decode (t/s) | 29.08 | 28.31 | 28.38 |
| ~80k decode (t/s) | 23.26 | 22.88 | 22.24 |
| ~15k prefill (t/s) | 226.9 | 231.6 | 234.1 |
| ~80k prefill (t/s) | 204.6 | 209.8 | 207.5 |
| short acceptance | 0.88–0.93 | 0.92 | 0.91 |

### 6.2 교차 A/B (round1 f16→q8(42)→q8(41), round2 q8(41)→q8(42)→f16)

| 설정 | short 중앙 (n=8) | mid 중앙 (n=6) | ~15k prefill 중앙 |
|--|--:|--:|--:|
| f16 KV, `n-cpu-moe` 42 | 32.42 | 28.75 | 225.1 |
| q8_0 KV, `n-cpu-moe` 42 | 32.04 | 28.21 | 224.9 |
| q8_0 KV, `n-cpu-moe` 41 | 32.39 | 29.32 | 228.5 |

### 6.3 80k 장문맥 교차 A/B (f16→q8→q8→f16, 각 2회)

| 설정 | ~80k decode (t/s) | ~80k prefill (t/s) | acceptance |
|--|--:|--:|--:|
| f16 KV, `n-cpu-moe` 42 | 24.04 / 23.08 / 21.20 / 23.65 (중앙 23.36) | 205.5 / 204.6 | 0.92 |
| q8_0 KV, `n-cpu-moe` 41 | 21.31 / 22.87 / 23.14 / 22.06 (중앙 22.46) | 208.5 / 208.8 | 0.90 |

**판정:** short는 차이가 없고, mid는 노이즈 범위이며, 80k는 q8_0이
중앙값 기준 약 4퍼센트 낮았습니다. prefill은 오히려 q8_0이 소폭 높았지만
1~2퍼센트로 노이즈 범위입니다. expert 층을 1개 GPU로 옮겨도 decode가
개선되지 않은 이유는 §2 진단과 일치합니다. 이 유닛의 decode는 CPU에 남은
41~42개 층의 expert GEMV에 묶여 있고, 층 하나를 옮겨 줄이는 CPU 작업량이
측정 분산보다 작습니다.

## 7. 판정

**비채택.** 두 축 모두 채택 근거가 없습니다.

1. **KV q8_0**: 크래시 없이 동작하고 VRAM을 약 1.1 GiB 회수하지만, decode와
   prefill이 개선되지 않습니다. 이 유닛은 VRAM이 아니라 CPU expert 연산이
   병목이므로, VRAM 회수 자체가 속도로 이어지지 않습니다. 회수한 VRAM으로
   expert 1층을 옮겨도 마찬가지입니다. 장문맥 attention 대역폭 감소 효과도
   관측되지 않았습니다(80k decode 소폭 하락).
2. **expert 층 이전**: 1층(ncm 41)은 q8_0 KV 전제에서만 로드되지만 이득이
   없고, 2층(ncm 40)은 MTP 드래프트가 안정적으로 들어가지 않습니다.

따라서 프로덕션 `deploy/golbang-cuda-flashnext.service`는 f16 KV와
`--n-cpu-moe 42`를 유지합니다. 다만 q8_0 KV가 안전하다는 사실과 회수량은
향후 다른 이유로 VRAM이 필요할 때(예: n_ctx 상향, 슬롯 증설) 재사용할 수
있으므로 위키에 기록합니다.

## 8. 재현

```bash
# 수동 기동 (KV 타입과 n-cpu-moe만 바꿔 비교)
LD_LIBRARY_PATH=/home/agurrrrr/code/local-llm/llama.cpp-cuda-upstream/build/bin:/opt/cuda-12.8/lib64 \
CUDA_VISIBLE_DEVICES=0,1 CUDA_DEVICE_ORDER=PCI_BUS_ID \
target-cuda/release/golbang-server \
  --model /home/agurrrrr/models/qwen3.8/Qwen3.8-Flash-Next-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --alias qwen3.8-flash-next --host 0.0.0.0 --port 8090 \
  --n-gpu-layers 99 --n-cpu-moe 41 --flash-attn on --tensor-split 11,1 \
  --kv-type-k q8_0 --kv-type-v q8_0 \
  --n-ctx 100000 --n-batch 4096 --n-ubatch 2048 --n-threads 32 --n-rs-seq 1 \
  --n-parallel 1 --queue-size 2 --jinja --reasoning-format deepseek \
  --temperature 1.0 --top-p 0.95 --top-k 20 \
  --model-draft /home/agurrrrr/models/qwen3.8/mtp-flashnext/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.90 \
  --api-key "$GOLBANG_API_KEY"

# KV 크기와 rotation 경로는 기동 로그에서 확인
#   llama_kv_cache: size = ... K (q8_0): ... V (q8_0): ...
#   llama_kv_cache: attn_rot_k = 1, n_embd_head_k_all = 256
```

원시 응답은 `docs/bench/raw/fn3-kv-q8/`에 있습니다. `baseline_f16*`는 f16
기준선, `kvq8_ncm42*`·`kvq8_ncm41*`는 단발 3구간, `il_*`는 교차 A/B,
`la_*`는 80k 교차, `q8_ncm40_11_1p5`는 2층 이동 시도(크래시)입니다.
