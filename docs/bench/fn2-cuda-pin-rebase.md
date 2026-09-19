# CUDA 핀 리베이스 A/B — #28896 rms_norm+mul fusion / #28901 hc ops

> 기록: 2026-09-19. 이슈 #255(FN-SPEED-2), 상위 #253. CUDA 핀을
> `1c4cfda6c`(= `c069aa7f5f` + PR #28243)에서 `53b1389d0`(= `911f6cdc8a` +
> PR #28243, conflict-resolved)으로 올린 뒤, MTP 추측 디코딩을 유지한 채
> 같은 조건에서 A/B했습니다. 관련: 위키 `qwen38-cuda-pin-rebase-28896-28901`,
> `qwen38-mtp` §7-8, `qwen38-flashnext-speedup`, `patches/README.md`.

## 한 줄

**속도 차이는 측정 노이즈 범위였습니다.** short decode가 중앙값 기준
30.73 → 31.84 t/s(+3.6%, n=9), ~15k는 28.82 → 28.87 t/s(+0.2%, n=5),
~80k는 22.38 → 23.33 t/s(+4.2%, n=1)로 모두 방향은 약간 우세했지만 폭이
작고 표본 분산이 큽니다. prefill은 226.7 → 221.3 t/s(@15k), 205.9 → 202.7
t/s(@80k)로 오히려 소폭 낮았습니다. MTP acceptance는 0.92 안팎으로
동일했습니다. **회귀는 없고 MTP가 그대로 동작하므로, 유지보수(핀 드리프트
해소)를 위해 채택**했습니다 — 가속을 근거로 채택한 것은 아닙니다.

#28896/#28901은 CPU·ROCm·Vulkan에서 prefill/노드 수를 줄이는 변경인데,
이 유닛은 42/48층 expert가 호스트 CPU에 있어 디코드가 **CPU expert GEMV**에
묶여 있습니다(`qwen38-flashnext-speedup` 진단). GPU 커널·그래프 노드 감소가
지배 비용을 건드리지 않아 기대한 이득이 나타나지 않은 것으로 봅니다.

## 1. 변경 내용

| 항목 | 이전 | 이후 |
|--|--|--|
| base | `c069aa7f5f` (tag b10924) | `911f6cdc8a` (origin/master, 2026-09-18) |
| patch | `cuda/0001` (PR #28243 vs c069aa7f5f) | `cuda/0001` (PR head `53b1389d0` vs `911f6cdc8a`) |
| 최종 핀 | `1c4cfda6c` | `53b1389d0` |
| 새로 포함 | – | #28896 `rms_norm + mul` fusion, #28901 `hc ops`, #28988 vulkan hc ops |

`53b1389d0`은 PR #28243이 master `911f6cdc8a`를 merge하며 충돌을 해소한
head입니다. 그래서 base를 `911f6cdc8a`로 올리고 `git diff 911f6cdc8a
53b1389d0`를 패치로 쓰면 **base + 패치 == PR head 트리**가 바이트 동일하게
재현됩니다(`patches/README.md`).

## 2. 조건

- 2×V100-SXM2-16GB + EPYC 7452(32C/64T), mmap, 페이지 캐시 웜.
- 두 바이너리 모두 `deploy/golbang-cuda-flashnext.service`와 같은 인자:
  `--n-gpu-layers 99 --n-cpu-moe 42 --flash-attn on --tensor-split 11,1
  --n-ctx 100000 --n-batch 4096 --n-ubatch 2048 --n-threads 32 --n-rs-seq 1
  --n-parallel 1 --queue-size 2 --jinja --reasoning-format deepseek
  --temperature 1.0 --top-p 0.95 --top-k 20`, MTP draft
  `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`, `--spec-type draft-mtp
  --spec-draft-n-max 4 --spec-draft-p-min 0.90`.
- 프롬프트는 `docs/bench/fn-mtp-nmax.md`와 같은 √2 무리수 증명 문제를, 긴
  구간에서는 기술 지문 뒤에 붙였습니다. 실제 토큰 수는 short 68, ~15k
  14,809, ~80k 78,169입니다(요청 `timings.prompt_n`).
- decode는 응답 `timings.predicted_per_second`, prefill은 `prompt_per_second`.
  같은 프롬프트를 반복하면 `cache_n`이 채워져 prefill은 첫 러닝만 유효합니다
  (`cache_n == 0`).

## 3. 결과 — 1차 (short×3, mid×2, long×1)

| 구간 | baseline `1c4cfda6c` | 신규 `53b1389d0` | Δ |
|--|--:|--:|--:|
| short decode (t/s) | 30.70 / 30.73 / 32.02 | 31.21 / 28.59 / 32.64 | – |
| ~15k decode (t/s) | 26.14 / 28.82 | 29.35 / 25.94 | – |
| ~80k decode (t/s) | 22.38 | 23.33 | +4.2% |
| ~15k prefill (t/s) | 226.7 | 221.3 | -2.4% |
| ~80k prefill (t/s) | 205.9 | 202.7 | -1.6% |
| short acceptance | 0.885–0.924 | 0.890–0.926 | – |
| ~80k acceptance | 0.897 | 0.927 | – |

## 4. 결과 — 반복 측정 (short×6, mid×3, 기존 표본 합산)

분산이 커서 short와 mid를 더 돌려 중앙값을 봤습니다(1차 표본과 합산).

| 구간 | n | baseline 중앙 (t/s) | 신규 중앙 (t/s) | Δ |
|--|--:|--:|--:|--:|
| short decode | 9 | 30.73 | 31.84 | +3.6% |
| ~15k decode | 5 | 28.82 | 28.87 | +0.2% |
| ~80k decode | 1 | 22.38 | 23.33 | +4.2% |
| short acceptance | 9 | 0.923 | 0.926 | – |
| ~15k acceptance | 5 | 0.911 | 0.901 | – |
| ~15k prefill | 2 (cache_n=0) | 226.7 / 225.7 | 221.3 / 229.8 | σ 안 |

개별 decode 값:

- baseline short: 29.53, 31.93, 31.58, 30.34, 31.35, 29.54, 30.70, 30.73, 32.02
- 신규 short: 32.24, 33.07, 30.58, 33.17, 31.84, 31.36, 31.21, 28.59, 32.64
- baseline ~15k: 29.95, 29.40, 28.08, 26.14, 28.82
- 신규 ~15k: 28.87, 27.03, 29.42, 29.35, 25.94

## 5. VRAM / 안전성

| | GPU0 used/free (MiB) | GPU1 used/free (MiB) |
|--|--:|--:|
| baseline `1c4cfda6c` | 14345 / 2039 | 14417 / 1967 |
| 신규 `53b1389d0` | 14897 / 1487 | 14369 / 2015 |

신규 트리에서 GPU0가 +552 MiB를 더 씁니다(#28901의 hc ops 버퍼로 봅니다).
여유 1.5 GiB로 OOM은 없었고, MTP draft는 정상 로드·동작했습니다
(`MTP draft context ready ... n_nextn=1 n_max=4 separate_draft=true`).

## 6. 판정

**채택 (forward rebase).** 속도는 유의미하게 변하지 않았습니다. 채택 근거는
속도가 아니라 (1) MTP가 새 base에서도 그대로 동작하고, (2) 회귀가 없으며,
(3) `1c4cfda6c`에 머무르면 다음 리베이스의 드리프트(이미 112 커밋)만 커지기
때문입니다. 롤백 트리는 `local-llm/llama.cpp-cuda-upstream-1c4cfda6c`에
트리·빌드·`.git`을 통째로 보존했습니다.

## 7. 재현

```bash
# 새 트리 받아 패치 적용 + 빌드 (CUDA 12.8은 호스트 gcc 16을 거부하므로
# 스크립트가 g++-14 고정 + --allow-unsupported-compiler를 넣는다)
scripts/build-llama.sh cuda --dir /home/agurrrrr/code/local-llm/llama.cpp-cuda-upstream --force

# golbang-server 재링크
CARGO_TARGET_DIR=target-cuda GOLBANG_GPU=cuda \
  cargo build -p golbang-server --release

# 유닛 재기동
sudo systemctl restart golbang-cuda-flashnext.service
```

원시 응답은 `docs/bench/raw/fn2-cuda-pin-rebase/`(baseline/, next/는 1차
러닝, `reps_*.json`은 반복 측정)에 있습니다.
