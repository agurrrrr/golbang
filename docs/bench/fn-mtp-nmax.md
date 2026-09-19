# MTP 추측 깊이·확률 튜닝 (n-max / p-min) — Qwen3.8-Flash-Next

> 기록: 2026-09-19. 이슈 #254(FN-SPEED-1), 상위 #253. 같은 바이너리
> (`target-cuda/release/golbang-server`, vendor 핀 `1c4cfda6c`)에서
> `--spec-draft-n-max`와 `--spec-draft-p-min`만 바꾼 A/B다. 관련:
> 위키 `qwen38-mtp` §7, `qwen38-flashnext-speedup`, `pld-flashnext`,
> llama.cpp Discussion #25198, PR #22673, bhodgens V100 벤치.

## 한 줄

현행 `n-max 2`(p-min은 코드 기본값 0.90)를 **`n-max 4` + `p-min 0.90`**으로
올리면 decode가 짧은 컨텍스트 **26.1 → 30.3 t/s(+16%)**, 약 15k **20.6 → 27.4
t/s(+33%)**, 약 80k **15.9 → 23.2 t/s(+45%)**로 올라갑니다. acceptance는
0.94~0.96으로 거의 그대로이고, mean accepted length만 2.4 → 3.7로 늘어납니다.
p-min을 0.85로 낮추면 acceptance가 0.89~0.94로 떨어져 손해였고, 0.95로 올리면
draft가 일찍 끊겨 이득이 줄었습니다. n-max는 4가 정점이고 5·6은 회귀했습니다.

## 0. 중요한 정정: p-min 기본값은 0.0이 아니라 0.90

이슈 본문과 직전 위키는 "현행 유닛은 p-min 미지정(기본 0.0)"이라고 적었지만,
실제 런타임 기본값은 **0.90**입니다.

- `golbang-server/src/main.rs:193` — `spec_draft_p_min: f32` 기본값 `0.90`
- `golbang-core/src/speculative.rs:82` — `SpecParams::default().p_min = 0.90`
- 기동 로그 실측(2026-09-18 00:33): `p_min=0.8999999761581421`

그래서 "기준선 = n-max 2 + p-min 없음"은 실제로 **n-max 2 + p-min 0.90**입니다.
이번 스윕은 이 현행 기본값을 기준선으로 두고 n-max를 올리는 방향과 p-min을
0.85/0.95로 흔드는 방향을 함께 봤습니다. 유닛에는 이제 p-min 0.90을 **명시**합니다.

## 1. 조건

- 2×V100-SXM2-16GB + EPYC 7452(32C/64T), mmap, 페이지 캐시 웜.
- 프로덕션 유닛과 같은 파라미터: `--n-gpu-layers 99 --n-cpu-moe 42 --flash-attn on
  --tensor-split 11,1 --n-ctx 100000 --n-batch 4096 --n-ubatch 2048 --n-threads 32
  --n-rs-seq 1 --n-parallel 1 --queue-size 2 --jinja --reasoning-format deepseek
  --temperature 1.0 --top-p 0.95 --top-k 20`, MTP draft
  `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`.
- 샘플링은 유닛 기본값(thinking 카드, temperature 1.0)을 그대로 썼습니다. 즉
  실제 서빙 조건입니다.
- 프롬프트:
  - **short**: "√2가 무리수임을 증명하라"는 사고 유도 문제, `max_tokens 1500`,
    3회.
  - **mid**: ~15k 토큰 지문 뒤에 같은 증명 문제, `max_tokens 1500`, 2회.
  - **long**: ~80k 토큰 지문 뒤에 같은 증명 문제, `max_tokens 1200`, 1회(일부 2회).
- `n_rs_seq`는 코드가 `n_max`까지 자동 상향합니다(로그: `n_rs_seq raised ... to=N`).
  전 구간에서 스냅샷(`snapshot_ms`)은 0이었고, PARTIAL_ONLY 스냅샷은 재발하지
  않았습니다(#247 함정 회피 확인).

## 2. 결과 (decode, tok/s)

유닛에 반영한 조합을 굵게 표시했습니다. 괄호는 개별 러닝, 대표값은 중앙값입니다.

| n-max | p-min | short 중앙(개별) | ~15k 중앙(개별) | ~80k (개별) |
|--:|--:|--:|--:|--:|
| 2 (현행) | 0.90 | 26.1 (26.9/22.4/26.1) | 20.6 (20.7/20.5) | 15.9 |
| 3 | 0.90 | 28.0 (28.9/22.8/28.0) | 27.0 (27.0/27.0) | 16.6/16.1 |
| 3 | 0.95 | 27.0 (22.1/27.0/27.3) | – | – |
| 3 | 0.85 | 24.1 (24.1/23.3/28.2) | – | – |
| **4** | **0.90** | **30.3 (29.7/30.3/30.3)** | **27.4 (27.3/27.6)** | **23.8/22.6** |
| 4 | 0.95 | 28.0 (28.0/29.5/22.7) | – | – |
| 4 | 0.85 | 29.5 (23.7/29.5/29.6) | – | – |
| 5 | 0.90 | 23.8 (23.8/30.8/22.7) | 24.4 (21.9/26.9) | – |
| 6 | 0.90 | 28.3 (28.3/22.4/28.4) | 23.9 (21.2/26.6) | – |

prefill은 조합과 무관하게 동일했습니다(15k ~228–233 t/s, 80k ~211–212 t/s).

## 3. acceptance / mean accepted length

| n-max | p-min | 구간 | acceptance | mean len |
|--:|--:|--|--:|--:|
| 2 | 0.90 | short | 0.951–0.976 | 2.41–2.69 |
| 2 | 0.90 | ~15k | 0.942–0.944 | 2.41–2.42 |
| 2 | 0.90 | ~80k | 0.944 | 2.45 |
| 3 | 0.90 | short | 0.955–0.965 | 2.74–3.25 |
| 3 | 0.90 | ~15k | 0.962–0.977 | – |
| 3 | 0.90 | ~80k | 0.911–0.929 | 2.56–2.67 |
| **4** | **0.90** | short | 0.954–0.971 | 3.65–3.78 |
| **4** | **0.90** | ~15k | 0.938–0.943 | 3.58–3.66 |
| **4** | **0.90** | ~80k | 0.957–0.965 | 3.76–3.86 |
| 4 | 0.85 | short | 0.893–0.935 | 2.83–3.70 |
| 4 | 0.95 | short | 0.964–0.968 | 2.78–3.68 |
| 5 | 0.90 | ~15k | 0.920–0.924 | – |
| 6 | 0.90 | ~15k | 0.904–0.926 | – |

- n-max를 4까지 올려도 **acceptance는 거의 유지**되고 mean len만 늘어납니다.
  이는 p-min 0.90이 확신 없는 draft를 이미 조기 중단시키기 때문입니다.
  즉 "확신 있는 구간에서만 더 깊게 뽑는" 조합입니다.
- p-min을 0.85로 낮추면(더 공격적으로 뽑으면) acceptance가 눈에 띄게 떨어져
  CPU expert 검증 낭비가 커지고 손해였습니다. 0.95는 draft가 일찍 끊겨
  mean len 이득이 줄었습니다.
- **n=4 정점, n=5·6 회귀.** dense V100 벤치의 "n=3 sweet spot"과 달리 이
  MoE/CPU-expert 구성에서는 4가 정점입니다. 검증 배치가 5토큰이 될 때 CPU
  expert GEMV가 GEMM에 더 가까워지는 효과로 봅니다. 5를 넘기면 GPU draft
  비용과 검증 유니온이 이득을 상쇄합니다.

## 4. VRAM / 안전성

| n-max | GPU0 used/free | GPU1 used/free |
|--:|--:|--:|
| 2 | 14043 / 2102 MiB | 14337 / 1809 MiB |
| 3 | 14149 / 1996 MiB | 14343 / 1803 MiB |
| 4 | 14255 / 1890 MiB | 14349 / 1797 MiB |

n-max 1단계당 GPU0 약 106 MiB, GPU1 약 6 MiB만 늘어 안전합니다. n-max 4에서도
GPU0 1890 MiB, GPU1 1797 MiB가 남습니다. OOM 없음. 출력은 정상 문장이었고,
추측 디코딩 특성상 **NUMERIC 등급(no-spec greedy와 byte-identical 아님)**은
유지됩니다.

## 5. 판정

**`--spec-draft-n-max 4 --spec-draft-p-min 0.90` 채택.** 세 컨텍스트 구간
모두에서 현행(n-max 2)보다 우세하고, 특히 80k에서 +45%로 큰 폭입니다.
p-min은 코드 기본값과 같은 0.90이지만, 의도를 드러내기 위해 유닛에 **명시**합니다.

## 6. 재현

```bash
golbang-server --model <flash-next 00001> \
  --n-gpu-layers 99 --n-cpu-moe 42 --flash-attn on --tensor-split 11,1 \
  --n-ctx 100000 --n-batch 4096 --n-ubatch 2048 --n-threads 32 --n-rs-seq 1 \
  --n-parallel 1 --queue-size 2 \
  --jinja --reasoning-format deepseek \
  --temperature 1.0 --top-p 0.95 --top-k 20 \
  --model-draft .../mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.90
```

원시 응답 JSON은 `docs/bench/raw/fn-mtp-nmax/`에 있습니다.
