# PLD (prompt lookup) A/B — Qwen3.8-Flash-Next

> 기록: 2026-09-17. 이슈 #247(HAL-3), 상위 #244. 같은 바이너리에서
> `--spec-type`만 바꾼 A/B다. 관련: 위키 `halogen-borrowable-techniques` §1,
> `qwen38-mtp`, `qwen38-flashnext-speedup`.

## 무엇을 넣었나

`SpecType::PromptLookup`(`prompt-lookup` | `pld`)을 추가했다. 요청 자신의 토큰
이력(프롬프트 + 생성분)에서 마지막 `pld_n`(=3)개 접미의 이전 출현을 찾아, 그
뒤 `pld_k`(=3)개를 복사해 초안으로 낸다. MTP가 켜져 있으면 MTP 초안 뒤에
PLD 사슬을 이어 붙인다. `verify_n_max()`는 PLD를 `pld_k`로만 세므로 ngram-mod의
64처럼 부풀지 않는다. greedy(`temperature=0`)이고 슬롯이 단독 생성 중일 때만
활성한다(`pld_allowed`). CLI는 `--spec-pld-n`/`--spec-pld-k`
(env `GOLBANG_SPEC_PLD_N/K`, 기본 3/3).

## 조건

2×V100-SXM2-16GB + EPYC 7452. 프로덕션 유닛과 같은 파라미터
(`--n-gpu-layers 99 --n-cpu-moe 42 --flash-attn on --tensor-split 11,1
--n-ctx 100000 --n-batch 4096 --n-ubatch 2048 --n-threads 32 --n-parallel 1`),
MTP draft `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`, `--spec-draft-n-max 2`.
greedy, `max_tokens 400`, 프롬프트와 무관하게 동일 바이너리.

- **copy**: 40개 함수 선언(정확한 이름·파라미터 목록)을 Markdown 표로 그대로
  옮기게 하는 요청(prompt 2474 토큰). 접미 복사가 많은 coding-agent형 작업.
- **prose**: 분산 합의 난점을 설명하는 산문(prompt 87 토큰). 회귀 대조군.

서버가 CPU expert를 42층 오프로드하므로 외부 CPU 부하에 민감하다. 한 차례
`llama-bench`(다른 작업)가 10코어를 점유한 구간의 측정은 전부 폐기하고,
부하가 잦아든 뒤 off/on을 3회 교차 측정해 중앙값을 썼다.

## 결과 (동일 바이너리, `--spec-type`만 변경)

| 프롬프트 | PLD | decode (t/s) | mean accepted len | acceptance | draft_n |
|--|--|--:|--:|--:|--:|
| copy | off | 35.43 | 2.99 | 1.000 | 263 |
| copy | on | **35.83** | **3.72** | 0.938 | 307 |
| prose | off | 23.87 | 2.41 | 0.929 | 155 |
| prose | on | 23.75 | 2.43 | 0.925 | 159 |

3회 교차 측정은 편차 없이 같은 값이 나왔다(예: copy on 35.01/35.95/35.83).

- **copy**: mean accepted length가 2.99 → 3.72(+24%)로 늘어 PLD가 실제로
  복사를 잡는다. 그러나 decode는 35.43 → 35.83 t/s로 **+1.1%**에 그쳤다.
  검증 배치가 3토큰(MTP 2 + 1 sampled)에서 4토큰으로 커지면서 CPU expert
  합집합 비용이 같이 늘어, 수락 1토큰의 이득을 상쇄하기 때문이다.
- **prose**: 23.87 → 23.75 t/s로 잡음 이내(0% 회귀). PLD가 거의 발화하지
  않는다(mean len 2.41 → 2.43).

## 판정

**이득이 목표(+5~10%)에 못 미친다.** 목표 대비 미달이며, MTP acceptance가
이미 0.98~1.00, mean len 2.49~2.78인 유닛에서는 PLD의 한계 이득이 작다는
이슈의 위험 예상이 맞았다. 그래서 유닛 옵션은 **기본 off**로 둔다. 기능은
구현되어 있어 `--spec-type draft-mtp,prompt-lookup`으로 켤 수 있다.

## 구현 함정: `n_rs_seq`

첫 A/B에서 PLD on이 copy 35.4 → 19.1 t/s로 **폭락**했다. 원인은 `n_rs_seq`였다.
프로덕션은 `--n-rs-seq 1`이고 코드가 MTP `n_max`(=2)까지 올리는데, PLD가 세 번째
초안을 붙이면 `drafts.len()=3 > n_rs_seq=2`가 되어 `snapshot_spec_slots`가 매
검증마다 115 MiB `PARTIAL_ONLY` 스냅샷을 떴다(`snapshot_ms=7357`). PLD의 reject
span을 덮도록 `n_rs_seq = max(n_max, pld_k)`로 올리고 나서야 정상 측정이 됐다
(`spec_rs_need`, 단위 테스트 포함). ngram-mod가 스냅샷을 동반하는 것과 같은
함정이므로, 추측 초안 길이를 늘리는 변경은 반드시 `n_rs_seq`를 함께 본다.

## 재현

```bash
golbang-server --model <flash-next 00001> \
  --n-gpu-layers 99 --n-cpu-moe 42 --flash-attn on --tensor-split 11,1 \
  --n-ctx 100000 --n-batch 4096 --n-ubatch 2048 --n-threads 32 --n-parallel 1 \
  --jinja --reasoning-format deepseek \
  --model-draft .../mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --spec-type draft-mtp,prompt-lookup --spec-draft-n-max 2 --spec-pld-n 3 --spec-pld-k 3
```
