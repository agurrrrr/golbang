# P7 — Qwen3.8-27B decode를 llama-server #364 속도 이상으로

> **이슈:** #31 · **선행:** P5(#28), MTP+비전(task 344/347), 핀 `3ac5658c7`(task 350) · **후행:** 없음
> **성격:** 같은 카드·같은 GGUF·같은 플래그에서 `golbang-qwen38` decode가
> 생산 `qwen3.8-27b-q6.service`(llama-server)의 **local-llm 작업 #364 실측**을
> 밴드별로 맞추거나 넘긴다. DSV4 / `n_cpu_moe` / P4 Rust 커널이 아니다.

---

## 1. 목표

`golbang-qwen38.service`로 Qwen3.8-27B Q6_K를 돌릴 때, **같은 MI50 · 같은
클럭/전력 · 같은 GGUF · 같은 spec/비전 플래그 · 같은 llama.cpp SHA**에서
llama-server보다 decode가 느리지 않게 만든다.

**완료 기준:** `docs/bench/p7.md`에 통제 A/B가 있고, 아래 밴드에서
golbang ≥ llama-server. 16토큰 버스트만으로 완료를 선언하지 않는다.

---

## 2. 배경 / 근거

### 2.1 대상은 Qwen3.8이지 DeepSeek가 아니다

2026-08-14 DSV4 통제 A/B(`docs/bench/golbang-vs-llama-server.md`)는 decode
~8 tok/s 동률이었다. 병목은 `n_cpu_moe=32`. **그 숫자는 이번 기준이 아니다.**

지금 생산/비교 대상은 dense 27B Q6_K + MTP + ngram-mod다.

### 2.2 llama-server 기준 (local-llm 작업 #364, 2026-08-15)

유닛: `qwen3.8-27b-q6.service`
바이너리: `/home/agurrrrr/code/local-llm/llama.cpp-upgrade/build/bin/llama-server`
SHA: `3ac5658c7`
모델: `/home/agurrrrr/models/qwen3.8/Qwen3.8-27B-Q6_K.gguf` + `mmproj-F16.gguf`
클럭: Level 5 exclusive **1386 MHz** (bitmask `0x20`) + **180W**
옵션: `-ngl 99 --ctx-size 80000 --parallel 1 -b 2048 -ub 2048 --jinja --reasoning on -fa on -t 8 --no-mmap --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --spec-draft-p-min 0.90 --spec-draft-type-k/v q8_0`

| 구간 | decode | draft accept | 비고 |
|------|-------:|-------------:|------|
| 짧은 스모크 16토큰 (`Reply with exactly: OK`) | **25.18 t/s** | 8/8 (100%, mean 3.00) | 아침 28.9에 가장 가까움 |
| 짧은 에이전트 턴 84토큰 | **25.08 t/s** | 53/53 mean 3.65 | accept 좋으면 여기까지 |
| 일반 에이전트 턴 100–230토큰 | **18–23 t/s** | 91–95% | 복실이 실사용 대역 |
| 장문 5822토큰 (task 710) | **17.16 t/s** | 2558/2871 (89%, mean 2.68) | ctx·열 누적 |
| 장문 12288토큰 (task 3978) | **18.52 t/s** | 6446/7377 (87%, mean 3.04) | |
| 실사용 요약 | **17–23 t/s** | 87–100% | 25는 순간 최고 |

25 t/s는 짧은 응답·MTP accept가 좋을 때만. 장문은 junction 열 + KV 때문에
내려간다. 클럭을 올려서 맞추지 말 것 (L6/L7 exclusive는 180W에서 붕괴,
`gpu-settings`).

### 2.3 golbang 현재 (2026-08-15, `golbang-qwen38` 저널)

핀·모델·플래그는 이미 맞춰 둠 (`deploy/golbang-qwen38.service`).
MTP 컨텍스트 로드, `spec=["draft-mtp","ngram-mod"]`, `p_min=0.90`.

| 구간 | decode | 비고 |
|------|-------:|------|
| 재기동 후 20 / 50 / 34토큰 | **10.49 / 10.38 / 10.78 t/s** | ~92–96 ms/tok |
| 85–120토큰 에이전트 | **8.8–10.3 t/s** | |
| 1520토큰 장문 (prompt 8745) | **10.93 t/s** | 91.5 ms/tok |
| 1385–2624토큰, ctx 크다 | **8.3–9.4 t/s** | 107–120 ms/tok |
| prefill | **127–154 t/s** | llama와 비슷하거나 더 나음 |

decode가 **약 2배** 느리다. prefill은 병목이 아니다.
저널에 llama-server식 `draft acceptance = …` 한 줄이 요청 종료 타이밍에
거의 안 남는다. `/metrics`의 `golbang_draft_*`로 확인해야 한다.

### 2.4 이미 한 일 (다시 하지 말 것)

- task 343: think 태그, jinja `reasoning_effort`, `--no-mmap`/`--alias`
- task 344: MTP + `--mmproj` + 유닛 플래그 정렬
- task 347: draft 후 MTP `seq_rm` (M-RoPE `X < Y`)
- task 350: 핀 `3ac5658c7` / `llama.cpp-upgrade` `.so`
- task 352/357: Qwen tool `arguments` 객체 파싱
- P6(#29): DSV4 rocprof. decode 1위는 CPU MoE. **Qwen dense에는 해당 없음**

---

## 3. 산출물

| 산출물 | 위치 | 설명 |
|--------|------|------|
| 통제 A/B | `docs/bench/p7.md` | 같은 프롬프트·같은 클럭·양쪽 유닛 |
| (필요 시) raw JSON | `docs/bench/raw/` | 재현용 |
| 코드 수정 | spec/스케줄/샘플 경로 | 측정이 지목한 것만 |
| 위키 | `qwen38-on-golbang` | 전후 숫자 + 원인 |

---

## 4. 하드웨어 / 공정 조건 (고정)

측정·개선 중에 아래를 **바꾸지 않는다.**

- GPU: MI50 gfx906 한 장. 동시 기동 불가.
- 클럭: Level 5 exclusive 1386 MHz (`rocm-smi --setsclk 5`). L6/L7 실험 금지.
- 전력: 180W. 올리지 말 것.
- 모델: 위 Q6_K + mmproj. 양자화/모델 교체 금지.
- SHA: `3ac5658c7` (`llama.cpp-upgrade`). 생산 `llama.cpp` HEAD에 pull 금지.
- 플래그: 유닛에 있는 spec/ctx/batch를 벤치 중에 임의로 키우지 말 것.
  `p_min` 등은 **측정 후** 후보가 되면 바꾸고 전후를 남긴다.
- `GGML_CUDA_REPACK=1` 은 측정 없이는 켜지 말 것 (생산도 꺼져 있음).

전환:

```bash
# llama-server → golbang
sudo systemctl stop qwen3.8-27b-q6.service
sudo systemctl start golbang-qwen38.service

# golbang → llama-server
sudo systemctl stop golbang-qwen38.service
sudo systemctl start qwen3.8-27b-q6.service
```

시작 전 `Conflicts`가 상대를 죽이므로 순서가 맞아야 한다.
listen / `offloaded 66/66` / health 확인 뒤에만 벤치.

작업이 끝나도 서비스가 비어 있으면 안 된다. 목표를 달성하면
`golbang-qwen38`을 남겨 두고, 못 미치거나 중간에 끊으면
**작업 시작 때 돌고 있던 유닛**을 다시 켠다.

---

## 5. 성공 기준 (DoD)

같은 프롬프트·temperature=0(가능하면)·비슷한 junction(±3°C)·같은 sclk에서:

| 밴드 | llama-server #364 | golbang 목표 |
|------|------------------:|-------------:|
| 16토큰 스모크 | 25.18 t/s | **≥ 25.18 t/s** |
| ~84토큰 | 25.08 t/s | **≥ 25.08 t/s** |
| 100–230토큰 에이전트 (같은 프롬프트 재현) | 18–23 | **같은 프롬프트에서 ≥ llama** |
| 장문 ≥1k 토큰 | 17–18.5 | **≥ llama** (열·KV를 표에 같이) |

추가로 남겨야 하는 것:

1. 요청 종료 로그에 draft 제안/수락/mean len (llama `draft acceptance`와 같은 정보).
2. `/metrics` `golbang_draft_tokens_total` / `golbang_draft_accepted_total`.
3. 각 밴드에서 sclk / 전력 / junction.
4. 위키 `qwen38-on-golbang`에 전후 표.

장문만 열이 달라 불공정하면 짧은·중간 밴드를 먼저 잠그고, 장문은
“같은 시작 온도에서 N토큰”으로 다시 잰다. **16토큰만 25가 나오고
실사용이 11이면 실패.**

정확도: temperature=0 스모크가 같은 문자열(또는 같은 툴 호출)을 내야 한다.
스펙큘 때문에 답이 바뀌면 수락 경로 버그다.

---

## 6. 단계별 작업

### 6.0 지금 상태 기록

- [ ] `systemctl is-active` 로 어떤 유닛이 8083을 잡았는지 적는다.
- [ ] `rocm-smi` sclk bitmask / 전력 / junction. L5가 아니면
      `sudo /opt/rocm/bin/rocm-smi --setsclk 5` 만 하고 끝낸다.
- [ ] 시작 유닛을 메모해 두어 마지막에 복구한다.

### 6.1 통제 베이스라인 (코드 고치기 전)

같은 세 프롬프트를 **양쪽**에서 돌린다.

1. `Reply with exactly: OK` (max_tokens 16~32)
2. 짧은 설명 한 단락 (max_tokens 84~128)
3. 중간 생성 (max_tokens 200+)

- [ ] 지금 돌고 있는 쪽이 llama-server면 그 숫자부터. 그다음 유닛 전환.
- [ ] golbang은 `eval time` 로그 + `/metrics` draft 카운터.
- [ ] llama-server는 `slot print_timing` + `draft acceptance`.
- [ ] 숫자를 `docs/bench/p7.md` 맨 위에 “고치기 전” 표로 남긴다.

예상: golbang ~10–11 t/s, llama 18–25 t/s, prefill은 비슷.

### 6.2 병목 분류 (추측으로 커널을 바꾸지 말 것)

아래를 **측정으로** 가른다. 한 번에 다 고치지 말고 1순위부터.

1. **스펙큘이 사실상 꺼져 있다**
   - draft 제안 수 ≈ 0, 또는 수락률 ≪ llama(87–100%, mean ~3).
   - `p_min=0.90` 이면 짧은 답에서 MTP draft가 0일 수 있음 (위키).
     ngram-mod가 먼저 채워야 한다. ngram 테이블이 비거나 `n_min`에
     걸리면 초안이 비고, 타깃은 토큰당 1 decode만 한다 → ~10 t/s가
     “스펙큘 없는 천장”일 수 있다.
   - 볼 곳: `maybe_fill_drafts`, `spec_draft` (ngram 우선 → MTP),
     `spec_process` / `pending_h`, draft 후 `seq_rm`,
     `emit_sampled` / `verify_and_emit`.
   - llama-server: `common_speculative_draft` 직후
     `llama_memory_seq_rm(ctx_dft, seq, ckpt.pos_max+1, -1)`.

2. **초안은 나오는데 검증이 더 비싸다**
   - 수락은 되는데 tok/s가 안 오른다 = 검증 배치 + 롤백 + replay가
     이득을 먹는다. `replay_spec_prefix`가 자주 도는지, `seq_rm` 실패
     후 전체 재디코드인지.
   - `verify_and_emit`에서 `accepted.is_empty()` 이면 토큰을 안 내고
     return 한다. 이 경로가 실제로 치면 한 스텝이 버려진다.

3. **호스트 오버헤드**
   - `maybe_fill_drafts`가 매 토큰 `prompt_tokens.clone()` + generated
     복사. 에이전트 턴 1만 토큰이면 매 스텝 O(n).
   - 로짓 전체 vocab(248320)을 Rust로 복사해 샘플.
   - llama-server는 백엔드 샘플러를 쓰려다 gfx906에 TOP_K가 없어
     CPU 폴백(경고 1회). 메인 병목은 아니라고 #364가 적어 둠.
   - 그래프 재사용 / 동기화 / `spawn_blocking` 왕복.

4. **KV / prefix**
   - `prefix checkpoint not usable; full prefill` 가 자주 뜬다.
     decode t/s와는 별개지만 다턴 체감을 깎는다. decode DoD 다음에.

5. **하지 않는 가설**
   - HIP 커널 재작성, P4 Rust 커널, DSV4 `n_cpu_moe`, 클럭 상향,
     모델 재양자화, ctx를 줄여서 속도 사기.

### 6.3 고치고 다시 잰다

- [ ] 한 원인당 커밋 단위로 고친다. 묶어서 “빨라짐”을 주장하지 말 것.
- [ ] 매번 6.1의 짧은·중간 밴드를 다시 잰다.
- [ ] 단위 테스트 (`golbang-core` spec/chat) + 릴리스 빌드.
- [ ] 생산 유닛 재기동 후 스모크 `1+1=` 과 툴 히스토리(jinja 폴백 없음).

### 6.4 최종 A/B + 복구

- [ ] 같은 프롬프트로 llama-server와 한 바퀴 더 (열이 비슷할 때).
- [ ] `docs/bench/p7.md` 최종 표. 위키 `qwen38-on-golbang` append.
- [ ] DoD 미달이면 이슈를 닫지 말고 남은 격차와 다음 가설을 적는다.
- [ ] 8083이 비어 있지 않게 유닛을 남긴다 (§4).

---

## 7. 하지 않는 것

- GPU 클럭/전력 튜닝으로 숫자 맞추기.
- DSV4 / `golbang-deepseek` 를 이 이슈의 벤치 대상으로 삼기.
- 생산 `llama.cpp` 트리에 `git pull`.
- 프로파일 없이 ggml-hip 커널 추측 패치.
- 비전 파이프라인 확장 (이미 `--mmproj` 있음. decode 속도가 주제).
- P4(#27) 착수.

---

## 8. 참고

- local-llm 작업 **#364** (기준 숫자), 위키 `qwen3.8-27b-q6-service`, `gpu-settings`
- golbang 위키 `qwen38-on-golbang`, `llama-cpp-upgrade-notes`
- 코드: `golbang-core/src/speculative.rs`, `scheduler.rs` (`maybe_fill_drafts`,
  `verify_and_emit`), `model.rs` (`spec_draft` / `spec_process`),
  `deploy/golbang-qwen38.service`
- llama-server: `common/speculative.cpp`, `tools/server/server.cpp` 검증 루프
- 이전 공정 A/B 형식: `docs/bench/golbang-vs-llama-server.md`
