# P5 — 다턴 prefix cache: evict 후에도 KV 생존

> **이슈:** #28 · **선행:** P3 (#26) · **후행:** P6 (#29, 권장) · P4 (#27)보다 먼저
> **성격:** P3 축소안이 생산에서 비는 구멍. **GPU 커널 0줄.**
> 같은 대화의 다음 턴 TTFT를 전량 prefill(40초대)에서 suffix만큼으로 줄인다.

---

## 1. 목표

Open WebUI 같은 다턴 클라이언트가 히스토리를 통째로 다시 보내도,
**이미 계산한 prefix KV를 슬롯에 남겨** suffix만 prefill한다.

**완료 기준:** 같은 대화 2턴째부터 journal `cache_n > 0` 이고,
3k대 히스토리 TTFT가 40초대가 아니라 suffix prefill 시간으로 줄어든다.
커널·HIP·Rust GPU 변경은 없다.

---

## 2. 배경 / 근거

P3(#26)는 `llama_memory_seq_*` 스파이크 후 **전역 store를 포기**하고
slot-local `SlotPrefixCache` + suffix prefill만 넣었다. 단위 테스트는 통과한다.

생산 `golbang-deepseek` (`n_parallel=1`) 실측은 전부 `cache_n=0` (위키 `dsv4-run-notes`, 2026-08-13):

| req | prompt_n | prefill | cache_n | 총 |
|-----|----------|---------|---------|------|
| 3 | 3140 | 40.81s | 0 | 58.41s |
| 4 | 3314 | 42.88s | 0 | 53.42s |
| 5 | 3453 | 44.37s | 0 | 58.33s |

코드 원인:

1. `scheduler.rs` `finish_slot`이 Stop/Length에도 **항상** `engine.clear_seq`를 호출한다.
   슬롯 Rust 필드는 남아도 GPU KV는 매번 사라진다.
2. `SlotPrefixCache::reuse()`는 LCP>0일 때만 `tokens`를 쓴다.
   첫 bind의 캐시는 비어 있으므로 두 번째 요청도 LCP=0이다.
3. 생성 토큰 ID를 슬롯에 모으지 않는다. 다음 턴 프롬프트는 이전 assistant 답을
   포함하므로, prompt-only 캐시로는 공통 접두사가 짧아지거나 0이 된다.

작업 #300 우선순위 1. llama-server와 같은 `libggml-hip.so`를 쓰므로
여기서 커널을 만져도 TTFT 40초는 안 줄어든다.

P3를 실패로 되돌리지 않는다. 축소안 범위 밖의 생산 버그다.

---

## 3. 산출물

| 산출물 | 위치 | 설명 |
|--------|------|------|
| 슬롯 시퀀스 토큰 | `golbang-core/src/slot.rs` | prompt + 생성 토큰 ID를 캐시에 유지 |
| evict 정책 | `golbang-core/src/scheduler.rs` `finish_slot` | Stop/Length는 KV 유지, 실패만 clear |
| suffix rm | `golbang-core/src/model.rs` | `llama_memory_seq_rm(seq, p0, -1)` |
| 벤치 | `docs/bench/p5.md` | 다턴 `cache_n` / TTFT 전후 |

---

## 4. 단계별 작업

### 4.1 생성 토큰을 슬롯 캐시에 붙인다

- [ ] decode마다 샘플된 토큰 ID를 `ActiveJob` (또는 슬롯) 벡터에 누적한다.
      지금은 `pending` 한 개만 있다.
- [ ] Stop/Length로 끝날 때 `SlotPrefixCache.tokens = prompt_tokens + generated_ids`.
- [ ] 캐시 토큰 수와 `n_past`(또는 `llama_memory_seq_pos_max+1`)가 같아야 한다.

### 4.2 성공 evict는 KV를 지우지 않는다

- [ ] `finish_slot`: `Stop` / `Length` → `clear_seq` 호출 금지. 슬롯은 Empty로 회수하되 seq KV는 남긴다.
- [ ] `Cancelled` / `Timeout` / decode 실패 → 지금처럼 `clear_seq` + `prefix_cache.reset()`.
- [ ] 빈 슬롯을 다른 대화가 잡을 수 있으므로, bind 때 LCP로 살릴지 말지를 결정한다.

### 4.3 bind: LCP만큼 남기고 suffix KV만 제거

- [ ] `reuse_len = SlotPrefixCache::reuse(new_prompt)`.
- [ ] `reuse_len == 0` → `clear_seq`, `n_past = 0`, 전량 prefill (현행).
- [ ] `reuse_len > 0` → `llama_memory_seq_rm(seq, reuse_len, -1)` 로
      위치 `reuse_len` 이후만 제거. `prompt_offset = n_past = reuse_len`.
      이미 있는 `clear_seq`(p0=p1=-1)와 별도 API가 필요하다.
- [ ] `batch.rs` suffix prefill 경로를 그대로 쓴다. remaining = `len - (offset+pos)`.

### 4.4 범위 밖 (하지 말 것)

- 전역 `PrefixStore` / `llama_memory_seq_cp` 교차 슬롯 복사.
- `n_parallel > 1` 세션 고정(affinity). 생산은 `--n-parallel 1`.
- GPU 커널, rocprof, P4 Rust 커널.
- KV를 디스크에 저장하거나 프로세스 재시작 후에도 살리는 일.

### 4.5 검증

- [ ] 단위: 같은 슬롯에 공통 prefix 두 요청을 연속 bind하면 두 번째 `prompt_offset > 0`.
- [ ] 단위: 공통 prefix 없음 → `clear_seq` 경로, `prompt_offset == 0`.
- [ ] 단위: 생성 토큰이 캐시에 붙어 다음 LCP가 assistant 답을 포함한다.
- [ ] 생산 또는 동등 조건: DSV4 IQ2_M, 다턴 2회 이상.
      2턴째 `cache_n` ≈ 1턴 prompt+completion (템플릿 경계 수 토큰 오차 허용).
      TTFT가 suffix 길이 / 실측 prefill tok/s 근처.

---

## 5. 완료 기준 (Definition of Done)

- [ ] 같은 대화 2턴째 journal `cache_n > 0`
- [ ] 3k대 히스토리에서 TTFT가 전량 prefill(40초대)이 아니라 suffix만큼
- [ ] 실패/취소 요청은 여전히 KV를 지운다
- [ ] GPU 커널 반환 0 (이 이슈 diff에 HIP/C++/커널 파일 없음)
- [ ] `docs/bench/p5.md`에 전후 숫자
- [ ] 산출물 커밋

---

## 6. 리스크 & 완화

| 리스크 | 영향 | 완화 |
|--------|------|------|
| 채팅 템플릿이 히스토리를 다시 토크나이즈해 LCP가 짧아짐 | cache_n이 기대보다 작음 | 실제 토큰 ID로 LCP. 템플릿 바이트 비교 금지 |
| 유휴 슬롯 KV가 VRAM을 차지 | n_ctx=60k에서 여유 부족 | 생산 n_parallel=1, 이미 그 KV를 요청 중에 씀. 새 대화(LCP=0)만 clear |
| 위치와 캐시 토큰 수 불일치 | 잘못된 n_past, 쓰레기 로짓 | bind 때 `seq_pos_max+1`과 reuse_len을 assert/로그 |
| P3 축소안 주석과 혼동 | 구현자가 전역 store를 다시 켬 | §4.4. `PrefixStore` 기본 비활성 유지 |

---

## 7. 예상 공수

**0.5 ~ 1일.** 스케줄러·슬롯만. FFI 심볼은 P3에서 이미 바인딩됨.

---

## 8. 다음 단계

- P6(#29): rocprof로 커널을 찍은 뒤, 지목된 것만 HIP C++.
  prefix가 살아 있어야 트레이스가 3k 전량 prefill GEMM에 먹히지 않는다.
- P4(#27)는 P6 프로파일 이후.
