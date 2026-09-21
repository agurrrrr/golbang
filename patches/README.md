# llama.cpp 핀 패치 (provenance)

golbang은 llama.cpp를 커밋 SHA로 고정해 FFI로 링크한다. 공개 원격에 없는 로컬
변경분을 여기에 패치로 보관한다. `scripts/build-llama.sh`가 공개 base 커밋을
받아 아래 순서대로 적용한 뒤 cmake로 빌드한다. 각 패치가 재현하는 최종 트리는
`golbang-sys/build.rs`의 핀 SHA와 (빌드 산출물 기준으로) 일치한다.

| 트랙 | base (공개) | 패치 | 최종 핀 |
|------|-------------|------|---------|
| `hip`, `vulkan` | ggml-org/llama.cpp `f8dbcd618` | `hip/0001` + `hip/0002` + `hip/0003` | `367ebbc20` + `0003` |
| `ds41`, `ds41-cuda` | vcruz305/llama.cpp `runtime/deepseek41` `f37da5711` | `ds41/0001` | `24032ea2b` |
| `cuda` | ggml-org/llama.cpp `911f6cdc8` | `cuda/0001` + `cuda/0002` + `cuda/0003` + `cuda/0004` | `e7ec442b7` (base + #28243 + #28136 + #28770 + Volta opt-in) |

base 커밋은 모두 공개 GitHub에서 SHA로 직접 fetch할 수 있다.

## hip / vulkan — `llama.cpp-glm5next`

`f8dbcd61893702976f9ab03be89c2b9f436d532c` (ggml-org `origin/master` 스냅샷) 위에:

1. `0001-glm5next-glm5next-feature-and-pr27754.patch` — GLM-5.3-Flash(`glm5next`)
   아키텍처 + PR #27754. `unslothai/llama.cpp`의 `glm5next` 브랜치를 base에 merge한
   결과(`8eb1fe1652dd49b71826131d325c8ad02befe767`)와 동일한 트리.
2. `0002-gfx906-furnace-ports.patch` — gfx906(MI50) HIP 포트 3건.
   `sixvolts/llamacpp-gfx906-furnace`에서 이식:
   - DPP 기반 warp reduction (`32f284244`)
   - mmq I=64 (`2ffa92b3c`)
   - GCN weight repack (`repack-gcn.cu`, 버퍼 타입 등록)
   결과 트리 `367ebbc20c2b20db411d5acf72b88d26a7c13d70`.
3. `0003-dsv41-flash-loader-and-kv-guard.patch` — DeepSeek-V4.1-Flash 로더 패치
   (미모델링 sparse-attention/engram 텐서를 `GGML_OP_NONE`으로 등록) + 빈
   compressed KV cache의 `build_input_{k,v}_rot` SIGFPE 가드. golbang 자체 변경.
   위키 `dsv41-on-golbang` 참조.

`vulkan`은 같은 트리(`llama.cpp-glm5next`)를 `build-vulkan/`에서 `GGML_VULKAN=ON`으로
빌드한 것이라 별도 패치가 없다.

## ds41 / ds41-cuda — `llama.cpp-ds41`

vcruz305 `runtime/deepseek41`의 `f37da57110ebbe07e982a934f2d444d9fd30eb09` 위에:

1. `0001-gfx906-furnace-ports.patch` — hip `0002`와 동일 내용(다른 base용).
   결과 트리 `24032ea2b12cc0cc38dfa58099bc1ecb6890d6fc`.

`ds41-cuda`는 같은 트리를 `build-cuda/`에서 V100(sm_70, CUDA 12.8)로 빌드한 것이라
별도 패치가 없다.

## cuda — `llama.cpp-cuda-upstream`

`911f6cdc8ab8a530b2bee09ee61471a6f3178eeb`(ggml-org `origin/master`, 2026-09-18) 위에:

1. `0001-qwen4exp-mtp-draft-head.patch` — ggml-org PR #28243 `qwen4exp` NextN/MTP
   드래프트 헤드 + cross-model shared-tensor borrowing. Qwen3.8-Flash-Next의 외부
   MTP head(`-md` + `--spec-type draft-mtp`)용. 위키 `qwen38-mtp` 참조.
   결과 트리 `53b1389d0bf98fa367e2a0ce0475008e762ebf28`.

   이 패치는 PR #28243의 head `53b1389d0`(PR이 master를 `911f6cdc8a`로 merge하며
   "Fix merge conflicts")를 base `911f6cdc8a`와 비교한 순수 diff다. 즉 base를
   `c069aa7f5f` → `911f6cdc8a`로 올려 #28896(`rms_norm + mul` fusion)과
   #28901(`hc ops`)을 정식으로 편입했고, MTP 변경분은 PR이 이미 해소한 충돌 상태
   그대로다. `base + 0001`의 트리는 PR head 트리와 바이트 동일하다
   (`git diff base 53b1389d0` → `0001`).

   이전 핀 `1c4cfda6c`(= `c069aa7f5f` + #28243)은 롤백용으로 보존한다. 위키
   `qwen38-cuda-pin-rebase-28896-28901` 참조.

2. `0002-qwen4exp-lazy-direct-reads.patch` — ggml-org PR #28136 (`--lazy-mode
   on-direct`: PR head `c6a9e5c9a` = `90fde1f7f` qwen4exp direct PLE reads +
   `c6a9e5c9a` shared reader/gemma4). `0001` 위에 적용하면 최종 트리
   `775aa4edc8ec16cb0d1a4876c05ca851d361a99e`가 된다. lazy 테이블의 행을
   mmap demand fault 대신 명시적 `pread()`로 읽어 cold prefill의 scatter
   gather·readahead 낭비를 줄인다. `llama.h`는 1645 → 1646줄(새
   `LLAMA_LAZY_MODE_DIRECT` enum). golbang은 `--lazy-mode`로 이 모드를
   노출한다. 채택 여부와 측정은 `docs/bench/fn4-lazy-direct-ple.md` 참조.

   원 PR은 `67a17c17c`(2026-09-03) 기반이라 `911f6cdc8a`에 직접 얹으면
   `src/llama-model.{cpp,h}`, `src/models/{models.h,qwen4exp.cpp}`에서
   충돌한다. 위 커밋들을 현재 핀(base + `0001`)에 순서대로 cherry-pick해
   해소했고, 충돌은 `tools/llama-bench/llama-bench.cpp`의 `--lazy-mode`
   도움말 한 곳뿐이었다(HEAD의 최신 도움말을 유지하고 `on-direct`만 추가).

3. `0003-qwen4-sparse-fa.patch` — ggml-org PR #28770(`CUDA: enable sparse
   fa for qwen4`, 2026-09-20 master 머지). `0002` 위에 적용한다. qwen4exp의
   QSA 마스크에서 실제로 보이는 열만 모아(`ggml_cuda_flash_attn_ext_compact_mask`)
   `ncols1`개 질의 타일마다 인덱스 목록을 만들고, FA가 그 열만 훑는다. 새
   지원 형태는 `DKQ==256 && DV==256 && ncols2==8`에 `ncols1==1|8`이다.
   qwen4exp의 그래프도 sparse를 켜도록 바뀐다(`build_attn_qsa`의
   `build_attn_mha(..., top_k->ne[0], ...)`). `llama.h`는 그대로 1646줄이다.
   전체가 `ggml-cuda/fattn-{common,mma-f16}.cuh`·`fattn.cu`·`qwen4exp.cpp`·
   `tests/test-backend-ops.cpp`에 한정된다. 측정은 위키 `qwen38-sparse-fa-volta`,
   `docs/bench/fn6-sparse-fa.md` 참조.

4. `0004-volta-sparse-fa-optin.patch` — golbang 자체 변경. 위 #28770의 sparse
   게이트는 `turing_mma_available(cc)`를 요구해 V100(sm_70)에서는 절대 켜지지
   않는다. 또한 Volta 분기(`switch_ncols2`)는 gqa 비율의 약수로 `ncols2`를
   고르는데 qwen4exp의 gqa는 12라 `ncols2==8`이 선택되지 않는다. 이 패치는
   `GOLBANG_VOLTA_SPARSE_FA=1`일 때 `DKQ/DV==256 && ncols2==8` 형태에 한해
   (a) `shall_use_sparse`가 Volta를 허용하고 (b) prefill/검증 배치(`Q->ne[1]>4`,
   `K->ne[1] >= max(4096, 16*n_kv_max)`)에서 `ncols2=8`을 강제한다. `ncols1==1`
   형태는 Volta에서 컴파일 불가(`ncols1*ncols2=8 < 32`)라 그대로 둔다. 기본
   비활성이며 유닛이 환경변수로 켠다. 결과: 83k prefill **+5~8%**, decode 무변화,
   그리디 출력은 NUMERIC(다른 커널, 비트 동일 아님). 위키 `qwen38-sparse-fa-volta`.

## 검증 방법

```bash
# 각 base를 worktree로 띄우고 패치를 순서대로 적용한 뒤 핀 트리와 비교한다.
git -C <llama-repo> worktree add --detach /tmp/pc <base-sha>
cd /tmp/pc && for p in patches/<tree>/*.patch; do git apply "$p"; done
git add -A && git diff --cached --stat <final-pin>   # hip은 0003만, 나머지는 비어야 함
```

## 라이선스

llama.cpp는 MIT다. `0001`/`0002`의 원작(ggml-org, unsloth, sixvolts/llamacpp-gfx906-furnace)과
`0003`의 저작(golbang) 모두 MIT 조건에서 재배포 가능하다. 포트 출처는 위에 표기한다.
