# llama.cpp 핀 패치 (provenance)

golbang은 llama.cpp를 커밋 SHA로 고정해 FFI로 링크한다. 공개 원격에 없는 로컬
변경분을 여기에 패치로 보관한다. `scripts/build-llama.sh`가 공개 base 커밋을
받아 아래 순서대로 적용한 뒤 cmake로 빌드한다. 각 패치가 재현하는 최종 트리는
`golbang-sys/build.rs`의 핀 SHA와 (빌드 산출물 기준으로) 일치한다.

| 트랙 | base (공개) | 패치 | 최종 핀 |
|------|-------------|------|---------|
| `hip`, `vulkan` | ggml-org/llama.cpp `f8dbcd618` | `hip/0001` + `hip/0002` + `hip/0003` | `367ebbc20` + `0003` |
| `ds41`, `ds41-cuda` | vcruz305/llama.cpp `runtime/deepseek41` `f37da5711` | `ds41/0001` | `24032ea2b` |
| `cuda` | ggml-org/llama.cpp `c069aa7f5` (tag `b10924`) | `cuda/0001` | `1c4cfda6c` |

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

`c069aa7f5f2beeead1a3a8e9f71510f1b64d0725`(ggml-org, tag `b10924`) 위에:

1. `0001-qwen4exp-mtp-draft-head.patch` — ggml-org PR #28243 `qwen4exp` NextN/MTP
   드래프트 헤드 + cross-model shared-tensor borrowing. Qwen3.8-Flash-Next의 외부
   MTP head(`-md` + `--spec-type draft-mtp`)용. 위키 `qwen38-mtp` 참조.
   결과 트리 `1c4cfda6cc8b28d91eca48a30623a70253ca21fc`.

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
