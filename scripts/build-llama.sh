#!/usr/bin/env bash
#
# scripts/build-llama.sh — llama.cpp 코어를 공개 소스에서 재현 빌드한다 (트랙 A).
#
# 게시된 base 커밋을 SHA로 받아 `patches/<tree>/`를 순서대로 적용하고 cmake로
# 빌드한다. 결과 트리는 `vendor/<tree>`에 놓이고, `golbang-sys/build.rs`의 핀
# 검사가 통과하도록 `.golbang-llama-pin` 마커를 남긴다.
#
# 사용 예:
#   scripts/build-llama.sh hip          # MI50(gfx906) HIP
#   scripts/build-llama.sh cuda         # V100(sm_70, CUDA 12.8)
#   scripts/build-llama.sh ds41         # DeepSeek-V4.1 네이티브 런타임 (MI50)
#   scripts/build-llama.sh vulkan
#   scripts/build-llama.sh cpu          # CPU 전용 (GPU 백엔드 없음, 컨테이너 배포용)
#
# 빌드 후:
#   GOLBANG_GPU=hip CARGO_TARGET_DIR=target-hip cargo build -p golbang-server --release
#
# `GOLBANG_LLAMA_DIR`을 지정하면 기본 `vendor/<tree>` 대신 그 경로를 쓴다.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_ROOT="$REPO_ROOT/patches"

die() { echo "error: $*" >&2; exit 1; }
info() { echo "[build-llama] $*"; }

usage() {
  sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

GPU=""
DIR=""
JOBS="$(nproc 2>/dev/null || echo 4)"
FORCE=0
DRY_RUN=0
NOBUILD=0

while [ $# -gt 0 ]; do
  case "$1" in
    hip|cuda|vulkan|ds41|ds41-cuda|cpu) GPU="$1"; shift ;;
    --dir) DIR="${2:?--dir needs a value}"; shift 2 ;;
    --jobs) JOBS="${2:?--jobs needs a value}"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --no-build) NOBUILD=1; shift ;;
    -h|--help) usage 0 ;;
    *) die "unknown argument: $1 (see --help)" ;;
  esac
done
[ -n "$GPU" ] || usage 1

GGML_URL="https://github.com/ggml-org/llama.cpp.git"
VCRUZ_URL="https://github.com/vcruz305/llama.cpp.git"
# 대상 타깃. 비우면 전체(`all`). CPU만 공유 라이브러리만 빌드한다 — 최신 트리의
# `llama` 앱 타깃이 `LLAMA_BUILD_EXAMPLES/SERVER=OFF`에서 impl 심볼을 못 찾아
# 링크에 실패하므로(golbang이 쓰지 않는 실행 파일), 필요한 라이브러리만 고른다.
TARGETS=()

# 트리/베이스/패치/빌드 디렉터리/최종 핀은 모두 build.rs와 일치해야 한다.
case "$GPU" in
  hip)
    TREE="llama.cpp-glm5next"; URL="$GGML_URL"
    BASE="f8dbcd61893702976f9ab03be89c2b9f436d532c"
    PATCHES=(hip/0001-glm5next-glm5next-feature-and-pr27754.patch
             hip/0002-gfx906-furnace-ports.patch
             hip/0003-dsv41-flash-loader-and-kv-guard.patch)
    BINDIR="build"
    CMAKE_FLAGS=(-DGGML_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx906)
    PIN="367ebbc20c2b20db411d5acf72b88d26a7c13d70"
    ;;
  vulkan)
    TREE="llama.cpp-glm5next"; URL="$GGML_URL"
    BASE="f8dbcd61893702976f9ab03be89c2b9f436d532c"
    PATCHES=(hip/0001-glm5next-glm5next-feature-and-pr27754.patch
             hip/0002-gfx906-furnace-ports.patch
             hip/0003-dsv41-flash-loader-and-kv-guard.patch)
    BINDIR="build-vulkan"
    CMAKE_FLAGS=(-DGGML_VULKAN=ON)
    PIN="367ebbc20c2b20db411d5acf72b88d26a7c13d70"
    ;;
  cuda)
    TREE="llama.cpp-cuda-upstream"; URL="$GGML_URL"
    BASE="c069aa7f5f2beeead1a3a8e9f71510f1b64d0725"
    PATCHES=(cuda/0001-qwen4exp-mtp-draft-head.patch)
    BINDIR="build"
    CMAKE_FLAGS=(-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70)
    PIN="1c4cfda6cc8b28d91eca48a30623a70253ca21fc"
    ;;
  cpu)
    TREE="llama.cpp-glm5next"; URL="$GGML_URL"
    BASE="f8dbcd61893702976f9ab03be89c2b9f436d532c"
    PATCHES=(hip/0001-glm5next-glm5next-feature-and-pr27754.patch
             hip/0002-gfx906-furnace-ports.patch
             hip/0003-dsv41-flash-loader-and-kv-guard.patch)
    BINDIR="build-cpu"
    # GPU 백엔드를 모두 끄고 호스트 CPU에 맞춰 빌드한다. dtc 컨테이너(amd64)에
    # 넣어 CPU 전용으로 서빙한다. AVX2/FMA/F16C는 네이티브 감지에 맡긴다.
    CMAKE_FLAGS=(-DGGML_NATIVE=ON)
    TARGETS=(llama ggml ggml-base ggml-cpu mtmd)
    PIN="367ebbc20c2b20db411d5acf72b88d26a7c13d70"
    ;;
  ds41|ds41-cuda)
    TREE="llama.cpp-ds41"; URL="$VCRUZ_URL"
    BASE="f37da57110ebbe07e982a934f2d444d9fd30eb09"
    PATCHES=(ds41/0001-gfx906-furnace-ports.patch)
    if [ "$GPU" = "ds41-cuda" ]; then
      BINDIR="build-cuda"
      CMAKE_FLAGS=(-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70)
      PIN="24032ea2b12cc0cc38dfa58099bc1ecb6890d6fc"
    else
      BINDIR="build"
      CMAKE_FLAGS=(-DGGML_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx906)
      PIN="24032ea2b12cc0cc38dfa58099bc1ecb6890d6fc"
    fi
    ;;
  *) die "unsupported GOLBANG_GPU: $GPU" ;;
esac

DIR="${DIR:-${GOLBANG_LLAMA_DIR:-$REPO_ROOT/vendor/$TREE}}"
mkdir -p "$(dirname "$DIR")"
PIN_FILE="$DIR/.golbang-llama-pin"

run() {
  if [ "$DRY_RUN" = 1 ]; then
    printf '+ %s\n' "$*"
  else
    "$@"
  fi
}

if [ "$DRY_RUN" = 1 ]; then
  info "GPU=$GPU tree=$TREE"
  info "dir=$DIR base=$BASE bindir=$DIR/$BINDIR"
  info "patches=${PATCHES[*]}"
  run rm -rf "$DIR"
  run git init -q "$DIR"
  run git -C "$DIR" remote add origin "$URL"
  run git -C "$DIR" fetch --depth 1 origin "$BASE"
  run git -C "$DIR" checkout --detach --force FETCH_HEAD
  run git -C "$DIR" reset --hard FETCH_HEAD
elif [ -f "$PIN_FILE" ] && [ "$(cat "$PIN_FILE")" = "$PIN" ] \
     && git -C "$DIR" rev-parse --git-dir >/dev/null 2>&1 && [ "$FORCE" != 1 ]; then
  info "이미 패치된 트리가 있습니다: $DIR ($PIN) — fetch/apply 생략"
elif [ -e "$DIR" ] && [ "$FORCE" != 1 ]; then
  die "$DIR 가 이미 존재하지만 핀 마커가 없습니다. --force로 다시 만들거나 지우세요."
fi

if [ "$DRY_RUN" != 1 ] && [ "$FORCE" = 1 ]; then
  info "기존 트리를 지웁니다: $DIR"
  rm -rf "$DIR"
fi

if [ "$DRY_RUN" != 1 ] && { [ ! -f "$PIN_FILE" ] || ! git -C "$DIR" rev-parse --git-dir >/dev/null 2>&1; }; then
  info "base 커밋을 받습니다: $TREE @ $BASE"
  git init -q "$DIR"
  if git -C "$DIR" remote get-url origin >/dev/null 2>&1; then
    git -C "$DIR" remote set-url origin "$URL"
  else
    git -C "$DIR" remote add origin "$URL"
  fi
  # GitHub는 도달 가능한 SHA를 직접 fetch할 수 있다. shallow가 막히면 full로 재시도.
  git -C "$DIR" fetch --depth 1 origin "$BASE" || git -C "$DIR" fetch origin "$BASE"
  git -C "$DIR" checkout --detach --force FETCH_HEAD
  git -C "$DIR" reset --hard FETCH_HEAD
  git -C "$DIR" clean -fd -e build -e build-vulkan -e build-cuda

  for p in "${PATCHES[@]}"; do
    pf="$PATCH_ROOT/$p"
    [ -f "$pf" ] || die "패치가 없습니다: $pf"
    info "apply $(basename "$p")"
    git -C "$DIR" apply --whitespace=nowarn "$pf" \
      || die "$p 적용 실패 (base=$BASE). patches/README.md 참조."
  done
  printf '%s\n' "$PIN" > "$PIN_FILE"
  info "패치 완료 → $PIN 마커 기록"
fi

if [ "$DRY_RUN" = 1 ]; then
  info "cmake -S $DIR -B $DIR/$BINDIR ${CMAKE_FLAGS[*]}"
  info "cmake --build $DIR/$BINDIR -j $JOBS"
  exit 0
fi

if [ "$NOBUILD" = 1 ]; then
  info "--no-build: fetch/apply 완료 (tree=$DIR, pin=$PIN)"
  exit 0
fi

CUDA_COMPILER_FLAG=()
if [ -x /opt/cuda-12.8/bin/nvcc ]; then
  CUDA_COMPILER_FLAG=(-DCMAKE_CUDA_COMPILER=/opt/cuda-12.8/bin/nvcc)
fi

info "cmake configure: $BINDIR (${CMAKE_FLAGS[*]})"
# `$ORIGIN` rpath로 빌드해 산출된 `.so`가 같은 디렉터리에서 서로를 찾게 한다
# (배포 tarball 재배치용 — scripts/package-release.sh 참조).
cmake -S "$DIR" -B "$DIR/$BINDIR" \
  "${CMAKE_FLAGS[@]}" "${CUDA_COMPILER_FLAG[@]}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  -DCMAKE_INSTALL_RPATH='$ORIGIN' \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF

info "cmake build: -j $JOBS"
if [ "${#TARGETS[@]}" -gt 0 ]; then
  cmake --build "$DIR/$BINDIR" --config Release -j "$JOBS" --target "${TARGETS[@]}"
else
  cmake --build "$DIR/$BINDIR" --config Release -j "$JOBS"
fi

info "완료. golbang 빌드:"
echo "  CARGO_TARGET_DIR=target-$GPU GOLBANG_GPU=$GPU \\"
echo "    GOLBANG_LLAMA_DIR=$DIR GOLBANG_LLAMA_BIN_DIR=$DIR/$BINDIR \\"
echo "    cargo build -p golbang-server --release"
