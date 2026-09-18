#!/usr/bin/env bash
#
# deploy/docker/cpu/build.sh — CPU 전용 golbang 이미지를 빌드한다.
#
# 1) `scripts/build-llama.sh cpu`로 llama.cpp CPU 전용 `.so`를 만든다.
# 2) `GOLBANG_GPU=cpu`로 golbang-server를 빌드한다.
# 3) 바이너리와 `.so`를 이 디렉터리로 스테이징한다 (.gitignore 대상).
# 4) `docker build`로 이미지를 만든다.
#
# 사용 예:
#   deploy/docker/cpu/build.sh                 # 이미지 golbang-cpu:latest
#   deploy/docker/cpu/build.sh my-reg/golbang-cpu:2026-09-18
#   SKIP_LLAMA=1 deploy/docker/cpu/build.sh    # llama.cpp가 이미 빌드된 경우
#
# 이후:
#   dtc image push <tag> && dtc deploy --image <tag> ...

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

TAG="${1:-golbang-cpu:latest}"
VENDOR_TREE="$REPO_ROOT/vendor/llama.cpp-glm5next"
BIN_DIR="$VENDOR_TREE/build-cpu/bin"
STAGE="$REPO_ROOT/deploy/docker/cpu"

info() { echo "[docker-cpu] $*"; }

if [ "${SKIP_LLAMA:-0}" != 1 ]; then
  info "llama.cpp CPU 빌드"
  scripts/build-llama.sh cpu --jobs "$(nproc)"
fi

info "golbang-server CPU 빌드"
# shellcheck disable=SC1091
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
GOLBANG_GPU=cpu CARGO_TARGET_DIR=target-cpu \
  cargo build -p golbang-server --release

[ -x "$REPO_ROOT/target-cpu/release/golbang-server" ] || {
  echo "error: target-cpu/release/golbang-server 없음" >&2; exit 1; }
[ -d "$BIN_DIR" ] || { echo "error: $BIN_DIR 없음" >&2; exit 1; }

info "스테이징: $STAGE"
rm -rf "$STAGE/lib"
mkdir -p "$STAGE/lib"
cp -f "$REPO_ROOT/target-cpu/release/golbang-server" "$STAGE/golbang-server"
for so in libllama.so* libggml.so* libggml-base.so* libggml-cpu.so* libmtmd.so*; do
  cp -a "$BIN_DIR"/$so "$STAGE/lib/" 2>/dev/null || true
done
[ -e "$STAGE/lib/libllama.so.0" ] || { echo "error: libllama.so가 스테이징되지 않았습니다" >&2; exit 1; }

info "docker build -t $TAG"
docker build -t "$TAG" "$STAGE"
info "완료: $TAG"
