#!/usr/bin/env bash
#
# scripts/package-release.sh — 백엔드별 바이너리 tarball을 만든다 (트랙 B).
#
# golbang-server + 매칭되는 llama.cpp/ggml `.so`를 한 디렉터리에 모으고,
# `$ORIGIN/lib` 기준으로 재배치 가능한 런타임을 만든다. patchelf가 있으면
# RUNPATH를 강제로 `$ORIGIN` 기준으로 다시 쓰고, 없으면 `run.sh` 래퍼가
# `LD_LIBRARY_PATH`로 보정한다.
#
# 사용 예:
#   scripts/package-release.sh              # 빌드된 모든 백엔드
#   scripts/package-release.sh hip cuda
#
# 먼저 `scripts/build-llama.sh <backend>`와 `cargo build`가 끝나 있어야 한다.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VERSION="$(grep -m1 '^version' Cargo.toml | sed -E 's/.*"([^"]+)".*/\1/')"
ARCH="$(uname -m)"
OUT="$REPO_ROOT/dist"

die() { echo "error: $*" >&2; exit 1; }
info() { echo "[package] $*"; }

backend_target() {
  case "$1" in
    hip)       echo "target-hip" ;;
    cuda)      echo "target-cuda" ;;
    vulkan)    echo "target-vulkan" ;;
    ds41)      echo "target-ds41" ;;
    ds41-cuda) echo "target-ds41-cuda" ;;
    cpu)       echo "target-cpu" ;;
    *) return 1 ;;
  esac
}

backend_tree() {
  case "$1" in
    hip)       echo "llama.cpp-glm5next" ;;
    cuda)      echo "llama.cpp-cuda-upstream" ;;
    vulkan)    echo "llama.cpp-glm5next" ;;
    ds41|ds41-cuda) echo "llama.cpp-ds41" ;;
    cpu)       echo "llama.cpp-glm5next" ;;
  esac
}

backend_bindir() {
  case "$1" in
    vulkan) echo "build-vulkan/bin" ;;
    ds41-cuda) echo "build-cuda/bin" ;;
    cpu) echo "build-cpu/bin" ;;
    *) echo "build/bin" ;;
  esac
}

BACKENDS=("$@")
if [ "${#BACKENDS[@]}" -eq 0 ]; then
  BACKENDS=(hip cuda vulkan ds41 ds41-cuda cpu)
fi

mkdir -p "$OUT"
STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
HAS_PATCHELF=0
if command -v patchelf >/dev/null 2>&1; then HAS_PATCHELF=1; fi

packed=0
for be in "${BACKENDS[@]}"; do
  target="$(backend_target "$be")" || die "unknown backend: $be"
  bindir="$REPO_ROOT/vendor/$(backend_tree "$be")/$(backend_bindir "$be")"
  if [ -n "${GOLBANG_LLAMA_BIN_DIR:-}" ]; then bindir="$GOLBANG_LLAMA_BIN_DIR"; fi

  bin="$REPO_ROOT/$target/release/golbang-server"
  if [ ! -x "$bin" ]; then
    info "SKIP $be — 빌드된 바이너리가 없습니다: $bin"
    continue
  fi
  if [ ! -d "$bindir" ]; then
    info "SKIP $be — llama.cpp bin 디렉터리가 없습니다: $bindir"
    continue
  fi

  name="golbang-$VERSION-$be-$ARCH"
  stage="$OUT/$name"
  rm -rf "$stage"
  mkdir -p "$stage/lib"

  cp -f "$bin" "$stage/golbang-server"
  cp -a "$bindir"/libllama.so*   "$stage/lib/" 2>/dev/null || true
  cp -a "$bindir"/libggml*.so*   "$stage/lib/" 2>/dev/null || true
  cp -a "$bindir"/libmtmd.so*    "$stage/lib/" 2>/dev/null || true
  [ -e "$stage/lib/libllama.so.0" ] || die "$be: libllama.so가 $bindir 에 없습니다"

  # 라이선스/문서/시크릿 예시
  cp -f LICENSE LICENSE-APACHE README.md "$stage/"
  if [ -f deploy/secrets.env.example ]; then cp -f deploy/secrets.env.example "$stage/"; fi

  # 재배치 가능한 RUNPATH. patchelf가 있으면 .so는 $ORIGIN, 바이너리는 $ORIGIN/lib.
  if [ "$HAS_PATCHELF" = 1 ]; then
    for so in "$stage"/lib/*.so*; do
      if [ -L "$so" ]; then continue; fi
      patchelf --set-rpath '$ORIGIN' "$so" 2>/dev/null || true
    done
    patchelf --set-rpath '$ORIGIN/lib' "$stage/golbang-server" 2>/dev/null || true
  else
    info "patchelf가 없어 RUNPATH를 다시 쓰지 못합니다. run.sh 래퍼로 보정합니다."
  fi

  # 래퍼: patchelf가 없거나 일부 .so의 RUNPATH가 절대경로여도 동작하게 한다.
  cat > "$stage/run.sh" <<'EOF'
#!/usr/bin/env bash
# 번들된 .so를 우선 사용하도록 LD_LIBRARY_PATH를 잡고 서버를 실행한다.
set -euo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$DIR/golbang-server" "$@"
EOF
  chmod +x "$stage/run.sh"

  pin="$(cat "$REPO_ROOT/vendor/$(backend_tree "$be")/.golbang-llama-pin" 2>/dev/null || echo unknown)"
  cat > "$stage/MANIFEST.txt" <<EOF
golbang $VERSION ($be/$ARCH)
built:   $STAMP
llama.cpp pin: $pin
binary:  $(basename "$bin")
run:     ./run.sh --model /path/to/model.gguf ...
EOF

  ( cd "$OUT" && tar czf "$name.tar.gz" "$name" )
  ( cd "$OUT" && sha256sum "$name.tar.gz" > "$name.tar.gz.sha256" )
  info "생성: dist/$name.tar.gz ($(du -h "$OUT/$name.tar.gz" | cut -f1))"
  packed=$((packed + 1))
done

[ "$packed" -gt 0 ] || die "패키징된 백엔드가 없습니다 (먼저 빌드하세요)"
info "완료: $packed 개"
