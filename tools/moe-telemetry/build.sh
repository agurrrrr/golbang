#!/usr/bin/env bash
# Build the DSV4.1 MoE routing telemetry tool against the llama.cpp-ds41 tree.
#
# Usage:
#   ./build.sh [cuda|hip]        # default: cuda
#
# Env overrides:
#   LLAMA_DS41_DIR   default /home/agurrrrr/code/local-llm/llama.cpp-ds41
#
# The tool uses only public llama.h / ggml-backend.h API (cb_eval), so it does
# not modify the runtime tree. It links the matching backend build
# (build-cuda/ -> V100 sm_70, build/ -> gfx906 HIP).
set -euo pipefail

BACKEND="${1:-cuda}"
TREE="${LLAMA_DS41_DIR:-/home/agurrrrr/code/local-llm/llama.cpp-ds41}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$BACKEND" in
  cuda)
    BUILD="${TREE}/build-cuda"
    BACKEND_LIB="${BUILD}/bin/libggml-cuda.so.0.23.0"
    EXTRA_LINK=(-L/opt/cuda-12.8/targets/x86_64-linux/lib/stubs -lcuda)
    EXTRA_RPATH="/opt/cuda-12.8/lib64"
    ;;
  hip)
    BUILD="${TREE}/build"
    BACKEND_LIB="${BUILD}/bin/libggml-hip.so.0.23.0"
    EXTRA_LINK=(-L/opt/rocm/lib -lamdhip64)
    EXTRA_RPATH="/opt/rocm/lib"
    ;;
  *)
    echo "usage: $0 [cuda|hip]" >&2
    exit 1
    ;;
esac

BIN="${BUILD}/bin"
COMMON_BASE="${BUILD}/common/libllama-common-base.a"

if [[ ! -d "$BIN" || ! -f "$COMMON_BASE" ]]; then
  echo "error: llama.cpp-ds41 $BACKEND build not found under $BUILD" >&2
  exit 1
fi

CXX="${CXX:-c++}"

"$CXX" -O3 -DNDEBUG \
  -DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_USE_CPU -DLLAMA_SHARED \
  -I"${TREE}/common/." \
  -I"${TREE}/vendor/nlohmann/.." \
  -I"${TREE}/vendor/sheredom/.." \
  -I"${TREE}/include" \
  -I"${TREE}/ggml/include" \
  -o "${HERE}/moe-telemetry" "${HERE}/moe-telemetry.cpp" \
  -Wl,-rpath,"${BIN}:${EXTRA_RPATH}" \
  "${BIN}/libllama-common.so.0.4.0" \
  "${BIN}/libllama.so.0.4.0" \
  "${BIN}/libggml.so.0.23.0" \
  "${BIN}/libggml-cpu.so.0.23.0" \
  "${BACKEND_LIB}" \
  "${BIN}/libggml-base.so.0.23.0" \
  "${EXTRA_LINK[@]}" \
  "${COMMON_BASE}" \
  -lpthread -ldl

echo "built: ${HERE}/moe-telemetry ($BACKEND)"
