#!/usr/bin/env bash
# Build the Q2_K AVX2 panel-kernel microbenchmark against the llama.cpp-ds41 tree.
#
# Usage:
#   ./build.sh            # uses the CPU libs in <tree>/build/bin
#
# Env overrides:
#   LLAMA_DS41_DIR   default /home/agurrrrr/code/local-llm/llama.cpp-ds41
#   BUILD            default <tree>/build  (override to point at another build)
#
# The tool links the ggml CPU libs and calls ggml_vec_dot_q2_K_q8_K as the
# generic baseline, so it does not modify the runtime tree.
set -euo pipefail

TREE="${LLAMA_DS41_DIR:-/home/agurrrrr/code/local-llm/llama.cpp-ds41}"
BUILD="${BUILD:-${TREE}/build}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -d "${BUILD}/bin" ]]; then
  echo "error: llama.cpp-ds41 build not found under ${BUILD}" >&2
  exit 1
fi

CXX="${CXX:-g++}"

"$CXX" -O3 -march=native -std=c++17 -DNDEBUG \
  -I"${TREE}/ggml/include" \
  -I"${TREE}/ggml/src" \
  -I"${TREE}/ggml/src/ggml-cpu" \
  -o "${HERE}/q2k-panel-bench" "${HERE}/q2k-panel-bench.cpp" \
  -L"${BUILD}/bin" -lggml-cpu -lggml -lggml-base \
  -Wl,-rpath,"${BUILD}/bin" \
  -lpthread -ldl

echo "built: ${HERE}/q2k-panel-bench"
