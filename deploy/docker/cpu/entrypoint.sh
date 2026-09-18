#!/bin/bash
#
# golbang-server CPU 전용 컨테이너 엔트리포인트.
#
# 대부분의 설정은 golbang-server가 직접 읽는 GOLBANG_* 환경변수로 전달된다
# (Dockerfile ENV 또는 dtc `--env`). 이 스크립트는 모델 파일 존재 확인과
# 없을 때의 내부 모델 서버 다운로드만 담당한다.
set -euo pipefail

MODEL="${GOLBANG_MODEL:-/models/Qwen3.8-4B-Q8_0.gguf}"
MODEL_NAME="$(basename "$MODEL")"
LOCAL_MODEL_URL="${LOCAL_MODEL_URL:-https://model-server.dropthe.codes}"

echo "=== golbang CPU server ==="
echo "Binary : $(/app/golbang-server --version 2>/dev/null || echo golbang-server)"
echo "Model  : $MODEL"
echo "Threads: ${GOLBANG_N_THREADS:-?}  ctx=${GOLBANG_N_CTX:-?}  parallel=${GOLBANG_N_PARALLEL:-?}"
echo "Spec   : ${GOLBANG_SPEC_TYPE:-none} (load_mtp=${GOLBANG_LOAD_MTP:-false})"
echo "Libraries:"
ls -la /app/lib/

# 모델은 dtc 스토리지 볼륨(/models)에 마운트되어 있어야 한다.
# 볼륨이 비어 있으면 내부 모델 서버에서 내려받는다.
if [ ! -f "$MODEL" ]; then
    echo "Model not found at $MODEL. Trying internal model server..."
    mkdir -p "$(dirname "$MODEL")" || true
    if wget --timeout=10 --tries=1 --progress=dot:mega -O "$MODEL.tmp" \
        "${LOCAL_MODEL_URL}/${MODEL_NAME}"; then
        mv "$MODEL.tmp" "$MODEL"
        echo "Downloaded from internal model server: $MODEL"
    else
        rm -f "$MODEL.tmp"
        echo "ERROR: model not available at $MODEL and internal model server download failed." >&2
        exit 1
    fi
else
    echo "Model found on mounted storage: $MODEL"
fi

exec /app/golbang-server
