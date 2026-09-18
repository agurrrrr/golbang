# golbang-server CPU-only container (dtc / DropTheCodes 배포용).
#
# `GOLBANG_GPU=cpu`로 빌드한 golbang-server와 같은 트리의 CPU 전용
# llama.cpp `.so`를 담는다. GPU 백엔드(HIP/CUDA/Vulkan)는 포함하지 않으며
# ggml-cpu가 정적으로 등록되므로 별도 로더 경로가 없어도 동작한다.
#
# 스테이징 레이아웃 (deploy/docker/cpu/ 를 build context로 사용):
#   golbang-server      target-cpu/release/golbang-server
#   lib/libllama.so*    vendor/llama.cpp-glm5next/build-cpu/bin/
#   lib/libggml*.so*
#   lib/libmtmd.so*
#   entrypoint.sh
#
# 베이스는 빌드 호스트(CachyOS/Arch)와 glibc/libstdc++가 맞는 Arch로 둔다.
# (ik_llama 배포와 동일한 이유: 최신 GLIBCXX 심볼 요구)
FROM archlinux:latest

RUN pacman -Syu --noconfirm --needed \
        glibc \
        gcc-libs \
        libgomp \
        ca-certificates \
        curl \
        wget \
        file \
    && pacman -Scc --noconfirm

WORKDIR /app

COPY golbang-server /app/golbang-server
COPY lib/ /app/lib/
COPY entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/golbang-server /app/entrypoint.sh

ENV LD_LIBRARY_PATH=/app/lib

# === 모델/서버 기본값: Qwen3.8-4B-Distill Q8_0 (qwen35 hybrid SSM) ===
# 실제 값은 dtc `--env`로 덮어쓴다. 아래는 컨테이너 단독 실행용 기본값이다.
# 주의: KV 풀 = GOLBANG_N_CTX * GOLBANG_N_PARALLEL 이다(kv-unified). 아래
# 기본값은 풀 32768(32768 x 1)이고, q8_0 KV 1GiB + 모델 4.3GiB + MTP 버퍼를
# 합쳐 약 6.5GiB를 쓴다. `dtc deploy --memory 16384` 이상을 권장한다.
#
# 스레드(2026-09-18 실측, Xeon E5-2650 v4 12C/24T 노드): 20이 최적이다.
# 16 대비 prefill +21%(56→68 t/s), decode +13%(18.8→21.2 t/s). 24(노드 전체 쿼터)
# 는 cgroup 쿼터를 다 써서 오히려 11 t/s로 붕괴하므로 `--cpu 20`으로 4 vCPU를
# 남겨 둔다.
ENV GOLBANG_MODEL=/models/Qwen3.8-4B-Q8_0.gguf
ENV GOLBANG_ALIAS=qwen38-4b-cpu
ENV GOLBANG_HOST=0.0.0.0
ENV GOLBANG_PORT=8080
ENV GOLBANG_N_GPU_LAYERS=0
ENV GOLBANG_N_CTX=32768
ENV GOLBANG_N_PARALLEL=1
ENV GOLBANG_QUEUE_SIZE=2
ENV GOLBANG_KV_UNIFIED=true
ENV GOLBANG_SINGLE_MAX_CTX=32768
ENV GOLBANG_N_THREADS=20
ENV GOLBANG_N_BATCH=2048
ENV GOLBANG_N_UBATCH=512
ENV GOLBANG_JINJA=true
# 임베디드 MTP(nextn_predict_layers=1) 추측 디코딩: 구조적/반복 생성에서 약 3x.
# 자유 서술은 초안 수락률이 낮아(0.2~0.6) 이득이 작고, p_min으로 조기 중단해야
# 초안 낭비가 줄어든다. 아래 p_min=0.5는 자유 서술 draft_ms 비중을 17%→10%로
# 낮추면서 구조적 작업(수락률 1.0, mean len 3.97)은 그대로 유지한다.
ENV GOLBANG_LOAD_MTP=true
ENV GOLBANG_SPEC_TYPE=draft-mtp
ENV GOLBANG_SPEC_DRAFT_N_MAX=3
ENV GOLBANG_SPEC_DRAFT_P_MIN=0.5

# 권장 샘플링 기본값 (empero-ai/Qwen3.8-4B-Distill 모델 카드: temp 0.6 / top_p 0.95 / top_k 20)
ENV GOLBANG_TEMPERATURE=0.6
ENV GOLBANG_TOP_P=0.95
ENV GOLBANG_TOP_K=20
ENV GOLBANG_MAX_TOKENS=0

VOLUME ["/models"]
EXPOSE 8080

ENTRYPOINT ["/app/entrypoint.sh"]
