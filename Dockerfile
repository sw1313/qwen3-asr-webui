#
# Qwen3-ASR WebUI Dockerfile (generic)
#
# 设计目标：
# - 默认使用 PyTorch CUDA runtime 基底（适合 Linux Docker + NVIDIA GPU）
# - 通过 build-arg / env 让镜像更可移植：端口、pip 源、缓存目录、可选 vLLM / FlashAttention
# - 兼容 Windows / Linux 宿主机：通过 volume 挂载模型/数据/输出/缓存目录
#
# 你可以在 build 时覆盖基础镜像（例如 CPU 或其它 CUDA 版本）：
#   docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime -t qwen3-asr-webui .
#   docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.4.1-cpu -t qwen3-asr-webui:cpu .
# 如需 VibeVoice-ASR（需要 vLLM + vibevoice 插件）：
#   docker build --build-arg INSTALL_VLLM=1 -t qwen3-asr-webui .
# 如需 FlashAttention 2（可能需要 nvcc），建议用 *devel* 基底：
#   docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel --build-arg INSTALL_FLASH_ATTN=1 -t qwen3-asr-webui:fa2 .
#
ARG BASE_IMAGE=pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

WORKDIR /app

# -------- build-time knobs --------
# 可选：使用国内 pip 镜像 / 企业私有源
ARG PIP_INDEX_URL=
ARG PIP_EXTRA_INDEX_URL=
ARG PIP_TRUSTED_HOST=

# 可选：安装 vLLM / FlashAttention 2
ARG INSTALL_VLLM=0
ARG INSTALL_VIBEVOICE=1
ARG VLLM_VERSION=0.14.0
ARG VIBEVOICE_PIP_SPEC=git+https://github.com/microsoft/VibeVoice.git
ARG VIBEVOICE_INSTALL_NO_DEPS=1
ARG INSTALL_FLASH_ATTN=0

# 可选：非 root 用户（Linux 挂载目录权限更友好；Windows 宿主通常无感）
ARG ENABLE_NONROOT=0
ARG UID=1000
ARG GID=1000
ARG RUN_USER=root

# -------- runtime defaults --------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Gradio/WebUI
    GRADIO_SERVER_NAME=0.0.0.0 \
    PORT=80 \
    # paths (recommended to mount)
    MODELS_DIR=/models \
    DATA_DIR=/data \
    OUTPUT_DIR=/output \
    WEBUI_CONFIG=/data/webui_config.json \
    # caches (recommended to mount)
    XDG_CACHE_HOME=/cache \
    HF_HOME=/cache/hf \
    HF_HUB_CACHE=/cache/hf/hub \
    TORCH_HOME=/cache/torch \
    # VibeVoice/vLLM runtime knobs (official docs)
    VIBEVOICE_FFMPEG_MAX_CONCURRENCY=64 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TRANSFORMERS_VERBOSITY=error \
    TRANSFORMERS_NO_TORCHVISION=1

# Install OS deps: ffmpeg for audio decode, git for pip install from VCS
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# Ensure mount points exist
RUN mkdir -p /models /data /output /cache

# Configure pip mirrors if provided
RUN if [ -n "$PIP_INDEX_URL" ]; then python -m pip config set global.index-url "$PIP_INDEX_URL"; fi && \
    if [ -n "$PIP_EXTRA_INDEX_URL" ]; then python -m pip config set global.extra-index-url "$PIP_EXTRA_INDEX_URL"; fi && \
    if [ -n "$PIP_TRUSTED_HOST" ]; then python -m pip config set global.trusted-host "$PIP_TRUSTED_HOST"; fi

# Install python deps
COPY app/requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -U pip && \
    python -m pip install --no-cache-dir -r /app/requirements.txt

# Optional: vLLM backend deps (Linux/Docker recommended; Windows generally not supported)
RUN if [ "$INSTALL_VLLM" = "1" ]; then \
      python -m pip install --no-cache-dir -U "qwen-asr[vllm]" "vllm==${VLLM_VERSION}"; \
    fi

# Optional: VibeVoice runtime/plugin for VibeVoice-ASR
# When INSTALL_VLLM=1, this is enabled by default (can disable via --build-arg INSTALL_VIBEVOICE=0).
# The VibeVoice pip package installs two top-level packages:
#   - vibevoice   (runtime library, also used for direct transformers inference with 4-bit models)
#   - vllm_plugin (vLLM integration; registers model via entry point)
RUN if [ "$INSTALL_VLLM" = "1" ] && [ "$INSTALL_VIBEVOICE" = "1" ]; then \
      if [ "$VIBEVOICE_INSTALL_NO_DEPS" = "1" ]; then \
        python -m pip install --no-cache-dir -U --no-deps "${VIBEVOICE_PIP_SPEC}" && \
        python -m pip install --no-cache-dir diffusers accelerate librosa pydub "bitsandbytes>=0.45.0"; \
      else \
        python -m pip install --no-cache-dir -U "${VIBEVOICE_PIP_SPEC}"; \
      fi && \
      python -m pip install --no-cache-dir -U "qwen-asr[vllm]" && \
      python -c "import vibevoice; print('vibevoice runtime OK')" && \
      python -c "import vllm_plugin; print('vllm_plugin OK:', vllm_plugin.__file__)" && \
      python -c "from vllm_plugin import register_vibevoice; register_vibevoice(); print('register_vibevoice() OK')" && \
      python -m pip show vibevoice; \
    fi

# Optional: FlashAttention 2 (may require build toolchain / nvcc depending on platform & wheels)
RUN if [ "$INSTALL_FLASH_ATTN" = "1" ]; then \
      apt-get update && apt-get install -y --no-install-recommends build-essential python3-dev git && rm -rf /var/lib/apt/lists/* && \
      python -m pip install --no-cache-dir -U flash-attn --no-build-isolation; \
    fi

COPY app/asr_webui /app/asr_webui
COPY app/app.py /app/app.py

# Optional: drop privileges (only if enabled)
RUN if [ "$ENABLE_NONROOT" = "1" ]; then \
      groupadd -g "$GID" app && useradd -m -u "$UID" -g "$GID" -s /bin/bash app && \
      chown -R "$UID:$GID" /app /data /output /cache; \
    fi
USER ${RUN_USER}

EXPOSE 80

CMD ["bash", "-lc", "python /app/app.py --host 0.0.0.0 --port ${PORT}"]
