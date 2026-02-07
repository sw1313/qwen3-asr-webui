# Qwen3-ASR 批量字幕生成 WebUI（SRT / LRC）

基于 `qwen-asr` 的批量转写 WebUI，支持：
- **ASR**：`Qwen3-ASR-1.7B`
- **强制对齐/时间戳**：`Qwen3-ForcedAligner-0.6B`
- **输出**：SRT / LRC
- **可选**：VAD 预处理（去长静音、按语音段转写）、vLLM 后端、FlashAttention 2

模型说明（Hugging Face）：
- [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [Qwen/Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)

---

## 快速开始（推荐路径）

### Windows（推荐：本地 transformers 运行，不需要 Docker）

> 适用：Windows 本机有 NVIDIA 显卡，且你只用 **transformers** 后端（本项目默认后端）。
> vLLM 在 Windows 上通常不可用/不稳定（项目里也做了提示），因此 Windows 直接本地跑是最省心的。

#### 1) 安装方式（更通用：兼容不同 CUDA / PyTorch 版本）

> 说明：**PyTorch 的 pip 轮子通常自带 CUDA 运行时**，一般不依赖你本机安装的 CUDA Toolkit 版本。
> 所以“你装了什么 CUDA”不关键，关键看 `torch.version.cuda`（也就是你装的 torch 轮子是 `cu121/cu124/...` 还是 `cpu`）。

你可以二选一：

- **方案 A（最通用，推荐）**：先按你自己的环境安装 PyTorch（选择合适的 `cuXXX`），再安装本项目依赖。
  - 安装 PyTorch：按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 选择你的 CUDA 版本（例如 cu121 / cu124 / cu126 等）。
  - 安装本项目依赖（尽量不动你已安装的 torch）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 先装你选择的 torch/torchaudio（参考 PyTorch 官网给出的命令）
# 然后装本项目其余依赖：
pip install --upgrade-strategy only-if-needed -r requirements.txt
```

- **方案 B（一键固定版本）**：直接使用仓库提供的 `cu130` 组合（适合“不想选版本”的情况）。
  - 注意：这是固定到 `torch==2.10.0+cu130`，如果你本机/驱动环境不匹配，建议改用方案 A。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-win-cu130.txt
```

可选自检：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

#### 2) 准备模型

- **方式 A（推荐）**：把模型目录放在 `.\models\Qwen3-ASR-1.7B` 与 `.\models\Qwen3-ForcedAligner-0.6B`
- **方式 B**：不放本地，直接填 Hugging Face 模型 ID（首次需要联网下载）

#### 3) 启动 WebUI

```powershell
python app.py --host 0.0.0.0 --port 7860
```

打开：`http://127.0.0.1:7860`

---

### Linux / NAS（可选：Docker 更省环境折腾）

#### 前置条件

- **Docker**：Linux 安装 Docker Engine。
- **GPU（可选）**：NVIDIA 显卡 + 驱动 + NVIDIA Container Toolkit（需要 `--gpus all`）。
- **模型文件**：建议放在宿主机目录并挂载进容器（不要打包进镜像）。

### 1) 准备目录结构（建议）

在项目根目录创建/准备：

- `models/`：存放模型目录（例如 `models/Qwen3-ASR-1.7B`、`models/Qwen3-ForcedAligner-0.6B`）
- `data/`：持久化目录（WebUI 配置 `data/webui_config.json`；任务状态/日志 `data/job_state.json`、`data/job.log`）
- `output/`：字幕输出目录
- `cache/`：缓存（HF 模型缓存、torch.hub 的 VAD 缓存等，强烈推荐）

> 仓库里已自带 `models/` 示例（可能很大）。生产/长期使用建议你把模型放到独立路径并挂载。

### 2) 构建镜像

Linux：

```bash
docker build -t qwen3-asr-webui .
```

可选（更快/更省显存的高级能力）：

```bash
# 安装 vLLM（Linux/Docker 推荐更快；Windows 一般不可用）
docker build --build-arg INSTALL_VLLM=1 -t qwen3-asr-webui:vllm .

# 安装 FlashAttention 2（建议使用带 nvcc 的 devel 基底）
docker build \
  --build-arg BASE_IMAGE=pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel \
  --build-arg INSTALL_FLASH_ATTN=1 \
  -t qwen3-asr-webui:fa2 .

# 同时安装 vLLM + FlashAttention 2（推荐：devel 基底更稳）
docker build \
  --build-arg BASE_IMAGE=pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel \
  --build-arg INSTALL_VLLM=1 \
  --build-arg INSTALL_FLASH_ATTN=1 \
  -t qwen3-asr-webui:vllm-fa2 .
```

### 3) 运行（A：docker compose，最省心）

编辑 `docker-compose.yml`（按需改端口与挂载路径），然后：

```bash
docker compose up -d --build
```

打开：`http://127.0.0.1:7860`

> 如果你需要 GPU：在 `docker-compose.yml` 里取消注释 `gpus: all`（并确保宿主机 GPU 环境已正确配置）。

### 4) 运行（B：docker run）

#### Linux Bash（GPU 示例）

```bash
docker run --rm -it --gpus all -p 7860:80 \
  -v "$(pwd)/models:/models:ro" \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/output:/output" \
  -v "$(pwd)/cache:/cache" \
  -e ASR_CHECKPOINT=/models/Qwen3-ASR-1.7B \
  -e ALIGNER_CHECKPOINT=/models/Qwen3-ForcedAligner-0.6B \
  qwen3-asr-webui
```

#### 群晖 / NAS（挂载分离缓存目录示例）

> 适用：你希望把 HF 缓存与 torch.hub 缓存分别挂载到 NAS（并避免 `TRANSFORMERS_CACHE` FutureWarning）。

```bash
docker run --gpus all -d --name qwen3-asr-webui --restart unless-stopped -p 7863:80 --shm-size=4g \
  -v /volume1/docker/asr/app:/app \
  -v /volume1/docker/asr/models:/models:ro \
  -v /volume1/docker/asr/data:/data \
  -v /volume1/docker/asr/output:/output \
  -v /volume1/docker/asr/cache/hf:/hf_cache \
  -v /volume1/docker/asr/cache/torch:/torch_cache \
  -v /volume1/docker/asr/vad_repo:/vad_repo:ro \
  -e ASR_CHECKPOINT=/models/Qwen3-ASR-1.7B \
  -e ALIGNER_CHECKPOINT=/models/Qwen3-ForcedAligner-0.6B \
  -e OUTPUT_DIR=/output \
  -e HF_HOME=/hf_cache \
  -e TORCH_HOME=/torch_cache \
  -e VAD_REPO_DIR=/vad_repo \
  qwen3-asr-webui
```

#### CPU 运行（不推荐：很慢，但更通用）

```bash
docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.4.1-cpu -t qwen3-asr-webui:cpu .
docker run --rm -it -p 7860:80 \
  -v "$(pwd)/models:/models:ro" -v "$(pwd)/data:/data" -v "$(pwd)/output:/output" -v "$(pwd)/cache:/cache" \
  -e ASR_CHECKPOINT=/models/Qwen3-ASR-1.7B -e ALIGNER_CHECKPOINT=/models/Qwen3-ForcedAligner-0.6B \
  qwen3-asr-webui:cpu
```

---

## 配置参数（容器运行时环境变量）

### 必选/常用

- **`ASR_CHECKPOINT`**：ASR 模型路径或 HF ID  
  - 例：`/models/Qwen3-ASR-1.7B` 或 `Qwen/Qwen3-ASR-1.7B`
- **`ALIGNER_CHECKPOINT`**：强制对齐模型路径或 HF ID  
  - 例：`/models/Qwen3-ForcedAligner-0.6B`
- **`OUTPUT_DIR`**：输出目录（默认 `/output`，建议挂载）
- **`PORT`**：容器内 WebUI 监听端口（默认 `80`；通常不用改，改宿主映射即可）

### 缓存/离线（强烈推荐挂载 `./cache:/cache`）

- **`HF_HOME`**：HF 缓存根目录（默认 `/cache/hf`）
- **`HF_HUB_CACHE`**：HF Hub 缓存（默认 `/cache/hf/hub`）
- **`TORCH_HOME`**：torch.hub 缓存（默认 `/cache/torch`，VAD 会用到）
- **`XDG_CACHE_HOME`**：通用缓存目录（默认 `/cache`）
- （兼容旧环境）如果你还设置了 **`TRANSFORMERS_CACHE`**：程序会自动迁移到 `HF_HOME/HF_HUB_CACHE` 并移除该变量，避免 Transformers 的 FutureWarning。推荐直接使用 `HF_HOME`。

### WebUI 配置持久化

- **`WEBUI_CONFIG`** / **`WEBUI_CONFIG_PATH`**：WebUI 配置文件路径  
  - 默认：`/data/webui_config.json`（建议挂载 `./data:/data`）
  - 行为：WebUI 所有控件修改会自动保存（包括文本框打字），默认每秒最多写入一次，适合 NAS 挂载。

### 任务状态（断线续显）

- 任务状态文件：`/data/job_state.json`
- 任务日志文件：`/data/job.log`
- **取消任务**或**容器重启后启动**：会按你的偏好自动清空 job 状态与日志。

### VAD（离线可用）

- **`VAD_REPO_DIR`**：本地 `silero-vad` 仓库目录（用于完全离线）  
  - 配合挂载：`-v /path/to/silero-vad:/vad_repo:ro` + `-e VAD_REPO_DIR=/vad_repo`

### vLLM 相关

- **`VLLM_MAX_MODEL_LEN`**：未显式指定时，默认给 vLLM 的 `max_model_len` 上限（默认 `16384`，防止 16GB 级显卡 KV cache 启动失败）

### 其它

- **`TRANSFORMERS_NO_TORCHVISION`**：默认 `1`（避免某些环境下 torchvision 二进制不匹配导致崩溃）
- **`ASR_PROGRESS_RTF`**：进度条“平滑预估”的实时因子（RTF，默认 `0.18`）。值越小进度走得越快；不影响真实推理速度。
- **`WEBUI_DONE_FLASH_S`**：单个文件完成后强制闪到 100% 的窗口（默认 `2.0` 秒，避免轮询错过 100%）。

---

## 构建参数（Docker build 的 build-arg）

- **`BASE_IMAGE`**：基础镜像（默认 `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime`）
- **`INSTALL_VLLM`**：是否安装 vLLM（`0/1`）
- **`INSTALL_FLASH_ATTN`**：是否安装 FlashAttention 2（`0/1`）
- **`PIP_INDEX_URL`** / **`PIP_EXTRA_INDEX_URL`** / **`PIP_TRUSTED_HOST`**：自定义 pip 源
- **`ENABLE_NONROOT`**：是否创建并切换到非 root 用户（`0/1`）
- **`UID`** / **`GID`**：非 root 用户 UID/GID（Linux 挂载写权限用）
- **`RUN_USER`**：最终运行用户（默认 `root`；启用非 root 时可设为 `app`）

示例（Linux 上避免输出文件权限是 root）：

```bash
docker build \
  --build-arg ENABLE_NONROOT=1 \
  --build-arg UID="$(id -u)" \
  --build-arg GID="$(id -g)" \
  --build-arg RUN_USER=app \
  -t qwen3-asr-webui:nonroot .
```

---

## 本地运行（不使用 Docker）

### Linux（有 GPU）

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py --host 0.0.0.0 --port 7860
```

### Windows

见上面的「Windows（推荐：本地 transformers 运行，不需要 Docker）」。

---

## 使用教程（WebUI 里怎么配）

### 最常用的流程

1) **输入**：选“目录扫描”，把 `input_dir` 指到你挂载/本地的音频目录（Docker 默认 `/data` 只是配置目录，不一定放音频）  
2) **输出**：默认输出到 `OUTPUT_DIR`（Docker 默认 `/output`，已挂载则会落盘到宿主机 `output/`）  
3) **模型**：默认会读环境变量 `ASR_CHECKPOINT / ALIGNER_CHECKPOINT`，通常不用再改  
4) 点击“开始批量生成”

### 任务执行模式（重要）

- **任务是 detached 的**：点击开始后，即使你关闭浏览器页面，后台也会继续处理直到完成/取消。
- **重新打开 WebUI 会自动续显**：进度条/日志通过轮询 `job_state.json/job.log` 恢复显示，无需保持原页面连接。
- **取消**：会立即停止当前与剩余任务，并尽力释放显存（vLLM 会终止整个进程组以回收 EngineCore）。

### VAD（可选）

- **适合**：长录音、静音多、批量处理时想更快/更稳
- **在线模式**：VAD 来源选 `auto/hub`（首次需要联网下载一次，后续走缓存）
- **离线模式**：
  - 把 `silero-vad` 仓库放到宿主机某目录
  - Docker 挂载到 `/vad_repo`，并设置 `VAD_REPO_DIR=/vad_repo`
  - WebUI 里把来源选 `local`

（可提前预热 hub 缓存）

- 直接启动一次 WebUI 并跑一个短文件即可触发下载，缓存会写到 `TORCH_HOME`（建议挂载持久化）。

### vLLM（可选）

- **更快**，但一般只建议在 **Linux/Docker** 用  
- 构建镜像时加：`--build-arg INSTALL_VLLM=1`
- WebUI 里后端选择：`vLLM（官方推荐更快）`
- 若显存不够：降低 `gpu_memory_utilization` 或设置 `VLLM_MAX_MODEL_LEN`

### 视频/无声/空字幕的鲁棒处理

- **支持视频文件**：会用 `ffmpeg` 自动提取音频后再做 ASR。
- **无音轨/无声/无有效文字**：不会报错，会输出**空字幕文件**（SRT/LRC），确保批处理不中断。

### 文件覆盖与目录时间

- **同名不同后缀**（例如 `a.mp3`、`a.flac`）都会生成 `a.srt`：当你勾选“覆盖已存在文件”时会正确覆盖。
- **原目录 mtime**：字幕写入采用原子替换，通常会更新字幕所在目录的 mtime。
- **专辑根目录 mtime（可选）**：WebUI 提供“更新专辑根目录修改时间”开关，可让输入目录的**一级子目录**在首次生成输出后被 touch 一次（适合按专辑排序的 NAS 场景）。

### FlashAttention 2（可选）

- WebUI 中 `attn_implementation` 填 `flash_attention_2`
- 并确保 `dtype` 为 `fp16` 或 `bf16`
- 失败多见于编译链 / CUDA / nvcc：优先在 Linux/Docker（devel 基底）里启用

---

## 常见问题（FAQ）

### 1) `Torch not compiled with CUDA enabled`

- **原因**：装了 CPU 版 torch
- **解决**：
  - Windows 本机：优先按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 安装与你环境匹配的 GPU 版 torch（`torch.version.cuda` 不应为空），然后再装 `requirements.txt`
  - 或者：用 `requirements-win-cu130.txt`（固定 cu130，一键但不一定匹配所有环境）
  - Docker：使用 CUDA 基底镜像 + `--gpus all`

### 2) Docker 里看不到 GPU

- **Linux**：确保已安装 NVIDIA Container Toolkit，并使用 `docker run --gpus all ...`
- **Windows**：确保 Docker Desktop 使用 WSL2，并开启 GPU 支持；命令同样是 `--gpus all`

### 3) vLLM 在 Windows 上报错

这是预期行为：vLLM 多数情况下需要 Linux/CUDA 环境。本项目在 Windows 上默认建议用 transformers 后端。

### 4) VAD 报离线/下载失败

- 挂载 `./cache:/cache` 确保持久缓存
- 或准备本地 `silero-vad` 仓库并设置 `VAD_REPO_DIR`

