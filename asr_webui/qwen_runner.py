from __future__ import annotations

import os
import platform
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

# transformers >=4.4x warns that TRANSFORMERS_CACHE is deprecated (will be removed in v5).
# If the container/environment still exports it, migrate to HF_HOME/HF_HUB_CACHE early
# (before importing qwen_asr/transformers) and delete TRANSFORMERS_CACHE to avoid warning spam.
def _migrate_hf_cache_env() -> None:
    tcache = os.environ.get("TRANSFORMERS_CACHE")
    if not tcache:
        return

    tcache_stripped = tcache.rstrip("/\\")
    base = os.path.basename(tcache_stripped).lower()
    hf_home = os.path.dirname(tcache_stripped) if base == "transformers" else tcache_stripped

    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
    os.environ.pop("TRANSFORMERS_CACHE", None)


_migrate_hf_cache_env()

# ASR 不需要 torchvision，但 transformers 在某些版本/环境下会自动导入 torchvision，
# 若 torchvision 与 torch 二进制不匹配会导致 `operator torchvision::nms does not exist` 直接崩溃。
# 这里默认禁用 torchvision 以提高在 Windows 等环境的鲁棒性。
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

from qwen_asr import Qwen3ASRModel  # noqa: E402


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


@dataclass(frozen=True)
class ASRConfig:
    asr_checkpoint: str
    aligner_checkpoint: str
    backend: Literal["transformers", "vllm"] = "transformers"
    device_map: str = "cuda:0"
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    max_inference_batch_size: int = 1
    max_new_tokens: int = 2048
    attn_implementation: str | None = None
    # advanced: pass-through kwargs (even if some are ignored by backend)
    asr_init_kwargs: dict[str, Any] | None = None
    aligner_init_kwargs: dict[str, Any] | None = None
    # vLLM specific
    vllm_kwargs: dict[str, Any] | None = None
    cuda_visible_devices: str | None = None


class _ModelCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: Any | None = None
        self._cfg: ASRConfig | None = None

    def get(self, cfg: ASRConfig):
        with self._lock:
            if self._model is not None and self._cfg == cfg:
                return self._model

            if cfg.cuda_visible_devices is not None and cfg.backend == "vllm":
                # vLLM device selection is via CUDA_VISIBLE_DEVICES (per official docs)
                os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.cuda_visible_devices)

            fa_kwargs: dict[str, Any] = dict(
                dtype=DTYPE_MAP[cfg.dtype],
                device_map=cfg.device_map,
            )
            if cfg.attn_implementation:
                fa_kwargs["attn_implementation"] = cfg.attn_implementation
            if cfg.aligner_init_kwargs:
                fa_kwargs.update(cfg.aligner_init_kwargs)

            if cfg.backend == "transformers":
                kwargs: dict[str, Any] = dict(
                    dtype=DTYPE_MAP[cfg.dtype],
                    device_map=cfg.device_map,
                    max_inference_batch_size=cfg.max_inference_batch_size,
                    max_new_tokens=cfg.max_new_tokens,
                    forced_aligner=cfg.aligner_checkpoint,
                    forced_aligner_kwargs=fa_kwargs,
                )
                if cfg.attn_implementation:
                    kwargs["attn_implementation"] = cfg.attn_implementation
                if cfg.asr_init_kwargs:
                    kwargs.update(cfg.asr_init_kwargs)
                self._model = Qwen3ASRModel.from_pretrained(cfg.asr_checkpoint, **kwargs)
            else:
                # vLLM backend (recommended for fastest inference per model card)
                if platform.system().lower().startswith("win"):
                    raise RuntimeError(
                        "vLLM 后端在 Windows 上通常不可用/不稳定（多数情况下缺少可用的 vLLM 轮子/依赖，且需要 Linux 上的编译与 CUDA 内核）。\n"
                        f"当前平台：{platform.platform()} | Python={platform.python_version()} | torch={torch.__version__}\n"
                        "建议：在 Linux/Docker（群晖）里启用 vLLM；Windows 本机请先用 transformers 后端。\n"
                        "如需查看更多细节：在 WebUI 勾选“显示详细错误（traceback）”。"
                    )

                kwargs2: dict[str, Any] = dict(
                    model=cfg.asr_checkpoint,
                    max_inference_batch_size=cfg.max_inference_batch_size,
                    max_new_tokens=cfg.max_new_tokens,
                    forced_aligner=cfg.aligner_checkpoint,
                    forced_aligner_kwargs=fa_kwargs,
                )
                if cfg.vllm_kwargs:
                    kwargs2.update(cfg.vllm_kwargs)
                if cfg.asr_init_kwargs:
                    # allow pass-through for qwen-asr wrapper (even if some keys are ignored)
                    kwargs2.update(cfg.asr_init_kwargs)

                # note: Qwen3ASRModel.LLM wraps vLLM init
                self._model = Qwen3ASRModel.LLM(**kwargs2)

            self._cfg = cfg
            return self._model

    def unload(self) -> None:
        """
        Best-effort unload to release GPU memory. Safe to call multiple times.
        """
        with self._lock:
            m = self._model
            self._model = None
            self._cfg = None

        # Try to gracefully shutdown vLLM engine if present (best effort).
        try:
            if m is not None:
                for attr in ("shutdown", "close", "stop"):
                    fn = getattr(m, attr, None)
                    if callable(fn):
                        try:
                            fn()
                        except Exception:
                            pass
        finally:
            try:
                del m
            except Exception:
                pass

        # Release CUDA memory if available.
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            pass


MODEL_CACHE = _ModelCache()


def unload_model() -> None:
    """
    Public API: unload cached model to release VRAM.
    """
    MODEL_CACHE.unload()


def _maybe_local_default(model_dir_name: str, hf_id: str) -> str:
    here = Path(__file__).resolve().parent.parent
    local = here / model_dir_name
    if local.exists():
        return str(local)
    # also check workspace root sibling (common when app.py at repo root)
    root = Path.cwd()
    local2 = root / model_dir_name
    if local2.exists():
        return str(local2)
    # docker/nas common mount point
    local3 = Path("/models") / model_dir_name
    if local3.exists():
        return str(local3)
    return hf_id


def _maybe_local_any(candidates: list[tuple[str, str]]) -> str:
    """
    Pick the first existing local model dir from candidates; otherwise return the first HF id.
    candidates: [(local_dir_name, hf_id), ...]
    """
    here = Path(__file__).resolve().parent.parent
    root = Path.cwd()
    for model_dir_name, _hf_id in candidates:
        try:
            local = here / model_dir_name
            if local.exists():
                return str(local)
        except Exception:
            pass
        try:
            local2 = root / model_dir_name
            if local2.exists():
                return str(local2)
        except Exception:
            pass
    return candidates[0][1]


def _prefer_models_dir(model_dir_name: str, hf_id: str) -> str:
    """
    For WebUI presets, prefer a local-looking path under a `models/` directory so the textbox
    shows something like `/models/Qwen3-ASR-0.6B` instead of `Qwen/...`.
    Falls back to HF id if no reasonable local models dir is present.
    """
    # 1) docker/nas typical mount
    try:
        m = Path("/models")
        if m.exists() and m.is_dir():
            return str(m / model_dir_name)
    except Exception:
        pass
    # 2) repo root ./models
    try:
        m2 = Path.cwd() / "models"
        if m2.exists() and m2.is_dir():
            return str(m2 / model_dir_name)
    except Exception:
        pass
    # 3) fallback to any detected local path, else HF
    return _maybe_local_default(model_dir_name, hf_id)


ASR_PRESETS: dict[str, str] = {
    "Qwen3-ASR-1.7B": _prefer_models_dir("Qwen3-ASR-1.7B", "Qwen/Qwen3-ASR-1.7B"),
    "Qwen3-ASR-0.6B": _prefer_models_dir("Qwen3-ASR-0.6B", "Qwen/Qwen3-ASR-0.6B"),
}

# Default: prefer local 1.7B, else local 0.6B, else HF 1.7B.
DEFAULT_ASR = _maybe_local_any(
    [
        ("Qwen3-ASR-1.7B", "Qwen/Qwen3-ASR-1.7B"),
        ("Qwen3-ASR-0.6B", "Qwen/Qwen3-ASR-0.6B"),
    ]
)
DEFAULT_ALIGNER = _maybe_local_default("Qwen3-ForcedAligner-0.6B", "Qwen/Qwen3-ForcedAligner-0.6B")

# allow overriding defaults by env (useful in docker)
DEFAULT_ASR = os.getenv("ASR_CHECKPOINT", DEFAULT_ASR)
DEFAULT_ALIGNER = os.getenv("ALIGNER_CHECKPOINT", DEFAULT_ALIGNER)


def transcribe_with_timestamps(
    audio: list[str] | str,
    *,
    cfg: ASRConfig,
    language: str | None = None,
    return_time_stamps: bool = True,
    transcribe_kwargs: dict[str, Any] | None = None,
) -> list[Any]:
    """
    Returns qwen-asr `results` list. Each item typically has:
    - .text
    - .language
    - .time_stamps (when enabled)
    """
    model = MODEL_CACHE.get(cfg)

    # qwen-asr can accept language per-audio list; we keep it simple here.
    kwargs: dict[str, Any] = dict(
        audio=audio,
        language=language,
        return_time_stamps=return_time_stamps,
    )
    if transcribe_kwargs:
        kwargs.update(transcribe_kwargs)

    results = model.transcribe(**kwargs)
    return results


def set_torch_threads(num_threads: int) -> None:
    # helpful on NAS: avoid oversubscribing CPU
    if num_threads and num_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)
        torch.set_num_threads(num_threads)

