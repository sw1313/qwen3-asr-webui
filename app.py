from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Literal

import multiprocessing as mp
from dataclasses import asdict
import html
# transformers >=4.4x warns that TRANSFORMERS_CACHE is deprecated (will be removed in v5).
# Many docker images / older configs still export it. To keep backward compatibility and
# also keep cache on a mounted NAS path, migrate it to HF_HOME/HF_HUB_CACHE and delete
# TRANSFORMERS_CACHE *before* any transformers-related import happens.
def _migrate_hf_cache_env() -> None:
    tcache = os.environ.get("TRANSFORMERS_CACHE")
    if not tcache:
        return

    # Heuristic: old TRANSFORMERS_CACHE often ends with ".../transformers".
    tcache_stripped = tcache.rstrip("/\\")
    base = os.path.basename(tcache_stripped).lower()
    hf_home = os.path.dirname(tcache_stripped) if base == "transformers" else tcache_stripped

    # Only set new vars if user didn't already set them explicitly.
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))

    # Remove deprecated var to avoid FutureWarning spam.
    os.environ.pop("TRANSFORMERS_CACHE", None)


_migrate_hf_cache_env()

# 同上：默认禁用 transformers 对 torchvision 的自动依赖，避免 Windows 下 torchvision::nms 崩溃
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

import gradio as gr

from asr_webui.batch_generate import CaptionConfig, generate_for_one_audio, write_output
from asr_webui.cancel_token import CancelToken
from asr_webui.config_store import default_webui_config_path, load_json, merge_update
from asr_webui.file_utils import MEDIA_EXTS_DEFAULT, VIDEO_EXTS_DEFAULT, list_audio_files
from asr_webui.file_utils import resolve_output_dir
from asr_webui.job_state import append_log as js_append_log
from asr_webui.job_state import clear_job as js_clear_job
from asr_webui.job_state import read_log_tail as js_read_log_tail
from asr_webui.job_state import read_state as js_read_state
from asr_webui.job_state import reset_job as js_reset_job
from asr_webui.job_state import update_state as js_update_state
from asr_webui.qwen_runner import ASRConfig, DEFAULT_ALIGNER, DEFAULT_ASR, set_torch_threads, unload_model
from asr_webui.vad import VadConfig, parse_json_dict as parse_vad_json_dict
from asr_webui.vllm_subprocess_batch_worker import vllm_worker_batch
from asr_webui.detached_runner import start_job as detached_start_job
from asr_webui.detached_runner import cancel_job as detached_cancel_job
from asr_webui.detached_runner import is_running as detached_is_running
import torch


LANG_PRESETS = [
    "(自动)",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "French",
    "German",
    "Italian",
    "Spanish",
    "Portuguese",
    "Russian",
    "Cantonese",
]


def _normalize_language(lang: str) -> str | None:
    lang = (lang or "").strip()
    if not lang or lang == "(自动)":
        return None
    return lang


def _normalize_uploads(uploaded: list[Any] | None) -> list[Path]:
    if not uploaded:
        return []
    out: list[Path] = []
    for x in uploaded:
        # gradio may return str path or dict-like objects with name/path
        if isinstance(x, str):
            out.append(Path(x))
        elif hasattr(x, "path") and isinstance(getattr(x, "path"), str):
            out.append(Path(getattr(x, "path")))
        elif hasattr(x, "name") and isinstance(getattr(x, "name"), str):
            out.append(Path(getattr(x, "name")))
        elif isinstance(x, dict) and "name" in x:
            out.append(Path(str(x["name"])))
        elif isinstance(x, dict) and "path" in x:
            out.append(Path(str(x["path"])))
        else:
            raise TypeError(f"无法识别上传文件对象: {type(x)}")
    return out


def run_batch(
    input_mode: Literal["dir", "upload"],
    input_dir: str,
    recursive: bool,
    exts_csv: str,
    uploads: list[Any] | None,
    output_format: Literal["srt", "lrc"],
    output_dir_mode: Literal["output", "same", "custom"],
    custom_output_dir: str,
    overwrite: bool,
    # model
    backend: str,
    asr_checkpoint: str,
    aligner_checkpoint: str,
    device_map: str,
    dtype: Literal["bf16", "fp16", "fp32"],
    max_inference_batch_size: int,
    max_new_tokens: int,
    attn_impl: str,
    asr_init_kwargs_json: str,
    aligner_init_kwargs_json: str,
    transcribe_kwargs_json: str,
    quiet_transformers: bool,
    # vllm
    vllm_gpu_memory_utilization: float,
    vllm_cuda_visible_devices: str,
    vllm_kwargs_json: str,
    debug_traceback: bool,
    # vad
    vad_enabled: bool,
    vad_backend: str,
    vad_source: str,
    vad_local_repo_dir: str,
    vad_target_sr: int,
    vad_threshold: float,
    vad_min_speech_ms: int,
    vad_max_speech_s: float,
    vad_min_silence_ms: int,
    vad_speech_pad_ms: int,
    vad_window_size_samples: int,
    vad_merge_gap_ms: int,
    vad_min_segment_ms: int,
    vad_kwargs_json: str,
    # caption
    max_chars_per_line: int,
    gap_break_s: float,
    break_on_punct: bool,
    # other
    language: str,
    torch_threads: int,
) -> tuple[str, list[str]]:
    set_torch_threads(torch_threads)

    if quiet_transformers:
        try:
            from transformers.utils import logging as hf_logging  # type: ignore

            hf_logging.set_verbosity_error()
            os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        except Exception:
            pass

    def parse_json_dict(name: str, s: str) -> dict:
        s = (s or "").strip()
        if not s:
            return {}
        try:
            obj = json.loads(s)
        except Exception as e:
            raise ValueError(f"{name} 不是合法 JSON：{e}")
        if not isinstance(obj, dict):
            raise ValueError(f"{name} 必须是 JSON 对象(dict)，但拿到的是 {type(obj)}")
        return obj

    if input_mode == "dir":
        exts = [x.strip() for x in (exts_csv or "").split(",") if x.strip()]
        files = list_audio_files(input_dir, recursive=recursive, exts=exts or MEDIA_EXTS_DEFAULT)
    else:
        files = _normalize_uploads(uploads)

    if not files:
        return "未找到音视频文件。请检查输入目录/后缀，或上传文件。\n", []

    backend_n = (backend or "transformers").strip().lower()
    requested_device = (device_map or "").strip().lower()
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        return (
            "检测到你选择了 CUDA（device_map 以 cuda 开头），但当前 Python 环境的 torch 不支持 CUDA。\n"
            "你现在的环境通常是安装了 CPU 版 torch（例如 torch==...+cpu），会导致报错：Torch not compiled with CUDA enabled。\n\n"
            "解决办法：\n"
            "1) 安装 CUDA 版 PyTorch（推荐新建 Python 3.12 环境再装）；或\n"
            "2) 临时把 device_map 改成 cpu（会非常慢）。\n",
            [],
        )

    try:
        asr_init_kwargs = parse_json_dict("ASR init kwargs", asr_init_kwargs_json)
        aligner_init_kwargs = parse_json_dict("Aligner init kwargs", aligner_init_kwargs_json)
        transcribe_kwargs = parse_json_dict("transcribe kwargs", transcribe_kwargs_json)
        vad_kwargs = parse_vad_json_dict("VAD kwargs", vad_kwargs_json)
        vllm_kwargs = parse_json_dict("vLLM kwargs", vllm_kwargs_json)
    except Exception as e:
        return f"参数解析失败：{e}\n", []

    # vLLM: the model advertises max seq len 65536, which often exceeds KV cache capacity
    # on 16GB-class GPUs. If user didn't specify, cap it to a safer default.
    if backend_n == "vllm":
        if "max_model_len" not in vllm_kwargs and "max_seq_len" not in vllm_kwargs:
            # Allow overriding via env for docker users.
            try:
                vllm_kwargs["max_model_len"] = int(os.getenv("VLLM_MAX_MODEL_LEN", "16384"))
            except Exception:
                vllm_kwargs["max_model_len"] = 16384

    asr_cfg = ASRConfig(
        backend=("vllm" if backend_n == "vllm" else "transformers"),
        asr_checkpoint=asr_checkpoint.strip(),
        aligner_checkpoint=aligner_checkpoint.strip(),
        device_map=device_map.strip() or "cuda:0",
        dtype=dtype,
        max_inference_batch_size=int(max_inference_batch_size),
        max_new_tokens=int(max_new_tokens),
        attn_implementation=(attn_impl.strip() or None),
        asr_init_kwargs=asr_init_kwargs or None,
        aligner_init_kwargs=aligner_init_kwargs or None,
        vllm_kwargs=(
            {
                **({"gpu_memory_utilization": float(vllm_gpu_memory_utilization)} if backend_n == "vllm" else {}),
                **(vllm_kwargs or {}),
            }
            or None
        ),
        cuda_visible_devices=(vllm_cuda_visible_devices.strip() or None),
    )
    cap_cfg = CaptionConfig(
        output_format=output_format,
        max_chars_per_line=int(max_chars_per_line),
        gap_break_s=float(gap_break_s),
        break_on_punct=bool(break_on_punct),
    )

    vad_cfg = VadConfig(
        enabled=bool(vad_enabled),
        backend="silero" if (vad_backend or "silero") == "silero" else "silero",
        source=(vad_source or "auto"),  # auto/local/hub
        local_repo_dir=(vad_local_repo_dir.strip() or None),
        target_sr=int(vad_target_sr),
        threshold=float(vad_threshold),
        min_speech_duration_ms=int(vad_min_speech_ms),
        max_speech_duration_s=float(vad_max_speech_s),
        min_silence_duration_ms=int(vad_min_silence_ms),
        speech_pad_ms=int(vad_speech_pad_ms),
        window_size_samples=int(vad_window_size_samples),
        merge_gap_ms=int(vad_merge_gap_ms),
        min_segment_ms=int(vad_min_segment_ms),
        vad_kwargs=vad_kwargs or None,
    )

    log, out_paths = "", []
    # keep old signature compatibility (non-streaming path)
    # actual streaming is implemented in `run_batch_stream`.
    return log, [str(p) for p in out_paths]


def run_batch_stream(
    input_mode: Literal["dir", "upload"],
    input_dir: str,
    recursive: bool,
    exts_csv: str,
    uploads: list[Any] | None,
    output_format: Literal["srt", "lrc"],
    output_dir_mode: Literal["output", "same", "custom"],
    custom_output_dir: str,
    overwrite: bool,
    # model
    backend: str,
    asr_checkpoint: str,
    aligner_checkpoint: str,
    device_map: str,
    dtype: Literal["bf16", "fp16", "fp32"],
    max_inference_batch_size: int,
    max_new_tokens: int,
    attn_impl: str,
    asr_init_kwargs_json: str,
    aligner_init_kwargs_json: str,
    transcribe_kwargs_json: str,
    quiet_transformers: bool,
    # vllm
    vllm_gpu_memory_utilization: float,
    vllm_cuda_visible_devices: str,
    vllm_kwargs_json: str,
    debug_traceback: bool,
    # vad
    vad_enabled: bool,
    vad_backend: str,
    vad_source: str,
    vad_local_repo_dir: str,
    vad_target_sr: int,
    vad_threshold: float,
    vad_min_speech_ms: int,
    vad_max_speech_s: float,
    vad_min_silence_ms: int,
    vad_speech_pad_ms: int,
    vad_window_size_samples: int,
    vad_merge_gap_ms: int,
    vad_min_segment_ms: int,
    vad_kwargs_json: str,
    # caption
    max_chars_per_line: int,
    gap_break_s: float,
    break_on_punct: bool,
    # prompt/context (qwen-asr: `context` arg)
    context_prompt: str,
    # other
    language: str,
    torch_threads: int,
    # control
    cancel_token: CancelToken,
):
    """
    Streaming batch runner: yields (log_text, out_files) continuously.
    Supports cooperative cancellation and best-effort GPU memory release.
    """
    cancel_token.reset()
    cancel_token.set_running(True)
    # ensure "load model only when start" semantics (best effort)
    unload_model()

    set_torch_threads(torch_threads)

    if quiet_transformers:
        try:
            from transformers.utils import logging as hf_logging  # type: ignore

            hf_logging.set_verbosity_error()
            os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        except Exception:
            pass

    def parse_json_dict(name: str, s: str) -> dict:
        s = (s or "").strip()
        if not s:
            return {}
        try:
            obj = json.loads(s)
        except Exception as e:
            raise ValueError(f"{name} 不是合法 JSON：{e}")
        if not isinstance(obj, dict):
            raise ValueError(f"{name} 必须是 JSON 对象(dict)，但拿到的是 {type(obj)}")
        return obj

    # Persist logs/state so users can close the web page and come back to resume viewing.
    job_id = time.strftime("%Y%m%d_%H%M%S")
    try:
        js_reset_job(job_id=job_id)
    except Exception:
        pass

    log_tail = collections.deque(maxlen=int(os.getenv("WEBUI_LOG_TAIL_LINES", "2000")))

    def _emit(line: str) -> None:
        log_tail.append(line)
        try:
            js_append_log(line)
        except Exception:
            pass

    def _log_text() -> str:
        return "".join(log_tail)

    out_paths: list[Path] = []
    # Per-audio progress is a time-based estimate. This is "fake but finishes at 100%":
    # bounded to 95% while worker is running, jumps to 100% when worker reports done.
    try:
        # Default is intentionally "fast-ish" to avoid showing 5-10% then finishing.
        rtf_est = float(os.getenv("ASR_PROGRESS_RTF", "0.18"))
    except Exception:
        rtf_est = 0.18

    # UI download list: user wants everything under mounted OUTPUT_DIR (e.g. /output).
    ui_output_dir = Path(os.getenv("WEBUI_OUTPUT_DIR") or os.getenv("OUTPUT_DIR") or str(Path.cwd() / "output"))
    ui_output_dir.mkdir(parents=True, exist_ok=True)

    def _status_html(pct: int, name: str | None, extra: str | None = None) -> str:
        pct = max(0, min(100, int(pct)))
        title = html.escape(name or "（空闲）")
        extra_s = f" | {html.escape(extra)}" if extra else ""
        # Try to match Gradio orange-ish progress color.
        return (
            "<div style='display:flex;flex-direction:column;gap:6px;'>"
            f"<div style='display:flex;justify-content:space-between;gap:12px;'>"
            f"<div><b>当前音频</b>：{title}{extra_s}</div><div><b>{pct}%</b></div>"
            "</div>"
            "<div style='width:100%;height:12px;border:1px solid #e5e7eb;border-radius:999px;overflow:hidden;background:#f3f4f6;'>"
            f"<div style='height:100%;width:{pct}%;background:#f97316;transition:width 0.15s linear;'></div>"
            "</div>"
            "</div>"
        )

    def _safe_name(s: str) -> str:
        s = (s or "").strip()
        s = s.replace("/", "_").replace("\\", "_")
        s = re.sub(r'[:*?"<>|]', "_", s)
        return s or "output"

    def _safe_rel_parts(rel: Path) -> list[str]:
        out: list[str] = []
        for part in rel.parts:
            p = _safe_name(part)
            out.append(p[:80] if len(p) > 80 else p)
        return out

    def _ui_target_for(audio_path: Path, *, ext: str, idx: int) -> Path:
        # Preserve relative folder structure under input_dir when possible to avoid name collisions.
        if input_mode == "dir":
            try:
                rel = audio_path.relative_to(Path(input_dir).expanduser())
                parent_parts = _safe_rel_parts(rel.parent)
                dst_dir = ui_output_dir.joinpath(*parent_parts) if parent_parts else ui_output_dir
                dst_dir.mkdir(parents=True, exist_ok=True)
                return dst_dir / f"{_safe_name(audio_path.stem)}{ext}"
            except Exception:
                pass
        # Fallback: prefix index to avoid collisions.
        dst_dir = ui_output_dir
        dst_dir.mkdir(parents=True, exist_ok=True)
        return dst_dir / f"{idx:04d}_{_safe_name(audio_path.stem)}{ext}"

    def _copy_to_ui(src: Path, *, audio_path: Path, ext: str, idx: int) -> Path:
        try:
            # If already under ui_output_dir, return as-is.
            try:
                if src.resolve().is_relative_to(ui_output_dir.resolve()):
                    return src
            except Exception:
                pass
            dst = _ui_target_for(audio_path, ext=ext, idx=idx)
            shutil.copy2(src, dst)
            return dst
        except Exception:
            # Worst-case: return src and rely on allowed_paths (if set by user).
            return src

    def _get_audio_duration_s(p: Path) -> float | None:
        # Try torchaudio.info first (fast, no full decode), then ffprobe as fallback.
        try:
            import torchaudio  # type: ignore

            info = torchaudio.info(str(p))
            if getattr(info, "num_frames", 0) and getattr(info, "sample_rate", 0):
                return float(info.num_frames) / float(info.sample_rate)
        except Exception:
            pass

        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            return None
        try:
            cmd = [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(p),
            ]
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
            if out:
                return float(out)
        except Exception:
            return None
        return None
    try:
        if input_mode == "dir":
            exts = [x.strip() for x in (exts_csv or "").split(",") if x.strip()]
            files = list_audio_files(input_dir, recursive=recursive, exts=exts or MEDIA_EXTS_DEFAULT)
            input_root = Path(input_dir).expanduser()
        else:
            files = _normalize_uploads(uploads)
            input_root = None

        if not files:
            _emit("未找到音视频文件。请检查输入目录/后缀，或上传文件。\n")
            try:
                js_update_state(dict(running=False, done=True, progress_pct=0, total=0, current_file=None))
            except Exception:
                pass
            yield _status_html(0, None), _log_text(), [], False
            return

        backend_n = (backend or "transformers").strip().lower()
        requested_device = (device_map or "").strip().lower()
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            msg = (
                "检测到你选择了 CUDA（device_map 以 cuda 开头），但当前 Python 环境的 torch 不支持 CUDA。\n"
                "你现在的环境通常是安装了 CPU 版 torch（例如 torch==...+cpu），会导致报错：Torch not compiled with CUDA enabled。\n\n"
                "解决办法：\n"
                "1) 安装 CUDA 版 PyTorch（推荐新建 Python 3.12 环境再装）；或\n"
                "2) 临时把 device_map 改成 cpu（会非常慢）。\n"
            )
            _emit(msg)
            try:
                js_update_state(dict(running=False, done=True, last_error="cuda not available"))
            except Exception:
                pass
            yield _status_html(0, None, "失败"), _log_text(), [], False
            return

        asr_init_kwargs = parse_json_dict("ASR init kwargs", asr_init_kwargs_json)
        aligner_init_kwargs = parse_json_dict("Aligner init kwargs", aligner_init_kwargs_json)
        transcribe_kwargs = parse_json_dict("transcribe kwargs", transcribe_kwargs_json)
        vad_kwargs = parse_vad_json_dict("VAD kwargs", vad_kwargs_json)
        vllm_kwargs = parse_json_dict("vLLM kwargs", vllm_kwargs_json)

        # qwen-asr transcribe() only exposes `context` as a prompt-like knob.
        # Filter unknown keys to avoid TypeError crashing the batch.
        dropped = [k for k in list(transcribe_kwargs.keys()) if k != "context"]
        for k in dropped:
            transcribe_kwargs.pop(k, None)
        if dropped:
            _emit(f"[提示] transcribe kwargs 不支持这些字段，已忽略：{', '.join(dropped)}\n")

        # Merge UI context (higher priority if provided).
        if (context_prompt or "").strip():
            transcribe_kwargs["context"] = (context_prompt or "").strip()

        # vLLM: cap max model len by default to avoid KV cache startup failure
        if backend_n == "vllm":
            # Reduce CUDA memory fragmentation risk for long runs.
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            if "max_model_len" not in vllm_kwargs and "max_seq_len" not in vllm_kwargs:
                try:
                    vllm_kwargs["max_model_len"] = int(os.getenv("VLLM_MAX_MODEL_LEN", "16384"))
                except Exception:
                    vllm_kwargs["max_model_len"] = 16384

        asr_cfg = ASRConfig(
            backend=("vllm" if backend_n == "vllm" else "transformers"),
            asr_checkpoint=asr_checkpoint.strip(),
            aligner_checkpoint=aligner_checkpoint.strip(),
            device_map=device_map.strip() or "cuda:0",
            dtype=dtype,
            max_inference_batch_size=int(max_inference_batch_size),
            max_new_tokens=int(max_new_tokens),
            attn_implementation=(attn_impl.strip() or None),
            asr_init_kwargs=asr_init_kwargs or None,
            aligner_init_kwargs=aligner_init_kwargs or None,
            vllm_kwargs=(
                {
                    **({"gpu_memory_utilization": float(vllm_gpu_memory_utilization)} if backend_n == "vllm" else {}),
                    **(vllm_kwargs or {}),
                }
                or None
            ),
            cuda_visible_devices=(vllm_cuda_visible_devices.strip() or None),
        )
        cap_cfg = CaptionConfig(
            output_format=output_format,
            max_chars_per_line=int(max_chars_per_line),
            gap_break_s=float(gap_break_s),
            break_on_punct=bool(break_on_punct),
        )
        vad_cfg = VadConfig(
            enabled=bool(vad_enabled),
            backend="silero" if (vad_backend or "silero") == "silero" else "silero",
            source=(vad_source or "auto"),  # auto/local/hub
            local_repo_dir=(vad_local_repo_dir.strip() or None),
            target_sr=int(vad_target_sr),
            threshold=float(vad_threshold),
            min_speech_duration_ms=int(vad_min_speech_ms),
            max_speech_duration_s=float(vad_max_speech_s),
            min_silence_duration_ms=int(vad_min_silence_ms),
            speech_pad_ms=int(vad_speech_pad_ms),
            window_size_samples=int(vad_window_size_samples),
            merge_gap_ms=int(vad_merge_gap_ms),
            min_segment_ms=int(vad_min_segment_ms),
            vad_kwargs=vad_kwargs or None,
        )

        n = len(files)
        default_out_dir = Path(os.getenv("OUTPUT_DIR", str(Path.cwd() / "output")))
        _emit(f"共 {n} 个音频，开始批量生成…\n")
        try:
            js_update_state(dict(running=True, done=False, total=n, progress_pct=0, current_file=None, current_idx=0))
        except Exception:
            pass
        # Only show downloadable files at the END (avoid gradio re-caching conflicts mid-run).
        ui_paths: dict[str, Path] = {}

        seen_folders: set[str] = set()

        def _folder_label(p: Path) -> str:
            # Group by immediate child folder under input_dir; do not include grandchildren.
            if input_mode != "dir" or input_root is None:
                return "（上传）"
            try:
                rel = p.relative_to(input_root)
                if len(rel.parts) <= 1:
                    return "（根目录）"
                return str(rel.parts[0])
            except Exception:
                return "（其它）"

        def _maybe_log_folder(p: Path) -> None:
            folder = _folder_label(p)
            if folder not in seen_folders:
                seen_folders.add(folder)
                _emit(f"[处理文件夹] {folder}\n")

        def _ui_list() -> list[str]:
            return [str(p) for p in ui_paths.values()]

        def _ui_add(p: Path | None) -> None:
            if p is None:
                return
            try:
                ui_paths[str(p)] = p
            except Exception:
                pass

        yield _status_html(0, None), _log_text(), [], True

        # -------- vLLM mode: run the whole batch in ONE isolated subprocess --------
        # This avoids accumulating multiple vLLM engines/EngineCore procs (which can OOM mid-run),
        # and still makes "cancel -> VRAM to 0" reliable (terminate that subprocess).
        if backend_n == "vllm":
            ctx = mp.get_context("spawn")
            start_idx0 = 0
            # We'll restart worker on OOM and continue from current index.
            gpu_util = float(vllm_gpu_memory_utilization)
            retry_same = 0

            def _start_worker() -> tuple[Any, Any, Any]:
                mp_cancel = ctx.Event()
                q = ctx.Queue()
                asr_cfg_d = asdict(asr_cfg)
                # adjust gpu_memory_utilization inside vllm_kwargs (engine config)
                try:
                    vk = dict(asr_cfg_d.get("vllm_kwargs") or {})
                    if backend_n == "vllm":
                        vk["gpu_memory_utilization"] = float(gpu_util)
                    asr_cfg_d["vllm_kwargs"] = vk
                except Exception:
                    pass

                p = ctx.Process(
                    target=vllm_worker_batch,
                    args=(q, mp_cancel),
                    kwargs=dict(
                        files=[str(x) for x in files],
                        start_idx0=int(start_idx0),
                        asr_cfg_d=asr_cfg_d,
                        lang_s=_normalize_language(language),
                        cap_cfg_d=asdict(cap_cfg),
                        transcribe_kwargs_obj=(transcribe_kwargs or None),
                        vad_cfg_d=asdict(vad_cfg),
                        output_dir_mode_s=output_dir_mode,
                        custom_output_dir_s=(custom_output_dir.strip() or None),
                        overwrite_b=bool(overwrite),
                        default_out_dir_s=str(default_out_dir),
                    ),
                    daemon=False,
                )
                cancel_token.attach_worker(proc=p, mp_cancel=mp_cancel)
                p.start()
                return p, q, mp_cancel

            p, q, mp_cancel = _start_worker()

            cur_name: str | None = None
            cur_idx = 0
            total = n
            init_t0 = time.perf_counter()
            file_t0: float | None = None
            file_expected_s: float | None = None
            base_pct: int = 0
            last_emit = 0.0
            yield _status_html(0, None, "初始化 vLLM…"), gr.update(), gr.update(), True

            while True:
                if cancel_token.is_cancelled():
                    mp_cancel.set()
                    cancel_token.terminate_worker()
                    _emit("已请求取消：已终止推理进程，释放显存…\n")
                    try:
                        js_update_state(dict(running=False, cancel_requested=True, done=True))
                    except Exception:
                        pass
                    yield _status_html(0, cur_name, "已取消"), _log_text(), _ui_list(), False
                    return

                try:
                    msg_type, payload = q.get(timeout=0.2)
                except Exception:
                    # If worker died unexpectedly, restart from current index.
                    try:
                        if p is not None and hasattr(p, "is_alive") and (not p.is_alive()):
                            _emit("[警告] vLLM 进程异常退出，正在重启并继续…\n")
                            try:
                                cancel_token.terminate_worker()
                            except Exception:
                                pass
                            p, q, mp_cancel = _start_worker()
                            try:
                                js_update_state(dict(progress_pct=0, current_file=cur_name, current_idx=cur_idx, total=total, last_error="worker died"))
                            except Exception:
                                pass
                            yield _status_html(0, cur_name, "重启 vLLM…"), _log_text(), [], True
                            continue
                    except Exception:
                        pass
                    # No message: advance a time-based estimate so progress doesn't look stuck.
                    now = time.perf_counter()
                    if now - last_emit < 0.25:
                        continue
                    last_emit = now

                    # During init (before first file_start), show gentle progress up to 5%.
                    if file_t0 is None:
                        elapsed = max(0.0, now - init_t0)
                        pct = int(min(5.0, (elapsed / 20.0) * 5.0))
                        yield _status_html(pct, None, "初始化 vLLM…"), gr.update(), gr.update(), True
                        continue

                    # During a file, smooth from base_pct up to 90% using expected time.
                    elapsed = max(0.0, now - file_t0)
                    expected = max(1.0, float(file_expected_s or 30.0))
                    # If VAD is enabled, keep pre-VAD progress in [0,9] and only
                    # jump to ~10-15% when VAD truly completes (child reports >=12).
                    if bool(vad_cfg.enabled) and int(base_pct) < 12:
                        target = 9
                    else:
                        target = 90
                    span = max(1, target - int(base_pct))
                    frac = min(0.98, elapsed / expected)
                    pct = max(int(base_pct), min(target, int(base_pct) + int(round(frac * span))))
                    yield _status_html(pct, cur_name, f"{cur_idx}/{total}"), gr.update(), gr.update(), True
                    continue

                if msg_type == "init":
                    yield _status_html(0, None, "初始化 vLLM…"), gr.update(), gr.update(), True
                elif msg_type == "file_start":
                    cur_idx, total, cur_name = payload
                    start_idx0 = int(cur_idx - 1)
                    retry_same = 0
                    try:
                        _maybe_log_folder(Path(files[cur_idx - 1]))
                    except Exception:
                        pass
                    _emit(f"[处理中] {cur_name} ({cur_idx}/{total})\n")
                    # reset per-file estimate timers
                    file_t0 = time.perf_counter()
                    base_pct = 0
                    try:
                        dur_s = _get_audio_duration_s(Path(files[cur_idx - 1]))
                    except Exception:
                        dur_s = None
                    file_expected_s = max(3.0, (dur_s or 30.0) * max(0.1, rtf_est))
                    try:
                        js_update_state(dict(progress_pct=0, current_file=cur_name, current_idx=cur_idx, total=total, running=True, done=False))
                    except Exception:
                        pass
                    yield _status_html(0, cur_name, f"{cur_idx}/{total}"), _log_text(), [], True
                elif msg_type == "progress":
                    try:
                        pct = int(payload)
                    except Exception:
                        pct = 0
                    base_pct = max(base_pct, pct)
                    try:
                        js_update_state(dict(progress_pct=int(pct), current_file=cur_name, current_idx=cur_idx, total=total))
                    except Exception:
                        pass
                    yield _status_html(pct, cur_name, f"{cur_idx}/{total}"), gr.update(), gr.update(), True
                elif msg_type == "file_done":
                    idx1, total, log_line, out_path_s = payload
                    _emit(str(log_line))
                    if out_path_s:
                        outp = Path(out_path_s)
                        out_paths.append(outp)
                        _ui_add(_copy_to_ui(outp, audio_path=Path(files[idx1 - 1]), ext=outp.suffix, idx=len(ui_paths) + 1))
                    # advance start index
                    start_idx0 = int(idx1)
                    file_t0 = None
                    try:
                        js_update_state(dict(progress_pct=100, current_file=cur_name, current_idx=idx1, total=total))
                    except Exception:
                        pass
                    yield _status_html(100, cur_name, f"{idx1}/{total}"), _log_text(), [], True
                elif msg_type == "file_error":
                    idx1, total, msg = payload
                    name = cur_name
                    if idx1 and 1 <= idx1 <= len(files):
                        name = Path(files[idx1 - 1]).name
                    _emit(f"[失败] {name}: {msg}\n")
                    # If OOM, restart vLLM worker (engine may be in bad state).
                    if isinstance(msg, str) and ("CUDA out of memory" in msg or "out of memory" in msg.lower()):
                        _emit("[警告] 检测到 OOM：重启 vLLM 并下调 gpu_memory_utilization 后继续…\n")
                        try:
                            mp_cancel.set()
                        except Exception:
                            pass
                        try:
                            cancel_token.terminate_worker()
                        except Exception:
                            pass
                        # retry same file once with lower util, then skip
                        if retry_same < 1:
                            retry_same += 1
                            start_idx0 = int(max(0, idx1 - 1))
                        else:
                            start_idx0 = int(idx1)
                        gpu_util = max(0.55, gpu_util - 0.05)
                        p, q, mp_cancel = _start_worker()
                        try:
                            js_update_state(dict(progress_pct=0, current_file=name, current_idx=idx1, total=total, last_error="oom restart"))
                        except Exception:
                            pass
                        yield _status_html(0, name, f"重启 vLLM（gpu_util={gpu_util:.2f}）"), _log_text(), [], True
                        continue
                    file_t0 = None
                    try:
                        js_update_state(dict(progress_pct=100, current_file=name, current_idx=idx1, total=total, last_error=str(msg)))
                    except Exception:
                        pass
                    yield _status_html(100, name, f"{idx1}/{total}"), _log_text(), [], True
                elif msg_type == "cancelled":
                    _emit("已取消：停止后续任务，准备释放显存…\n")
                    try:
                        js_update_state(dict(running=False, cancel_requested=True, done=True))
                    except Exception:
                        pass
                    yield _status_html(0, cur_name, "已取消"), _log_text(), _ui_list(), False
                    return
                elif msg_type == "done":
                    break

            try:
                p.join(timeout=2.0)
            except Exception:
                pass
            cancel_token.attach_worker(proc=None, mp_cancel=mp_cancel)

            _emit("全部完成，准备释放显存…\n")
            try:
                js_update_state(dict(running=False, done=True, progress_pct=100, output_files=_ui_list()))
            except Exception:
                pass
            yield _status_html(100, None, "全部完成"), _log_text(), _ui_list(), False
            return

        # -------- transformers mode (in-process, thread-estimated progress) --------
        for i, audio_path in enumerate(files):
            if cancel_token.is_cancelled():
                _emit("已取消：停止后续任务，准备释放显存…\n")
                try:
                    js_update_state(dict(running=False, cancel_requested=True, done=True))
                except Exception:
                    pass
                yield _status_html(0, None, "已取消"), _log_text(), _ui_list(), False
                return

            _maybe_log_folder(audio_path)
            _emit(f"[处理中] {audio_path.name} ({i+1}/{n})\n")
            try:
                js_update_state(dict(progress_pct=0, current_file=audio_path.name, current_idx=i + 1, total=n, running=True, done=False))
            except Exception:
                pass
            yield _status_html(0, audio_path.name, f"{i+1}/{n}"), _log_text(), [], True

            # Per-audio "relative real" progress: time-based estimate bounded to 95% until done.
            dur_s = _get_audio_duration_s(audio_path)
            expected_s = max(3.0, (dur_s or 30.0) * max(0.1, rtf_est))

            result: dict[str, Any] = {}
            err: dict[str, Any] = {}

            def _worker():
                try:
                    log_line, content, ext = generate_for_one_audio(
                        audio_path,
        asr_cfg=asr_cfg,
        language=_normalize_language(language),
        caption_cfg=cap_cfg,
        transcribe_kwargs=transcribe_kwargs or None,
        vad_cfg=vad_cfg,
                    )
                    out_path: Path | None = None
                    if ext:
                        out_dir = resolve_output_dir(
                            input_audio_path=audio_path,
                            mode=output_dir_mode,
                            custom_dir=(custom_output_dir.strip() or None),
                            default_output_dir=default_out_dir,
                        )
                        out_path = write_output(
                            audio_path=audio_path,
                            content=content,
                            ext=ext,
                            output_dir=out_dir,
                            overwrite=overwrite,
                        )
                    result.update(dict(log_line=log_line, out_path=out_path))
                except Exception as e:
                    err["e"] = e
                    if debug_traceback:
                        import traceback

                        err["tb"] = traceback.format_exc()

            t0 = time.perf_counter()
            th = threading.Thread(target=_worker, daemon=True)
            th.start()

            last_emit = 0.0
            while th.is_alive():
                if cancel_token.is_cancelled():
                    _emit("已请求取消：等待当前任务收尾…\n")
                    try:
                        js_update_state(dict(running=False, cancel_requested=True))
                    except Exception:
                        pass
                    yield _status_html(0, audio_path.name, "已取消"), _log_text(), _ui_list(), False
                    break

                elapsed = max(0.0, time.perf_counter() - t0)
                frac = min(0.95, (elapsed / max(0.1, expected_s)) * 0.95)
                audio_pct = int(round(frac * 100))

                now = time.perf_counter()
                if now - last_emit >= 0.25:
                    last_emit = now
                    try:
                        js_update_state(dict(progress_pct=int(audio_pct), current_file=audio_path.name, current_idx=i + 1, total=n))
                    except Exception:
                        pass
                    yield _status_html(audio_pct, audio_path.name, f"{i+1}/{n}"), gr.update(), gr.update(), True
                time.sleep(0.05)

            th.join()
            elapsed_total = max(0.0, time.perf_counter() - t0)

            if err:
                e = err.get("e")
                tb = err.get("tb")
                if tb:
                    _emit(f"[失败] {audio_path}: {e}\n{tb}\n")
                else:
                    _emit(f"[失败] {audio_path}: {e}\n")
                try:
                    js_update_state(dict(progress_pct=100, current_file=audio_path.name, current_idx=i + 1, total=n, last_error=str(e)))
                except Exception:
                    pass
                yield _status_html(100, audio_path.name, f"{i+1}/{n}"), _log_text(), [], True
            else:
                _emit(str(result.get("log_line", "")))
                op = result.get("out_path")
                if isinstance(op, Path):
                    out_paths.append(op)
                    _ui_add(_copy_to_ui(op, audio_path=audio_path, ext=op.suffix, idx=len(ui_paths) + 1))
                # update RTF estimate (EMA)
                if dur_s and dur_s > 0.5:
                    rtf_now = elapsed_total / float(dur_s)
                    rtf_est = max(0.05, min(10.0, 0.85 * rtf_est + 0.15 * rtf_now))
                try:
                    js_update_state(dict(progress_pct=100, current_file=audio_path.name, current_idx=i + 1, total=n))
                except Exception:
                    pass
                yield _status_html(100, audio_path.name, f"{i+1}/{n}"), _log_text(), [], True

        _emit("全部完成，准备释放显存…\n")
        try:
            js_update_state(dict(running=False, done=True, progress_pct=100, output_files=_ui_list()))
        except Exception:
            pass
        yield _status_html(100, None, "全部完成"), _log_text(), _ui_list(), False
    except Exception as e:
        _emit(f"\n[失败] {e}\n")
        try:
            js_update_state(dict(running=False, done=True, last_error=str(e)))
        except Exception:
            pass
        yield _status_html(0, None, "失败"), _log_text(), [], False
    finally:
        # Always unload after a batch so next click reloads model into VRAM.
        unload_model()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        cancel_token.set_running(False)


def build_ui() -> gr.Blocks:
    cfg_path = default_webui_config_path()
    persisted = load_json(cfg_path)
    if not cfg_path.exists():
        # Create a seed file early so mounts/permissions issues are visible,
        # and users can edit config manually if desired.
        try:
            merge_update(cfg_path, {})
        except Exception:
            pass

    # Config migration: older configs may have audio-only `exts_csv`.
    # Ensure video suffixes are included by default once video support is enabled.
    try:
        cur_exts = str(persisted.get("exts_csv", "") if isinstance(persisted, dict) else "").strip()
        if cur_exts:
            parts = [x.strip().lower() for x in cur_exts.split(",") if x.strip()]
            parts = [p if p.startswith(".") else f".{p}" for p in parts]
            has_video = any(p in set(VIDEO_EXTS_DEFAULT) for p in parts)
            if not has_video:
                merged = parts + [x for x in VIDEO_EXTS_DEFAULT if x not in parts]
                new_exts = ",".join(merged)
                persisted["exts_csv"] = new_exts
                merge_update(cfg_path, {"exts_csv": new_exts})
    except Exception:
        pass

    def pick(key: str, default: Any, *, valid: set[Any] | None = None) -> Any:
        if isinstance(persisted, dict) and key in persisted:
            v = persisted.get(key)
        else:
            v = default
        if valid is not None and v not in valid:
            return default
        return v

    def as_int(v: Any, default: int) -> int:
        try:
            return int(v)
        except Exception:
            return int(default)

    def as_float(v: Any, default: float) -> float:
        try:
            return float(v)
        except Exception:
            return float(default)

    def as_bool(v: Any, default: bool) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"1", "true", "yes", "y", "on"}:
                return True
            if s in {"0", "false", "no", "n", "off"}:
                return False
        return bool(default)

    with gr.Blocks(title="Qwen3-ASR 批量字幕生成 (SRT/LRC)") as demo:
        gr.Markdown(
            "批量转写并生成带时间轴的字幕（SRT）/歌词（LRC）。"
            " 模型基于 `qwen-asr`：`Qwen3-ASR-1.7B` + `Qwen3-ForcedAligner-0.6B`。"
        )

        config_status = gr.Markdown(
            f"配置文件：`{cfg_path.as_posix()}`（挂载 `data/` 后可持久化）。",
        )

        # Cancel is handled globally by detached runner; no per-session CancelToken needed.

        with gr.Accordion("输入", open=True):
            input_mode = gr.Radio(
                choices=[("目录扫描", "dir"), ("拖拽上传", "upload")],
                value=pick("input_mode", "dir", valid={"dir", "upload"}),
                label="输入方式",
            )

            with gr.Row():
                input_dir = gr.Textbox(value=str(pick("input_dir", str(Path.cwd()))), label="输入目录（仅目录扫描时生效）")
                recursive = gr.Checkbox(value=as_bool(pick("recursive", True), True), label="递归子目录")

            exts_csv = gr.Textbox(
                value=str(pick("exts_csv", ",".join(MEDIA_EXTS_DEFAULT))),
                label="音视频后缀（逗号分隔，仅目录扫描时生效）",
            )

            uploads = gr.Files(label="拖拽上传音视频（可多选）", file_count="multiple")

            touch_album_root_mtime = gr.Checkbox(
                value=as_bool(pick("touch_album_root_mtime", False), False),
                label="更新专辑根目录修改时间（可选：触发输入目录一级子文件夹 mtime 更新）",
            )

        with gr.Accordion("输出", open=True):
            with gr.Row():
                output_format = gr.Radio(
                    choices=["srt", "lrc"],
                    value=pick("output_format", "srt", valid={"srt", "lrc"}),
                    label="输出格式",
                )
                overwrite = gr.Checkbox(value=as_bool(pick("overwrite", False), False), label="覆盖已存在文件")

            with gr.Row():
                output_dir_mode = gr.Radio(
                    choices=[("默认 output/ 目录", "output"), ("与输入文件同目录", "same"), ("自定义目录", "custom")],
                    value=pick("output_dir_mode", "output", valid={"output", "same", "custom"}),
                    label="输出路径策略",
                )
                custom_output_dir = gr.Textbox(value=str(pick("custom_output_dir", "")), label="自定义输出目录（仅自定义时生效）")

        with gr.Accordion("字幕分句/切行（基于强制对齐 token 时间戳）", open=False):
            with gr.Row():
                max_chars_per_line = gr.Slider(
                    10,
                    120,
                    value=as_int(pick("max_chars_per_line", 40), 40),
                    step=1,
                    label="每行最大字符数（触发切分）",
                )
                gap_break_s = gr.Slider(
                    0.0,
                    3.0,
                    value=as_float(pick("gap_break_s", 0.8), 0.8),
                    step=0.1,
                    label="停顿切分阈值（秒）",
                )
            break_on_punct = gr.Checkbox(value=as_bool(pick("break_on_punct", True), True), label="遇到标点即切分")

        with gr.Accordion("VAD 预处理（可选：先做语音活动检测，再按语音段转写并回填绝对时间轴）", open=False):
            vad_enabled = gr.Checkbox(value=as_bool(pick("vad_enabled", False), False), label="启用 VAD")
            vad_backend = gr.Dropdown(choices=["silero"], value="silero", label="VAD 后端")

            with gr.Row():
                vad_source = gr.Dropdown(
                    choices=[("自动（优先本地，其次自动下载）", "auto"), ("仅本地（离线）", "local"), ("仅自动下载（hub）", "hub")],
                    value=pick("vad_source", "auto", valid={"auto", "local", "hub"}),
                    label="VAD 来源模式",
                )
                vad_local_repo_dir = gr.Textbox(
                    value=str(pick("vad_local_repo_dir", os.getenv("VAD_REPO_DIR", ""))),
                    label="本地 silero-vad 目录（可选；source=local 时必填；也可用环境变量 VAD_REPO_DIR）",
                )

            with gr.Row():
                vad_target_sr = gr.Dropdown(
                    choices=[8000, 16000, 32000, 48000],
                    value=as_int(pick("vad_target_sr", 16000), 16000),
                    label="VAD 采样率（解码/重采样）",
                )
                vad_threshold = gr.Slider(0.0, 1.0, value=as_float(pick("vad_threshold", 0.5), 0.5), step=0.01, label="threshold")

            with gr.Row():
                vad_min_speech_ms = gr.Slider(
                    0,
                    2000,
                    value=as_int(pick("vad_min_speech_ms", 250), 250),
                    step=10,
                    label="min_speech_duration_ms",
                )
                vad_min_silence_ms = gr.Slider(
                    0,
                    2000,
                    value=as_int(pick("vad_min_silence_ms", 100), 100),
                    step=10,
                    label="min_silence_duration_ms",
                )
                vad_speech_pad_ms = gr.Slider(
                    0,
                    1000,
                    value=as_int(pick("vad_speech_pad_ms", 30), 30),
                    step=10,
                    label="speech_pad_ms",
                )

            with gr.Row():
                vad_max_speech_s = gr.Slider(
                    1,
                    300,
                    value=as_float(pick("vad_max_speech_s", 60), 60),
                    step=1,
                    label="max_speech_duration_s（单段上限，过长会被切段）",
                )
                vad_window_size_samples = gr.Dropdown(
                    choices=[256, 512, 768, 1024, 1536],
                    value=as_int(pick("vad_window_size_samples", 512), 512),
                    label="window_size_samples",
                )

            with gr.Row():
                vad_merge_gap_ms = gr.Slider(
                    0,
                    2000,
                    value=as_int(pick("vad_merge_gap_ms", 120), 120),
                    step=10,
                    label="merge_gap_ms（段间静音<=该值则合并）",
                )
                vad_min_segment_ms = gr.Slider(
                    0,
                    5000,
                    value=as_int(pick("vad_min_segment_ms", 300), 300),
                    step=10,
                    label="min_segment_ms（过滤过短段）",
                )

            vad_kwargs_json = gr.Code(
                value=str(pick("vad_kwargs_json", "{}")),
                language="json",
                label="VAD kwargs（JSON 透传到 silero get_speech_timestamps；可填入你想传的所有参数）",
            )

        with gr.Accordion("模型参数（transformers backend）", open=False):
            backend = gr.Radio(
                choices=[("transformers（默认）", "transformers"), ("vLLM（官方推荐更快）", "vllm")],
                value=pick("backend", "transformers", valid={"transformers", "vllm"}),
                label="后端选择",
            )
            asr_checkpoint = gr.Textbox(value=str(pick("asr_checkpoint", DEFAULT_ASR)), label="ASR 模型（HF ID 或本地路径）")
            aligner_checkpoint = gr.Textbox(value=str(pick("aligner_checkpoint", DEFAULT_ALIGNER)), label="ForcedAligner 模型（HF ID 或本地路径）")

            with gr.Row():
                device_map = gr.Textbox(value=str(pick("device_map", "cuda:0")), label="device_map")
                dtype = gr.Dropdown(
                    choices=["bf16", "fp16", "fp32"],
                    value=pick("dtype", "fp16", valid={"bf16", "fp16", "fp32"}),
                    label="dtype",
                )

            with gr.Row():
                max_inference_batch_size = gr.Slider(
                    1,
                    32,
                    value=as_int(pick("max_inference_batch_size", 1), 1),
                    step=1,
                    label="max_inference_batch_size",
                )
                max_new_tokens = gr.Slider(
                    128,
                    8192,
                    value=as_int(pick("max_new_tokens", 2048), 2048),
                    step=128,
                    label="max_new_tokens",
                )

            attn_impl = gr.Textbox(
                value=str(pick("attn_impl", "")),
                label="attn_implementation（可选，如 flash_attention_2；需要安装 flash-attn 且 dtype=fp16/bf16）",
            )

            with gr.Accordion("vLLM 参数（选择 vLLM 后端时生效）", open=False):
                vllm_gpu_memory_utilization = gr.Slider(
                    0.1,
                    0.95,
                    value=as_float(pick("vllm_gpu_memory_utilization", 0.7), 0.7),
                    step=0.01,
                    label="gpu_memory_utilization（官方示例 0.7；越大越容易 OOM）",
                )
                vllm_cuda_visible_devices = gr.Textbox(
                    value=str(pick("vllm_cuda_visible_devices", os.getenv("CUDA_VISIBLE_DEVICES", "0"))),
                    label="CUDA_VISIBLE_DEVICES（vLLM 用它选卡，如 0 或 0,1；留空则不覆盖环境变量）",
                )
                vllm_kwargs_json = gr.Code(
                    value=str(pick("vllm_kwargs_json", "{}")),
                    language="json",
                    label="vLLM kwargs（JSON 透传到 Qwen3ASRModel.LLM；例如 tensor_parallel_size 等）",
                )

            with gr.Accordion("高级参数（JSON 透传，允许传入非必要/可能被忽略的参数）", open=False):
                context_prompt = gr.Textbox(
                    value=str(pick("context_prompt", "")),
                    lines=4,
                    label="提示词/上下文 context（可选，影响输出风格/热词；qwen-asr 官方参数名就是 context）",
                    placeholder="例如：这是日语音声ASMR，请只输出转写文本，不要解释；专有名词：XXX, YYY；说话人：勇者/触手怪...",
                )
                quiet_transformers = gr.Checkbox(
                    value=as_bool(pick("quiet_transformers", False), False),
                    label="减少 transformers 提示噪音（不影响推理）",
                )
                asr_init_kwargs_json = gr.Code(
                    value=str(pick("asr_init_kwargs_json", "{}")),
                    language="json",
                    label="ASR init kwargs（传给 Qwen3ASRModel.from_pretrained）",
                )
                aligner_init_kwargs_json = gr.Code(
                    value=str(pick("aligner_init_kwargs_json", "{}")),
                    language="json",
                    label="Aligner init kwargs（合并进 forced_aligner_kwargs）",
                )
                transcribe_kwargs_json = gr.Code(
                    value=str(pick("transcribe_kwargs_json", "{}")),
                    language="json",
                    label="transcribe kwargs（仅支持 {\"context\": \"...\"}；其它字段会被忽略）",
                )

        with gr.Accordion("其它", open=False):
            with gr.Row():
                language = gr.Dropdown(
                    choices=LANG_PRESETS,
                    value=str(pick("language", "(自动)")),
                    label="语言（可选）",
                    allow_custom_value=True,
                )
                torch_threads = gr.Slider(
                    1,
                    32,
                    value=as_int(pick("torch_threads", 4), 4),
                    step=1,
                    label="CPU 线程（NAS 建议 2-8）",
                )
            debug_traceback = gr.Checkbox(
                value=as_bool(pick("debug_traceback", False), False),
                label="显示详细错误（traceback，排查 vLLM/VAD/依赖问题时打开）",
            )

        with gr.Row():
            run_btn = gr.Button("开始批量生成", variant="primary")
            cancel_btn = gr.Button("取消", variant="stop")
        status = gr.HTML(value="", label="进度/状态")
        log = gr.Textbox(label="日志", lines=12)
        out_files = gr.Files(label="输出文件（可下载）", file_count="multiple")

        def _poll_job_state():
            st = js_read_state() or {}
            if not st:
                # When cancelled / cleared, hard-reset UI (no stale progress/log).
                return "", "", []
            pct = int(st.get("progress_pct", 0) or 0)
            name = st.get("current_file")
            cur = int(st.get("current_idx", 0) or 0)
            total = int(st.get("total", 0) or 0)
            running = bool(st.get("running", False))
            extra = (f"{cur}/{total}" if total else ("运行中" if running else "空闲"))

            # Flash 100% on file completion so UI doesn't miss it when files switch quickly.
            try:
                # Timer tick is 1.0s; default flash window must exceed that or UI can miss 100%.
                flash_s = float(os.getenv("WEBUI_DONE_FLASH_S", "2.0"))
            except Exception:
                flash_s = 1.5
            try:
                done_at = float(st.get("last_file_done_at") or 0.0)
            except Exception:
                done_at = 0.0
            if done_at > 0 and (time.time() - done_at) < flash_s:
                try:
                    done_name = st.get("last_file_done_name") or name
                    done_idx = int(st.get("last_file_done_idx") or cur)
                except Exception:
                    done_name = name
                    done_idx = cur
                pct = 100
                name = done_name
                if total:
                    extra = f"{done_idx}/{total} 完成"
                else:
                    extra = "完成"
            # status bar HTML: reuse same style as runner
            try:
                shtml = (
                    "<div style='display:flex;flex-direction:column;gap:6px;'>"
                    f"<div style='display:flex;justify-content:space-between;gap:12px;'>"
                    f"<div><b>当前音频</b>：{html.escape(str(name) if name else '（空闲）')} | {html.escape(extra)}</div>"
                    f"<div><b>{max(0, min(100, pct))}%</b></div></div>"
                    "<div style='width:100%;height:12px;border:1px solid #e5e7eb;border-radius:999px;overflow:hidden;background:#f3f4f6;'>"
                    f"<div style='height:100%;width:{max(0, min(100, pct))}%;background:#f97316;transition:width 0.15s linear;'></div>"
                    "</div></div>"
                )
            except Exception:
                shtml = ""

            log_txt = js_read_log_tail(max_bytes=1024 * 256)
            files = st.get("output_files") or []
            if not isinstance(files, list):
                files = []
            # only show files when done (or when provided)
            if running:
                files = []
            return shtml, log_txt, files

        # Polling timer: enables "close page and come back" resume view.
        timer = gr.Timer(1.0)
        timer.tick(fn=_poll_job_state, inputs=None, outputs=[status, log, out_files], queue=False)

        persist_keys = [
            "input_mode",
            "input_dir",
            "recursive",
            "exts_csv",
            "touch_album_root_mtime",
            "output_format",
            "output_dir_mode",
            "custom_output_dir",
            "overwrite",
            "backend",
            "asr_checkpoint",
            "aligner_checkpoint",
            "device_map",
            "dtype",
            "max_inference_batch_size",
            "max_new_tokens",
            "attn_impl",
            "quiet_transformers",
            "context_prompt",
            "asr_init_kwargs_json",
            "aligner_init_kwargs_json",
            "transcribe_kwargs_json",
            "vllm_gpu_memory_utilization",
            "vllm_cuda_visible_devices",
            "vllm_kwargs_json",
            "debug_traceback",
            "vad_enabled",
            "vad_backend",
            "vad_source",
            "vad_local_repo_dir",
            "vad_target_sr",
            "vad_threshold",
            "vad_min_speech_ms",
            "vad_max_speech_s",
            "vad_min_silence_ms",
            "vad_speech_pad_ms",
            "vad_window_size_samples",
            "vad_merge_gap_ms",
            "vad_min_segment_ms",
            "vad_kwargs_json",
            "max_chars_per_line",
            "gap_break_s",
            "break_on_punct",
            "language",
            "torch_threads",
        ]
        persist_inputs = [
            input_mode,
            input_dir,
            recursive,
            exts_csv,
            touch_album_root_mtime,
            output_format,
            output_dir_mode,
            custom_output_dir,
            overwrite,
            backend,
            asr_checkpoint,
            aligner_checkpoint,
            device_map,
            dtype,
            max_inference_batch_size,
            max_new_tokens,
            attn_impl,
            quiet_transformers,
            context_prompt,
            asr_init_kwargs_json,
            aligner_init_kwargs_json,
            transcribe_kwargs_json,
            vllm_gpu_memory_utilization,
            vllm_cuda_visible_devices,
            vllm_kwargs_json,
            debug_traceback,
            vad_enabled,
            vad_backend,
            vad_source,
            vad_local_repo_dir,
            vad_target_sr,
            vad_threshold,
            vad_min_speech_ms,
            vad_max_speech_s,
            vad_min_silence_ms,
            vad_speech_pad_ms,
            vad_window_size_samples,
            vad_merge_gap_ms,
            vad_min_segment_ms,
            vad_kwargs_json,
            max_chars_per_line,
            gap_break_s,
            break_on_punct,
            language,
            torch_threads,
        ]

        def persist_config(*vals: Any) -> str:
            try:
                updates = {k: v for k, v in zip(persist_keys, vals)}
                merge_update(cfg_path, updates)
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                return f"配置文件：`{cfg_path.as_posix()}` | 最近保存：**{ts}**"
            except Exception as e:
                return f"配置保存失败：`{cfg_path.as_posix()}` | 错误：**{e}**"

        # Auto-persist: any settings change writes to data/webui_config.json (mounted => persistent).
        for c in persist_inputs:
            try:
                c.change(fn=persist_config, inputs=persist_inputs, outputs=[config_status], queue=False)
            except Exception:
                # be tolerant to gradio component differences across versions
                pass

        # Textboxes / code editors often don't trigger .change() until blur/enter.
        # Make autosave robust by marking config "dirty" on .input(), then a timer saves periodically.
        cfg_dirty = gr.State(False)

        def _mark_cfg_dirty(*_vals: Any) -> bool:
            return True

        def _persist_if_dirty(*vals_and_dirty: Any) -> tuple[Any, bool]:
            try:
                dirty = bool(vals_and_dirty[-1])
            except Exception:
                dirty = False
            if not dirty:
                return gr.update(), False
            vals = vals_and_dirty[:-1]
            msg = persist_config(*vals)
            return msg, False

        for c in persist_inputs:
            try:
                c.input(fn=_mark_cfg_dirty, inputs=None, outputs=[cfg_dirty], queue=False)
            except Exception:
                pass

        # Reuse the existing 1s timer tick cadence: save config if dirty.
        try:
            timer.tick(
                fn=_persist_if_dirty,
                inputs=persist_inputs + [cfg_dirty],
                outputs=[config_status, cfg_dirty],
                queue=False,
            )
        except Exception:
            pass

        def _start_detached(
            input_mode,
            input_dir,
            recursive,
            exts_csv,
            uploads,
            output_format,
            output_dir_mode,
            custom_output_dir,
            overwrite,
            backend,
            asr_checkpoint,
            aligner_checkpoint,
            device_map,
            dtype,
            max_inference_batch_size,
            max_new_tokens,
            attn_impl,
            asr_init_kwargs_json,
            aligner_init_kwargs_json,
            transcribe_kwargs_json,
            quiet_transformers,
            vllm_gpu_memory_utilization,
            vllm_cuda_visible_devices,
            vllm_kwargs_json,
            debug_traceback,
            vad_enabled,
            vad_backend,
            vad_source,
            vad_local_repo_dir,
            vad_target_sr,
            vad_threshold,
            vad_min_speech_ms,
            vad_max_speech_s,
            vad_min_silence_ms,
            vad_speech_pad_ms,
            vad_window_size_samples,
            vad_merge_gap_ms,
            vad_min_segment_ms,
            vad_kwargs_json,
            max_chars_per_line,
            gap_break_s,
            break_on_punct,
            context_prompt,
            language,
            torch_threads,
            touch_album_root_mtime,
        ):
            msg = detached_start_job(
                input_mode=input_mode,
                input_dir=input_dir,
                recursive=recursive,
                exts_csv=exts_csv,
                uploads=uploads,
                output_format=output_format,
                output_dir_mode=output_dir_mode,
                custom_output_dir=custom_output_dir,
                overwrite=overwrite,
                backend=backend,
                asr_checkpoint=asr_checkpoint,
                aligner_checkpoint=aligner_checkpoint,
                device_map=device_map,
                dtype=dtype,
                max_inference_batch_size=max_inference_batch_size,
                max_new_tokens=max_new_tokens,
                attn_impl=attn_impl,
                asr_init_kwargs_json=asr_init_kwargs_json,
                aligner_init_kwargs_json=aligner_init_kwargs_json,
                transcribe_kwargs_json=transcribe_kwargs_json,
                quiet_transformers=quiet_transformers,
                vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
                vllm_cuda_visible_devices=vllm_cuda_visible_devices,
                vllm_kwargs_json=vllm_kwargs_json,
                debug_traceback=debug_traceback,
                vad_enabled=vad_enabled,
                vad_backend=vad_backend,
                vad_source=vad_source,
                vad_local_repo_dir=vad_local_repo_dir,
                vad_target_sr=vad_target_sr,
                vad_threshold=vad_threshold,
                vad_min_speech_ms=vad_min_speech_ms,
                vad_max_speech_s=vad_max_speech_s,
                vad_min_silence_ms=vad_min_silence_ms,
                vad_speech_pad_ms=vad_speech_pad_ms,
                vad_window_size_samples=vad_window_size_samples,
                vad_merge_gap_ms=vad_merge_gap_ms,
                vad_min_segment_ms=vad_min_segment_ms,
                vad_kwargs_json=vad_kwargs_json,
                max_chars_per_line=max_chars_per_line,
                gap_break_s=gap_break_s,
                break_on_punct=break_on_punct,
                context_prompt=context_prompt,
                language=language,
                torch_threads=torch_threads,
                touch_album_root_mtime=touch_album_root_mtime,
            )
            # Force-refresh UI immediately (avoid relying on Timer if page has been open long).
            shtml, log_txt, files = _poll_job_state()
            # Always surface the action result at the top (log tail may still show old content).
            log_txt = (msg + "\n") + (log_txt or "")
            if not shtml:
                # If state hasn't been written yet, still show a deterministic "starting" status.
                shtml = (
                    "<div style='display:flex;flex-direction:column;gap:6px;'>"
                    "<div style='display:flex;justify-content:space-between;gap:12px;'>"
                    "<div><b>当前音频</b>：启动中… | 0%</div>"
                    "<div><b>0%</b></div></div>"
                    "<div style='width:100%;height:12px;border:1px solid #e5e7eb;border-radius:999px;overflow:hidden;background:#f3f4f6;'>"
                    "<div style='height:100%;width:0%;background:#f97316;transition:width 0.15s linear;'></div>"
                    "</div></div>"
                )
            return shtml, log_txt, files

        run_evt = run_btn.click(
            fn=_start_detached,
            inputs=[
                input_mode,
                input_dir,
                recursive,
                exts_csv,
                uploads,
                output_format,
                output_dir_mode,
                custom_output_dir,
                overwrite,
                backend,
                asr_checkpoint,
                aligner_checkpoint,
                device_map,
                dtype,
                max_inference_batch_size,
                max_new_tokens,
                attn_impl,
                asr_init_kwargs_json,
                aligner_init_kwargs_json,
                transcribe_kwargs_json,
                quiet_transformers,
                vllm_gpu_memory_utilization,
                vllm_cuda_visible_devices,
                vllm_kwargs_json,
                debug_traceback,
                vad_enabled,
                vad_backend,
                vad_source,
                vad_local_repo_dir,
                vad_target_sr,
                vad_threshold,
                vad_min_speech_ms,
                vad_max_speech_s,
                vad_min_silence_ms,
                vad_speech_pad_ms,
                vad_window_size_samples,
                vad_merge_gap_ms,
                vad_min_segment_ms,
                vad_kwargs_json,
                max_chars_per_line,
                gap_break_s,
                break_on_punct,
                context_prompt,
                language,
                torch_threads,
                touch_album_root_mtime,
            ],
            outputs=[status, log, out_files],
            queue=False,
        )

        def _cancel_now():
            msg = detached_cancel_job()
            # Hard reset UI immediately. Also always show the cancel message.
            return "", (msg + "\n"), []

        cancel_btn.click(
            fn=_cancel_now,
            inputs=[],
            outputs=[status, log, out_files],
            queue=False,
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "7860")))
    parser.add_argument("--share", action="store_true", default=False)
    args = parser.parse_args()

    demo = build_ui()
    demo.queue()
    # Allow gr.Files to serve outputs from mounted OUTPUT_DIR (e.g. /output).
    allowed = []
    out_dir = (os.getenv("OUTPUT_DIR") or "").strip()
    if out_dir:
        allowed.append(out_dir)
    # User preference: on container start/restart, clear any previous job log/state.
    try:
        js_clear_job()
    except Exception:
        pass
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, allowed_paths=allowed or None)


if __name__ == "__main__":
    main()

