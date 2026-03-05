"""
VibeVoice-ASR runner: manages a vLLM serve subprocess and calls the OpenAI-compatible API.

Official deployment reference:
  https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-vllm-asr.md
  https://github.com/microsoft/VibeVoice/blob/main/vllm_plugin/scripts/start_server.py
"""
from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from .subtitle_formats import Segment

logger = logging.getLogger(__name__)

_VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".ts"}


def _guess_mime(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".mp4": "video/mp4",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".opus": "audio/ogg",
    }.get(ext, "application/octet-stream")


def _is_video(p: Path) -> bool:
    return p.suffix.lower() in _VIDEO_EXTS


def _extract_audio(video: Path) -> Path:
    fd, tmp = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video), "-vn", "-acodec", "libmp3lame", "-q:a", "2", tmp],
        check=True, capture_output=True,
    )
    return Path(tmp)


def _audio_duration_ffprobe(p: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(p)],
        stderr=subprocess.DEVNULL, text=True,
    ).strip()
    return float(out)


@dataclass
class VibeVoiceConfig:
    model_path: str = ""
    vllm_port: int = 8199
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 4096
    max_num_seqs: int = 4
    dtype: str = "bfloat16"
    tensor_parallel_size: int = 1
    hotwords: str = ""
    auto_recover: bool = False
    enforce_eager: bool = True
    quantization: str = ""
    load_format: str = ""


_VLLM_LOG_PATH = Path("/tmp/vibevoice_vllm_serve.log")
_LAUNCHER_PATH = Path("/tmp/_vibevoice_vllm_launcher.py")

_LAUNCHER_CODE = r'''#!/usr/bin/env python3
"""
Launcher for VibeVoice vLLM serve.

Ensures the vllm_plugin package is importable (it ships with VibeVoice and
registers the model via vllm.general_plugins entry points), restores any
vLLM source files corrupted by earlier patch attempts, sets PYTHONPATH so
that child subprocesses can also find vllm_plugin, then starts vLLM serve.
"""
import sys, os, importlib, importlib.util, site, glob, shutil, subprocess as _sp


def _ensure_vllm_plugin():
    """Ensure vllm_plugin is importable."""
    try:
        import vllm_plugin
        print(f"[launcher] vllm_plugin OK: {vllm_plugin.__file__}", flush=True)
        return True
    except ImportError:
        pass

    search = []
    try:
        search.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        search.append(site.getusersitepackages())
    except Exception:
        pass
    for sp in search:
        vp = os.path.join(sp, "vllm_plugin")
        if os.path.isdir(vp) and os.path.isfile(os.path.join(vp, "__init__.py")):
            if sp not in sys.path:
                sys.path.insert(0, sp)
            os.environ["PYTHONPATH"] = sp + os.pathsep + os.environ.get("PYTHONPATH", "")
            print(f"[launcher] found vllm_plugin at {vp}, added to path", flush=True)
            try:
                import vllm_plugin
                return True
            except ImportError:
                pass

    print("[launcher] vllm_plugin not found, installing from VibeVoice repo...", flush=True)
    try:
        _sp.check_call(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--no-cache-dir",
             "git+https://github.com/microsoft/VibeVoice.git"],
            timeout=180,
        )
        importlib.invalidate_caches()
        import vllm_plugin
        print(f"[launcher] vllm_plugin installed: {vllm_plugin.__file__}", flush=True)
        return True
    except Exception as e:
        print(f"[launcher] pip install failed: {e}", flush=True)

    return False


def _restore_patched_files():
    """Restore vLLM files corrupted by earlier patch attempts."""
    try:
        spec = importlib.util.find_spec("vllm")
        if not spec or not spec.origin:
            return
        vd = os.path.dirname(spec.origin)
        for rel in ["config/model.py", "engine/arg_utils.py",
                     "transformers_utils/model_arch_config_convertor.py"]:
            fp = os.path.join(vd, rel)
            bak = fp + ".vvbak"
            if os.path.isfile(bak):
                shutil.copy2(bak, fp)
                parent = os.path.dirname(fp)
                base = os.path.basename(fp).replace(".py", ".")
                cache = os.path.join(parent, "__pycache__")
                if os.path.isdir(cache):
                    for fn in os.listdir(cache):
                        if fn.startswith(base):
                            try:
                                os.remove(os.path.join(cache, fn))
                            except Exception:
                                pass
                print(f"[launcher] restored: {fp}", flush=True)
    except Exception as e:
        print(f"[launcher] restore error: {e}", flush=True)


def _find_model_dir():
    for i, a in enumerate(sys.argv):
        if a == "serve" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


_KNOWN_ARCH_ALIASES = {
    "VibeVoiceASRForConditionalGeneration": "VibeVoiceForASRTraining",
}


def _fix_model_config_arch(model_dir):
    """Rewrite architectures in config.json so all processes see a registered name."""
    import json
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        archs = cfg.get("architectures", [])
        changed = False
        for i, a in enumerate(archs):
            if a in _KNOWN_ARCH_ALIASES:
                print(f"[launcher] rewriting architecture {a!r} -> {_KNOWN_ARCH_ALIASES[a]!r}", flush=True)
                archs[i] = _KNOWN_ARCH_ALIASES[a]
                changed = True
        if changed:
            cfg["architectures"] = archs
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            print(f"[launcher] config.json updated in {model_dir}", flush=True)
    except Exception as e:
        print(f"[launcher] config.json arch fix failed: {e}", flush=True)


def _patch_model_for_bnb(model_dir):
    """Add packed_modules_mapping to VibeVoiceForCausalLM if model uses bitsandbytes."""
    import json
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        qcfg = cfg.get("quantization_config", {})
        if qcfg.get("quant_method") != "bitsandbytes":
            return
    except Exception:
        return
    try:
        import vllm_plugin.model as _vm
        src_path = _vm.__file__
        with open(src_path, "r", encoding="utf-8") as f:
            src = f.read()
        if "packed_modules_mapping" in src:
            print("[launcher] packed_modules_mapping already in vllm_plugin/model.py", flush=True)
            return
        patch = (
            "\n\n# --- BitsAndBytes support (auto-patched by launcher) ---\n"
            "if not hasattr(VibeVoiceForCausalLM, 'packed_modules_mapping'):\n"
            "    VibeVoiceForCausalLM.packed_modules_mapping = {\n"
            '        "qkv_proj": ["q_proj", "k_proj", "v_proj"],\n'
            '        "gate_up_proj": ["gate_proj", "up_proj"],\n'
            "    }\n"
        )
        with open(src_path, "a", encoding="utf-8") as f:
            f.write(patch)
        cache_dir = os.path.join(os.path.dirname(src_path), "__pycache__")
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
        print("[launcher] patched VibeVoiceForCausalLM with packed_modules_mapping", flush=True)
    except Exception as e:
        print(f"[launcher] bnb model patch failed: {e}", flush=True)


def _generate_tokenizer(model_dir):
    """Generate tokenizer files if missing."""
    if os.path.isfile(os.path.join(model_dir, "tokenizer.json")):
        print(f"[launcher] tokenizer.json already present in {model_dir}", flush=True)
        return
    print(f"[launcher] tokenizer.json not found in {model_dir}, generating...", flush=True)
    tok_env = os.environ.copy()
    tok_env.pop("HF_HUB_OFFLINE", None)
    tok_env.pop("TRANSFORMERS_OFFLINE", None)
    tok_env["HF_HUB_OFFLINE"] = "0"
    try:
        _sp.check_call(
            [sys.executable, "-m", "vllm_plugin.tools.generate_tokenizer_files",
             "--output", model_dir],
            timeout=180,
            env=tok_env,
        )
        print("[launcher] tokenizer files generated OK", flush=True)
    except Exception as e:
        print(f"[launcher] tokenizer generation failed: {e}", flush=True)
        print(
            "[launcher] HINT: The tool downloads vocab.json from Qwen/Qwen2.5-7B.\n"
            "  If no internet access, run this ONCE on a machine with network:\n"
            f"    python -m vllm_plugin.tools.generate_tokenizer_files --output {model_dir}\n"
            "  Then copy the generated tokenizer files into the model directory.",
            flush=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    # Step 1
    if not _ensure_vllm_plugin():
        print(
            "[launcher] FATAL: vllm_plugin not available.\n"
            "  VibeVoice-ASR requires it for vLLM integration.\n"
            "  Please rebuild Docker image with --build-arg INSTALL_VLLM=1\n"
            "  or ensure network access so the launcher can pip-install it.",
            flush=True,
        )
        sys.exit(1)

    # Step 2
    _restore_patched_files()

    # Step 3: Clean up old .pth files
    try:
        for sp in (site.getsitepackages() if hasattr(site, "getsitepackages") else []):
            for pattern in ["_vv_autoregister.*", "_vv_site_register.*"]:
                for f in glob.glob(os.path.join(sp, pattern)):
                    try:
                        os.remove(f)
                    except Exception:
                        pass
    except Exception:
        pass

    # Step 4: PYTHONPATH
    try:
        import vllm_plugin as _vvp
        _vvp_parent = os.path.dirname(os.path.dirname(_vvp.__file__))
        _pp = os.environ.get("PYTHONPATH", "")
        if _vvp_parent not in _pp:
            os.environ["PYTHONPATH"] = (_vvp_parent + os.pathsep + _pp) if _pp else _vvp_parent
        print(f"[launcher] PYTHONPATH includes {_vvp_parent}", flush=True)
    except Exception as e:
        print(f"[launcher] PYTHONPATH setup: {e}", flush=True)

    # Step 5: Diagnostic
    try:
        from importlib.metadata import entry_points as _ep_fn
        _vllm_eps = []
        _all_eps = _ep_fn()
        if hasattr(_all_eps, "select"):
            _vllm_eps = list(_all_eps.select(group="vllm.general_plugins"))
        elif isinstance(_all_eps, dict):
            _vllm_eps = list(_all_eps.get("vllm.general_plugins", []))
        print(f"[launcher] vllm.general_plugins: {[str(e) for e in _vllm_eps]}", flush=True)
    except Exception as e:
        print(f"[launcher] entry_points diagnostic: {e}", flush=True)

    # Step 6: Register extra architecture names (e.g. 4-bit quantized model)
    try:
        from vllm.model_executor.models import ModelRegistry
        from vllm_plugin.model import VibeVoiceForCausalLM
        for _arch in ("VibeVoiceASRForConditionalGeneration",
                       "VibeVoiceForASRTraining", "VibeVoice"):
            try:
                ModelRegistry.register_model(_arch, VibeVoiceForCausalLM)
            except Exception:
                pass
        print("[launcher] extra architectures registered", flush=True)
    except Exception as e:
        print(f"[launcher] arch registration: {e}", flush=True)

    # Step 7: Fix config.json architecture name, patch for BNB, generate tokenizer
    _md = _find_model_dir()
    if _md:
        _fix_model_config_arch(_md)
        _patch_model_for_bnb(_md)
        _generate_tokenizer(_md)

    # Step 8: Start vLLM serve
    print("[launcher] starting vLLM serve...", flush=True)
    from vllm.entrypoints.cli.main import main
    sys.exit(main())
'''


class _VibeVoiceServer:
    """Manages one vLLM serve subprocess for VibeVoice-ASR."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._cfg: VibeVoiceConfig | None = None
        self._ready = False
        self._log_file: Any = None
        self._tail_buf: list[str] = []
        self._reader_thread: threading.Thread | None = None

    @property
    def log_path(self) -> Path:
        return _VLLM_LOG_PATH

    def ensure_running(self, cfg: VibeVoiceConfig, *, timeout_s: float = 300) -> str:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None and self._cfg == cfg:
                return f"http://127.0.0.1:{cfg.vllm_port}"
            self._stop_locked()
            self._start_locked(cfg)
            self._cfg = cfg

        base = f"http://127.0.0.1:{cfg.vllm_port}"
        deadline = time.monotonic() + timeout_s
        last_log_report = time.monotonic()
        while time.monotonic() < deadline:
            # Check if process has already exited (crashed)
            if self._proc is not None and self._proc.poll() is not None:
                rc = self._proc.returncode
                tail = self._get_tail(80)
                raise RuntimeError(
                    f"vLLM serve 进程已退出（exit_code={rc}）。\n"
                    f"--- 最后 {len(tail)} 行日志（完整日志: {_VLLM_LOG_PATH}）---\n"
                    + "\n".join(tail)
                )
            # Periodically log progress so user knows it's still waiting
            now = time.monotonic()
            if now - last_log_report >= 30:
                elapsed = int(now - (deadline - timeout_s))
                tail_1 = self._get_tail(1)
                last_line = tail_1[0] if tail_1 else "(无输出)"
                logger.info(
                    "vLLM serve 启动中… %ds/%ds  最新: %s",
                    elapsed, int(timeout_s), last_line,
                )
                last_log_report = now
            try:
                r = requests.get(f"{base}/health", timeout=2)
                if r.status_code == 200:
                    self._ready = True
                    logger.info("vLLM serve 就绪 (%s)", base)
                    return base
            except Exception:
                pass
            time.sleep(3)
        tail = self._get_tail(30)
        raise RuntimeError(
            f"vLLM serve 未在 {timeout_s}s 内就绪（port={cfg.vllm_port}）。\n"
            f"进程仍在运行但 /health 未返回 200。\n"
            f"--- 最后 {len(tail)} 行日志（完整日志: {_VLLM_LOG_PATH}）---\n"
            + "\n".join(tail)
        )

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _get_tail(self, n: int = 30) -> list[str]:
        return list(self._tail_buf[-n:])

    def _reader_loop(self, pipe: Any) -> None:
        """Background thread: read subprocess stdout line by line, write to log file + ring buffer."""
        try:
            for raw_line in iter(pipe.readline, b""):
                try:
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\n\r")
                except Exception:
                    line = repr(raw_line)
                self._tail_buf.append(line)
                if len(self._tail_buf) > 400:
                    self._tail_buf[:] = self._tail_buf[-300:]
                if self._log_file:
                    try:
                        self._log_file.write(line + "\n")
                        self._log_file.flush()
                    except Exception:
                        pass
        except Exception:
            pass

    def _start_locked(self, cfg: VibeVoiceConfig) -> None:
        _LAUNCHER_PATH.write_text(_LAUNCHER_CODE, encoding="utf-8")

        vllm_args = [
            "serve", cfg.model_path,
            "--served-model-name", "vibevoice",
            "--trust-remote-code",
            "--dtype", cfg.dtype,
            "--max-num-seqs", str(cfg.max_num_seqs),
            "--max-model-len", str(cfg.max_model_len),
            "--gpu-memory-utilization", str(cfg.gpu_memory_utilization),
            "--no-enable-prefix-caching",
            "--enable-chunked-prefill",
            "--chat-template-content-format", "openai",
            "--tensor-parallel-size", str(cfg.tensor_parallel_size),
            "--port", str(cfg.vllm_port),
            "--disable-frontend-multiprocessing",
        ]
        if cfg.enforce_eager:
            vllm_args.append("--enforce-eager")
        if cfg.quantization:
            vllm_args.extend(["--quantization", cfg.quantization])
        if cfg.load_format:
            vllm_args.extend(["--load-format", cfg.load_format])
        import sys
        cmd = [sys.executable, str(_LAUNCHER_PATH)] + vllm_args
        logger.info("Starting vLLM serve via launcher: %s", " ".join(cmd))

        try:
            self._log_file = open(_VLLM_LOG_PATH, "w", encoding="utf-8")
            self._log_file.write(f"CMD: {' '.join(cmd)}\n")
            self._log_file.write(f"TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            self._log_file.write("=" * 60 + "\n")
            self._log_file.flush()
        except Exception:
            self._log_file = None

        self._tail_buf.clear()
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._reader_thread = threading.Thread(
            target=self._reader_loop, args=(self._proc.stdout,), daemon=True
        )
        self._reader_thread.start()
        self._ready = False

    def _stop_locked(self) -> None:
        p = self._proc
        self._proc = None
        self._cfg = None
        self._ready = False
        if self._log_file:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None
        if p is None:
            return
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception:
            try:
                p.terminate()
            except Exception:
                pass
        try:
            p.wait(timeout=10)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass


VIBEVOICE_SERVER = _VibeVoiceServer()


def _build_payload(
    audio_b64: str,
    mime: str,
    duration: float,
    hotwords: str = "",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 32768,
) -> dict:
    show_keys = ["Start time", "End time", "Speaker ID", "Content"]
    if hotwords.strip():
        prompt = (
            f"This is a {duration:.2f} seconds audio, with extra info: {hotwords.strip()}\n\n"
            f"Please transcribe it with these keys: " + ", ".join(show_keys)
        )
    else:
        prompt = (
            f"This is a {duration:.2f} seconds audio, please transcribe it with these keys: "
            + ", ".join(show_keys)
        )
    data_url = f"data:{mime};base64,{audio_b64}"
    return {
        "model": "vibevoice",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that transcribes audio input into text output in JSON format."},
            {"role": "user", "content": [
                {"type": "audio_url", "audio_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ]},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }


def _parse_vibevoice_json(raw: str) -> list[Segment]:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    if not raw.startswith("["):
        idx = raw.find("[")
        if idx >= 0:
            raw = raw[idx:]

    try:
        arr = json.loads(raw)
    except json.JSONDecodeError:
        last = raw.rfind("},")
        if last > 0:
            raw = raw[: last + 1] + "]"
            try:
                arr = json.loads(raw)
            except json.JSONDecodeError:
                return []
        else:
            return []

    segments: list[Segment] = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        start = float(item.get("Start time", item.get("start_time", 0)))
        end = float(item.get("End time", item.get("end_time", 0)))
        text = str(item.get("Content", item.get("content", "")))
        if text.strip():
            segments.append(Segment(start_s=start, end_s=end, text=text.strip()))
    return segments


def _is_bnb_model(model_path: str) -> bool:
    """Check if model uses bitsandbytes quantization from its config.json."""
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.is_file():
        return False
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("quantization_config", {}).get("quant_method") == "bitsandbytes"
    except Exception:
        return False


class _DirectModel:
    """Holds a transformers-loaded VibeVoice ASR model (for bitsandbytes 4-bit)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: Any = None
        self._processor: Any = None
        self._device: str = ""
        self._model_path: str = ""

    def get(self, model_path: str):
        import torch
        with self._lock:
            if self._model is not None and self._model_path == model_path:
                return self._model, self._processor, self._device
            self._unload()
            logger.info("Loading VibeVoice ASR model directly (transformers): %s", model_path)
            from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
            from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration

            self._processor = VibeVoiceASRProcessor.from_pretrained(model_path)
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
            try:
                attn = "flash_attention_2"
                self._model = VibeVoiceASRForConditionalGeneration.from_pretrained(
                    model_path, dtype=dtype, device_map=self._device,
                    attn_implementation=attn, trust_remote_code=True,
                )
            except Exception:
                attn = "sdpa"
                self._model = VibeVoiceASRForConditionalGeneration.from_pretrained(
                    model_path, dtype=dtype, device_map=self._device,
                    attn_implementation=attn, trust_remote_code=True,
                )
            self._model.eval()
            self._model_path = model_path
            logger.info("VibeVoice ASR model loaded on %s (attn=%s)", self._device, attn)
            return self._model, self._processor, self._device

    def unload(self) -> None:
        with self._lock:
            self._unload()

    def _unload(self) -> None:
        if self._model is not None:
            import torch, gc
            del self._model
            self._model = None
            self._processor = None
            self._model_path = ""
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


_DIRECT_MODEL = _DirectModel()


def _transcribe_direct(
    work_path: Path,
    model_path: str,
    hotwords: str = "",
    max_new_tokens: int = 8192,
) -> str:
    """Transcribe using transformers direct inference (for bitsandbytes models)."""
    import torch
    model, processor, device = _DIRECT_MODEL.get(model_path)

    context_info = hotwords.strip() if hotwords else None
    inputs = processor(
        audio=str(work_path),
        return_tensors="pt",
        add_generation_prompt=True,
        context_info=context_info,
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": processor.pad_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
    }
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    return processor.decode(generated_ids, skip_special_tokens=True)


def transcribe_vibevoice(
    audio_path: Path | str,
    *,
    cfg: VibeVoiceConfig,
    hotwords: str = "",
    progress_cb: Any | None = None,
) -> tuple[str, list[Segment]]:
    """
    Transcribe one audio file via VibeVoice.
    Routes to direct transformers inference for bitsandbytes models,
    or vLLM serve API for full-precision models.
    Returns (full_text, segments).
    """
    audio_path = Path(audio_path)

    tmp_audio: Path | None = None
    work_path = audio_path
    if _is_video(audio_path):
        tmp_audio = _extract_audio(audio_path)
        work_path = tmp_audio

    use_direct = _is_bnb_model(cfg.model_path)

    try:
        if use_direct:
            if progress_cb:
                try:
                    progress_cb(10, "transformers 直接推理")
                except Exception:
                    pass
            raw_text = _transcribe_direct(work_path, cfg.model_path, hotwords=hotwords or cfg.hotwords or "")
        else:
            try:
                base_url = VIBEVOICE_SERVER.ensure_running(cfg)
            except Exception:
                if tmp_audio:
                    try:
                        tmp_audio.unlink(missing_ok=True)
                    except Exception:
                        pass
                raise

            try:
                duration = _audio_duration_ffprobe(work_path)
            except Exception:
                duration = 0.0

            with open(work_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")

            mime = _guess_mime(str(work_path))
            hw = hotwords or cfg.hotwords or ""

            if progress_cb:
                try:
                    progress_cb(15, "发送到 VibeVoice API")
                except Exception:
                    pass

            payload = _build_payload(audio_b64, mime, duration, hotwords=hw)
            url = f"{base_url}/v1/chat/completions"

            try:
                resp = requests.post(url, json=payload, timeout=max(300, duration * 5))
                resp.raise_for_status()
            except Exception:
                if tmp_audio:
                    try:
                        tmp_audio.unlink(missing_ok=True)
                    except Exception:
                        pass
                raise

            data = resp.json()
            raw_text = ""
            try:
                raw_text = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                pass
    except Exception:
        if tmp_audio:
            try:
                tmp_audio.unlink(missing_ok=True)
            except Exception:
                pass
        raise

    if progress_cb:
        try:
            progress_cb(80, "解析结果")
        except Exception:
            pass

    segments = _parse_vibevoice_json(raw_text)
    full_text = " ".join(s.text for s in segments)

    if tmp_audio:
        try:
            tmp_audio.unlink(missing_ok=True)
        except Exception:
            pass

    return full_text, segments


def stop_server() -> None:
    VIBEVOICE_SERVER.stop()
    _DIRECT_MODEL.unload()
