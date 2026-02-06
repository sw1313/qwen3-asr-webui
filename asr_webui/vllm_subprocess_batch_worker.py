from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any


def vllm_worker_batch(
    q: Any,
    cancel_ev: Any,
    *,
    files: list[str],
    start_idx0: int = 0,
    asr_cfg_d: dict,
    lang_s: str | None,
    cap_cfg_d: dict,
    transcribe_kwargs_obj: dict | None,
    vad_cfg_d: dict,
    output_dir_mode_s: str,
    custom_output_dir_s: str | None,
    overwrite_b: bool,
    default_out_dir_s: str,
) -> None:
    """
    Run a whole batch inside ONE vLLM process to avoid VRAM growth / engine leftovers.

    Messages via `q`:
      - ("init", None)
      - ("file_start", (idx1, total, filename))
      - ("progress", pct_int_0_100)
      - ("file_done", (idx1, total, log_line, out_path_str_or_None))
      - ("file_error", (idx1, total, message))
      - ("cancelled", None)
      - ("done", None)
    """
    # Put this worker in its own process group/session so parent can kill the whole tree
    # (including vLLM EngineCore) on cancel.
    try:
        if os.name == "posix":
            os.setsid()
    except Exception:
        pass

    # Local-first: if checkpoint is local dir with safetensors, force offline.
    try:
        ckpt = str(asr_cfg_d.get("asr_checkpoint", "") or "").strip()
        p = Path(ckpt).expanduser()
        if p.exists() and p.is_dir():
            has_safetensors = any(p.glob("*.safetensors")) or (p / "model.safetensors.index.json").exists()
            if has_safetensors:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    except Exception:
        pass

    # Filter noisy vLLM hub-probe logs for local paths.
    class _HideVllmLocalPathSafetensorsProbe(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
            try:
                msg = record.getMessage()
            except Exception:
                return True
            if "Error retrieving safetensors" in msg and "Repo id must be in the form" in msg:
                return False
            return True

    for name in (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.model_loader",
        "vllm.model_executor.model_loader.repo_utils",
    ):
        try:
            logging.getLogger(name).addFilter(_HideVllmLocalPathSafetensorsProbe())
        except Exception:
            pass

    q.put(("init", None))

    try:
        from asr_webui.batch_generate import CaptionConfig, generate_for_one_audio, write_output
        from asr_webui.file_utils import resolve_output_dir
        from asr_webui.qwen_runner import ASRConfig, unload_model
        from asr_webui.vad import VadConfig

        asr_cfg = ASRConfig(**asr_cfg_d)
        cap_cfg = CaptionConfig(**cap_cfg_d)
        vad_cfg = VadConfig(**vad_cfg_d)

        total = len(files)
        start_idx0_n = max(0, min(int(start_idx0 or 0), total))
        for i0, f in enumerate(files[start_idx0_n:], start=start_idx0_n):
            if cancel_ev is not None and cancel_ev.is_set():
                q.put(("cancelled", None))
                break

            ap = Path(f)
            q.put(("file_start", (i0 + 1, total, ap.name)))
            cur_pct = 0

            def _progress_cb(pct: int, _stage: str = "") -> None:
                nonlocal cur_pct
                try:
                    pct_i = int(pct)
                except Exception:
                    return
                if pct_i <= cur_pct:
                    return
                cur_pct = max(0, min(100, pct_i))
                q.put(("progress", cur_pct))

            try:
                log_line, content, ext = generate_for_one_audio(
                    ap,
                    asr_cfg=asr_cfg,
                    language=lang_s,
                    caption_cfg=cap_cfg,
                    transcribe_kwargs=transcribe_kwargs_obj,
                    vad_cfg=vad_cfg,
                    progress_cb=_progress_cb,
                )
                out_path_s = None
                if ext and not (cancel_ev is not None and cancel_ev.is_set()):
                    out_dir = resolve_output_dir(
                        input_audio_path=ap,
                        mode=output_dir_mode_s,
                        custom_dir=custom_output_dir_s,
                        default_output_dir=Path(default_out_dir_s),
                    )
                    outp = write_output(
                        audio_path=ap,
                        content=content,
                        ext=ext,
                        output_dir=out_dir,
                        overwrite=overwrite_b,
                    )
                    out_path_s = str(outp)
                _progress_cb(100, "done")
                q.put(("file_done", (i0 + 1, total, log_line, out_path_s)))
            except Exception as e:
                q.put(("file_error", (i0 + 1, total, str(e))))

        q.put(("done", None))
    except Exception as e:
        q.put(("file_error", (0, len(files), str(e))))
    finally:
        # Best-effort VRAM cleanup.
        try:
            from asr_webui.qwen_runner import unload_model

            unload_model()
        except Exception:
            pass
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

