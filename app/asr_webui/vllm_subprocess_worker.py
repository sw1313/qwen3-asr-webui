from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any


def vllm_worker_one(
    q: Any,
    cancel_ev: Any,
    *,
    audio_path_s: str,
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
    Run one-audio inference in a separate process so that terminating the process
    fully releases CUDA context / vLLM graphs on cancel.
    Communicates via `q`:
      - ("start", filename)
      - ("done", (log_line, out_path_str_or_None))
      - ("error", message)
      - ("cancelled", None)
    """
    try:
        # Local-first behavior:
        # - If model checkpoint points to a local directory that already contains safetensors,
        #   force offline so vLLM/transformers won't probe HuggingFace Hub (and avoid noisy logs).
        # - If local weights are missing, allow online behavior (user may pass HF repo id).
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

        # vLLM sometimes logs an ERROR when probing HuggingFace Hub metadata for a local path
        # like "/models/xxx". It then falls back to local files and continues successfully.
        # This is noisy but harmless; filter it out to avoid confusing users.
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

        from asr_webui.batch_generate import CaptionConfig, generate_for_one_audio, write_output
        from asr_webui.file_utils import resolve_output_dir
        from asr_webui.qwen_runner import ASRConfig, unload_model
        from asr_webui.vad import VadConfig
        import torch

        if cancel_ev is not None and cancel_ev.is_set():
            q.put(("cancelled", None))
            return

        ap = Path(audio_path_s)
        q.put(("start", ap.name))
        q.put(("progress", 2))

        asr_cfg = ASRConfig(**asr_cfg_d)
        cap_cfg = CaptionConfig(**cap_cfg_d)
        vad_cfg = VadConfig(**vad_cfg_d)

        # stage hint: VAD enabled usually adds pre-processing time.
        if getattr(vad_cfg, "enabled", False):
            q.put(("progress", 10))
        else:
            q.put(("progress", 5))

        log_line, content, ext = generate_for_one_audio(
            ap,
            asr_cfg=asr_cfg,
            language=lang_s,
            caption_cfg=cap_cfg,
            transcribe_kwargs=transcribe_kwargs_obj,
            vad_cfg=vad_cfg,
        )

        q.put(("progress", 92))

        out_path_s = None
        if content and ext and not (cancel_ev is not None and cancel_ev.is_set()):
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

        q.put(("progress", 100))
        q.put(("done", (log_line, out_path_s)))
    except Exception as e:
        q.put(("error", str(e)))
    finally:
        # Best-effort VRAM cleanup inside worker.
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

