from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import multiprocessing as mp

import torch

from .batch_generate import CaptionConfig, generate_for_one_audio, write_output
from .cancel_token import CancelToken
from .file_utils import MEDIA_EXTS_DEFAULT, list_audio_files, resolve_output_dir
from .job_state import append_log as js_append_log
from .job_state import clear_job as js_clear_job
from .job_state import reset_job as js_reset_job
from .job_state import update_state as js_update_state
from .qwen_runner import ASRConfig, unload_model
from .vad import VadConfig, parse_json_dict as parse_vad_json_dict
from .vllm_subprocess_batch_worker import vllm_worker_batch, vllm_worker_queue


_JOB_LOCK = threading.Lock()
_JOB_THREAD: threading.Thread | None = None
_CANCEL_EV = threading.Event()
_CURRENT_JOB_ID: str | None = None

# Keep a local list to avoid importing heavy modules for a simple suffix check.
_VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".ts"}


def _emit(line: str) -> None:
    try:
        js_append_log(line)
    except Exception:
        pass


def _safe_name(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r'[:*?"<>|]', "_", s)
    return s or "output"


def _copy_to_ui(src: Path, *, audio_path: Path, ui_output_dir: Path, ext: str, idx: int, input_mode: str, input_dir: str) -> Path:
    try:
        ui_output_dir.mkdir(parents=True, exist_ok=True)
        # If already under ui_output_dir, return as-is.
        try:
            if src.resolve().is_relative_to(ui_output_dir.resolve()):
                return src
        except Exception:
            pass

        if input_mode == "dir":
            try:
                rel = audio_path.relative_to(Path(input_dir).expanduser())
                parent_parts = [_safe_name(p)[:80] for p in rel.parent.parts]
                dst_dir = ui_output_dir.joinpath(*parent_parts) if parent_parts else ui_output_dir
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / f"{_safe_name(audio_path.stem)}{ext}"
            except Exception:
                dst = ui_output_dir / f"{idx:04d}_{_safe_name(audio_path.stem)}{ext}"
        else:
            dst = ui_output_dir / f"{idx:04d}_{_safe_name(audio_path.stem)}{ext}"

        shutil.copy2(src, dst)
        return dst
    except Exception:
        return src


def _folder_label(*, p: Path, input_mode: str, input_dir: str) -> str:
    if input_mode != "dir":
        return "（上传）"
    try:
        root = Path(input_dir).expanduser()
        rel = p.relative_to(root)
        if len(rel.parts) <= 1:
            return "（根目录）"
        return str(rel.parts[0])
    except Exception:
        return "（其它）"


def _album_root_dir(p: Path, *, input_mode: str, input_dir: str) -> Path | None:
    """
    Album root = immediate child directory under input_dir.
    Example: input_dir=/musics/asmr, file=/musics/asmr/AlbumA/track1.mp3 -> /musics/asmr/AlbumA
    """
    if input_mode != "dir":
        return None
    try:
        root = Path(input_dir).expanduser()
        rel = p.relative_to(root)
        if len(rel.parts) <= 1:
            return None
        return root / rel.parts[0]
    except Exception:
        return None

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


def _get_media_duration_s(p: Path) -> float | None:
    """
    Duration getter used heavily by dedup preprocessing.
    Performance note: prefer in-process `torchaudio.info()` for *audio* to avoid spawning
    tens of thousands of `ffprobe` processes on big libraries. For video, use ffprobe.
    """
    suf = (p.suffix or "").lower()
    is_video = suf in _VIDEO_EXTS

    # 1) For audio: torchaudio first (fast, no subprocess).
    try:
        import torchaudio  # type: ignore

        info = torchaudio.info(str(p))
        if getattr(info, "num_frames", 0) and getattr(info, "sample_rate", 0):
            return float(info.num_frames) / float(info.sample_rate)
    except Exception:
        pass

    # 2) For video (or torchaudio failure): ffprobe fallback.
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        # If we *know* it's video, go straight to ffprobe. If unknown, still try ffprobe as fallback.
        if is_video or True:
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
                pass
    return None


def is_running() -> bool:
    t = _JOB_THREAD
    return bool(t is not None and t.is_alive())


def request_cancel() -> None:
    _CANCEL_EV.set()


def start_job(**params: Any) -> str:
    """
    Start a detached background job. Returns a human-readable message.
    """
    global _JOB_THREAD
    global _CURRENT_JOB_ID
    with _JOB_LOCK:
        if _JOB_THREAD is not None and _JOB_THREAD.is_alive():
            return "已有任务在运行中。"
        _CANCEL_EV.clear()
        job_id = time.strftime("%Y%m%d_%H%M%S")
        _CURRENT_JOB_ID = job_id
        # Reset job state synchronously so UI can refresh immediately after clicking "开始".
        try:
            js_reset_job(job_id=job_id)
            js_append_log("任务启动中…（可关闭页面，稍后回来会自动续显）\n")
            js_update_state(dict(running=True, done=False, cancel_requested=False, progress_pct=0, current_file=None, current_idx=0))
        except Exception:
            pass
        _JOB_THREAD = threading.Thread(target=_run_job, kwargs={**params, "job_id": job_id}, daemon=True)
        _JOB_THREAD.start()
        return "已启动任务（可关闭页面，稍后回来会自动续显）。"


def cancel_job() -> str:
    global _CURRENT_JOB_ID
    request_cancel()
    # Prevent any further writes from the current job thread.
    _CURRENT_JOB_ID = None
    # User preference: clear job log/state immediately on cancel click.
    try:
        js_clear_job()
    except Exception:
        pass
    return "已请求取消。"


def _run_job(**p: Any) -> None:
    """
    Actual job runner.
    """
    job_id = str(p.get("job_id") or time.strftime("%Y%m%d_%H%M%S"))

    def _active() -> bool:
        return bool(_CURRENT_JOB_ID == job_id and (not _CANCEL_EV.is_set()))
    try:
        if _CURRENT_JOB_ID != job_id:
            return
        js_reset_job(job_id=job_id)
    except Exception:
        pass

    # Ensure model is only loaded when job starts.
    try:
        unload_model()
    except Exception:
        pass

    try:
        input_mode = str(p.get("input_mode") or "dir")
        input_dir = str(p.get("input_dir") or "")
        recursive = bool(p.get("recursive", True))
        exts_csv = str(p.get("exts_csv") or "")
        uploads = p.get("uploads")
        output_format = str(p.get("output_format") or "srt")
        output_dir_mode = str(p.get("output_dir_mode") or "output")
        custom_output_dir = str(p.get("custom_output_dir") or "")
        overwrite = bool(p.get("overwrite", False))

        backend = str(p.get("backend") or "transformers")
        asr_checkpoint = str(p.get("asr_checkpoint") or "").strip()
        aligner_checkpoint = str(p.get("aligner_checkpoint") or "").strip()
        device_map = str(p.get("device_map") or "cuda:0")
        dtype = str(p.get("dtype") or "fp16")
        max_inference_batch_size = int(p.get("max_inference_batch_size") or 1)
        max_new_tokens = int(p.get("max_new_tokens") or 2048)
        attn_impl = str(p.get("attn_impl") or "")
        asr_init_kwargs_json = str(p.get("asr_init_kwargs_json") or "{}")
        aligner_init_kwargs_json = str(p.get("aligner_init_kwargs_json") or "{}")
        transcribe_kwargs_json = str(p.get("transcribe_kwargs_json") or "{}")
        hallucination_cfg_json = str(p.get("hallucination_cfg_json") or "{}")
        context_prompt = str(p.get("context_prompt") or "")

        vllm_gpu_memory_utilization = float(p.get("vllm_gpu_memory_utilization") or 0.7)
        vllm_cuda_visible_devices = str(p.get("vllm_cuda_visible_devices") or "").strip()
        vllm_kwargs_json = str(p.get("vllm_kwargs_json") or "{}")

        debug_traceback = bool(p.get("debug_traceback", False))

        vad_enabled = bool(p.get("vad_enabled", False))
        vad_backend = str(p.get("vad_backend") or "silero")
        vad_source = str(p.get("vad_source") or "auto")
        vad_local_repo_dir = str(p.get("vad_local_repo_dir") or "")
        vad_target_sr = int(p.get("vad_target_sr") or 16000)
        vad_threshold = float(p.get("vad_threshold") or 0.5)
        vad_min_speech_ms = int(p.get("vad_min_speech_ms") or 250)
        vad_max_speech_s = float(p.get("vad_max_speech_s") or 60.0)
        vad_min_silence_ms = int(p.get("vad_min_silence_ms") or 100)
        vad_speech_pad_ms = int(p.get("vad_speech_pad_ms") or 30)
        vad_window_size_samples = int(p.get("vad_window_size_samples") or 512)
        vad_merge_gap_ms = int(p.get("vad_merge_gap_ms") or 120)
        vad_min_segment_ms = int(p.get("vad_min_segment_ms") or 300)
        vad_kwargs_json = str(p.get("vad_kwargs_json") or "{}")

        max_chars_per_line = int(p.get("max_chars_per_line") or 40)
        gap_break_s = float(p.get("gap_break_s") or 0.8)
        break_on_punct = bool(p.get("break_on_punct", True))

        language = str(p.get("language") or "")
        torch_threads = int(p.get("torch_threads") or 4)
        touch_album_root_mtime = bool(p.get("touch_album_root_mtime", False))
        dedup_same_name_same_duration = bool(p.get("dedup_same_name_same_duration", False))
        skip_if_subtitle_exists = bool(p.get("skip_if_subtitle_exists", False))
        toolkit_post_process_enabled = bool(p.get("toolkit_post_process_enabled", False))
        toolkit_post_process_threshold = int(p.get("toolkit_post_process_threshold") or 20)

        # Prepare inputs
        if input_mode == "dir":
            exts = [x.strip() for x in (exts_csv or "").split(",") if x.strip()]
            files = list_audio_files(input_dir, recursive=recursive, exts=exts or MEDIA_EXTS_DEFAULT)
        else:
            files = _normalize_uploads(uploads)

        # Stable de-dup: avoid duplicated paths causing duplicated logs/copies.
        try:
            seen_paths: set[str] = set()
            uniq_files: list[Path] = []
            for fp in files:
                try:
                    import os as _os

                    k = _os.path.realpath(str(fp))
                except Exception:
                    k = str(fp)
                if k in seen_paths:
                    continue
                seen_paths.add(k)
                uniq_files.append(fp)
            files = uniq_files
        except Exception:
            pass

        if not files:
            if _active():
                _emit("未找到音视频文件。请检查输入目录/后缀，或上传文件。\n")
                js_update_state(dict(running=False, done=True, progress_pct=0, total=0, current_file=None))
            return

        backend_n = backend.strip().lower()
        if device_map.strip().lower().startswith("cuda") and not torch.cuda.is_available():
            _emit("当前环境 torch 不支持 CUDA，但你选择了 cuda device_map。\n")
            js_update_state(dict(running=False, done=True, last_error="cuda not available"))
            return

        # Parse json kwargs
        def parse_json_dict(name: str, s: str) -> dict:
            s = (s or "").strip()
            if not s:
                return {}
            obj = json.loads(s)
            if not isinstance(obj, dict):
                raise ValueError(f"{name} 必须是 dict")
            return obj

        import json

        asr_init_kwargs = parse_json_dict("ASR init kwargs", asr_init_kwargs_json)
        aligner_init_kwargs = parse_json_dict("Aligner init kwargs", aligner_init_kwargs_json)
        transcribe_kwargs = parse_json_dict("transcribe kwargs", transcribe_kwargs_json)
        vad_kwargs = parse_vad_json_dict("VAD kwargs", vad_kwargs_json)
        vllm_kwargs = parse_json_dict("vLLM kwargs", vllm_kwargs_json)
        hallucination_cfg_obj = parse_json_dict("hallucination cfg", hallucination_cfg_json)

        # Only keep `context` and merge UI context.
        transcribe_kwargs = {k: v for k, v in transcribe_kwargs.items() if k == "context"}
        if context_prompt.strip():
            transcribe_kwargs["context"] = context_prompt.strip()

        # vLLM defaults
        if backend_n == "vllm":
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            if "max_model_len" not in vllm_kwargs and "max_seq_len" not in vllm_kwargs:
                vllm_kwargs["max_model_len"] = int(os.getenv("VLLM_MAX_MODEL_LEN", "12288"))

        asr_cfg = ASRConfig(
            backend=("vllm" if backend_n == "vllm" else "transformers"),
            asr_checkpoint=asr_checkpoint,
            aligner_checkpoint=aligner_checkpoint,
            device_map=device_map or "cuda:0",
            dtype=dtype,  # type: ignore[arg-type]
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
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
            cuda_visible_devices=(vllm_cuda_visible_devices or None),
        )
        cap_cfg = CaptionConfig(
            output_format="srt" if output_format == "srt" else "lrc",
            max_chars_per_line=max_chars_per_line,
            gap_break_s=gap_break_s,
            break_on_punct=break_on_punct,
        )
        vad_cfg = VadConfig(
            enabled=vad_enabled,
            backend="silero" if (vad_backend or "silero") == "silero" else "silero",
            source=(vad_source or "auto"),
            local_repo_dir=(vad_local_repo_dir.strip() or None),
            target_sr=vad_target_sr,
            threshold=vad_threshold,
            min_speech_duration_ms=vad_min_speech_ms,
            max_speech_duration_s=vad_max_speech_s,
            min_silence_duration_ms=vad_min_silence_ms,
            speech_pad_ms=vad_speech_pad_ms,
            window_size_samples=vad_window_size_samples,
            merge_gap_ms=vad_merge_gap_ms,
            min_segment_ms=vad_min_segment_ms,
            vad_kwargs=vad_kwargs or None,
        )

        n = len(files)
        try:
            import datetime as _dt

            mt = Path(__file__).stat().st_mtime
            ts = _dt.datetime.fromtimestamp(mt).strftime("%Y-%m-%d %H:%M:%S")
            _emit(f"[版本] detached_runner.py mtime={ts}\n")
        except Exception:
            pass
        _emit(f"共 {n} 个音频，开始批量生成…\n")
        js_update_state(dict(running=True, done=False, total=n, progress_pct=0, current_file=None, current_idx=0))

        ui_output_dir = Path(os.getenv("WEBUI_OUTPUT_DIR") or os.getenv("OUTPUT_DIR") or str(Path.cwd() / "output"))
        default_out_dir = Path(os.getenv("OUTPUT_DIR", str(Path.cwd() / "output")))

        output_files: dict[str, str] = {}
        seen_folders: set[str] = set()
        touched_albums: set[str] = set()

        ext_primary = ".srt" if output_format == "srt" else ".lrc"
        ext_alt = ".lrc" if ext_primary == ".srt" else ".srt"

        def _out_dir_for(fp: Path) -> Path:
            return resolve_output_dir(
                input_audio_path=fp,
                mode=output_dir_mode,
                custom_dir=(custom_output_dir.strip() or None),
                default_output_dir=default_out_dir,
            )

        def _find_existing_subtitle(fp: Path) -> Path | None:
            """
            User expectation: "同路径" = the media file's own directory.
            Only check fp.parent (same directory as media) for .srt/.lrc.
            """
            # Same directory as the media file.
            try:
                d0 = fp.parent
                stem = fp.stem
                # Check both formats regardless of current output_format, and be case-tolerant.
                for suf in (".srt", ".SRT", ".lrc", ".LRC"):
                    p = d0 / f"{stem}{suf}"
                    if p.exists():
                        return p
            except Exception:
                pass
            return None

        def _copy_subtitle(src: Path, dst: Path) -> Path | None:
            try:
                if not overwrite and dst.exists():
                    return None
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    if src.resolve() == dst.resolve():
                        return dst
                except Exception:
                    pass
                shutil.copy2(src, dst)
                return dst
            except Exception:
                return None

        def _same_dir_sub_path(fp: Path, suffix: str) -> Path:
            return fp.parent / f"{fp.stem}{suffix}"

        # Folder-by-folder execution helpers (user preference):
        # - Log one folder at a time
        # - Apply "existing subtitle skip" and "same-name same-duration dedup" within that folder
        # - If all files in a folder are skipped, jump to next folder immediately.

        # Global index mapping for consistent UI status (idx/total).
        orig_idx_of: dict[str, int] = {str(_fp): int(i) for i, _fp in enumerate(files, start=1)}

        def _iter_folder_batches() -> list[tuple[str, list[Path]]]:
            if input_mode != "dir":
                return [("（上传）", list(files))]
            root = Path(input_dir).expanduser()
            buckets: dict[str, list[Path]] = {}
            for fp in files:
                ar = _album_root_dir(fp, input_mode=input_mode, input_dir=input_dir)
                if ar is None:
                    buckets.setdefault("（根目录）", []).append(fp)
                    continue
                try:
                    rel = ar.relative_to(root)
                    label = str(rel.parts[0]) if rel.parts else "（根目录）"
                except Exception:
                    label = str(ar.name or "（其它）")
                buckets.setdefault(label, []).append(fp)
            return [(k, buckets[k]) for k in sorted(buckets.keys())]

        def _fmt_media(fp: Path) -> str:
            """
            For clearer logs when the album contains multiple same-named files in subfolders.
            """
            if input_mode != "dir":
                return fp.name
            ar = _album_root_dir(fp, input_mode=input_mode, input_dir=input_dir)
            if ar is None:
                return fp.name
            try:
                rel = fp.relative_to(ar)
                return rel.as_posix()
            except Exception:
                return fp.name

        def _prepare_folder(
            folder_files: list[Path],
            *,
            existing_subtitle_pre: dict[str, str] | None = None,
        ) -> tuple[list[Path], dict[Path, list[Path]], dict[str, str], dict[Path, Path], set[Path], list[tuple[Path, Path, Path]]]:
            """
            Returns:
            - worker_files: canonical files to transcribe in this folder
            - dups_of: canon -> [duplicates]
            - existing_subtitle: str(fp) -> str(existing_sub_path)
            - canonical_of: dup -> canon
            - done_media: media files that should NOT be transcribed (existing subtitle wins)
            - precopy_actions: (src_media_with_subtitle, src_subtitle_path, dst_media_to_receive_copy)
            """
            # 1) Existing subtitle map (only when overwrite is off).
            existing_subtitle: dict[str, str] = dict(existing_subtitle_pre or {})
            if skip_if_subtitle_exists:
                for fp in folder_files:
                    if _CANCEL_EV.is_set() or (not _active()):
                        break
                    if str(fp) in existing_subtitle:
                        continue
                    ex = _find_existing_subtitle(fp)
                    if ex is not None:
                        existing_subtitle[str(fp)] = str(ex)

            # User preference: existing-subtitle skip happens first. If all files already have subtitles,
            # skip this folder without any dedup scanning.
            done_media: set[Path] = set()
            precopy_actions: list[tuple[Path, Path, Path]] = []
            if skip_if_subtitle_exists and folder_files and len(existing_subtitle) >= len(folder_files):
                done_media.update(folder_files)
                return [], {}, existing_subtitle, {}, done_media, precopy_actions

            # 2) Dedup within folder by (stem + duration).
            canonical_of: dict[Path, Path] = {}
            dups_of: dict[Path, list[Path]] = {}
            if dedup_same_name_same_duration and input_mode == "dir":
                groups: dict[str, list[Path]] = {}
                for fp in folder_files:
                    if _CANCEL_EV.is_set() or (not _active()):
                        break
                    groups.setdefault((fp.stem or "").lower(), []).append(fp)

                pref_order = [
                    ".flac",
                    ".wav",
                    ".m4a",
                    ".aac",
                    ".ogg",
                    ".mp3",
                    ".wma",
                    ".mp4",
                    ".mkv",
                    ".mov",
                    ".webm",
                    ".avi",
                    ".m4v",
                    ".ts",
                ]
                pref_rank = {ext: i for i, ext in enumerate(pref_order)}

                def _choose_canon(cands: list[Path]) -> Path:
                    def _score(x: Path) -> tuple[int, int]:
                        r = pref_rank.get(x.suffix.lower(), 999)
                        try:
                            sz = int(x.stat().st_size)
                        except Exception:
                            sz = 0
                        return (r, -sz)

                    return min(cands, key=_score)

                dur_cache: dict[str, float | None] = {}

                def _duration(fp: Path) -> float | None:
                    k = str(fp)
                    if k in dur_cache:
                        return dur_cache[k]
                    d = _get_media_duration_s(fp)
                    dur_cache[k] = d
                    return d

                tol_s = float(os.getenv("WEBUI_DEDUP_DURATION_TOL_S", "0.2"))
                processed = 0
                for stem_k, items in groups.items():
                    processed += 1
                    if processed % 200 == 0:
                        try:
                            if _active():
                                _emit(f"[预处理] 去重扫描中…（{processed}/{len(groups)} stem）\n")
                        except Exception:
                            pass
                    if _CANCEL_EV.is_set() or (not _active()):
                        break
                    if len(items) <= 1:
                        continue
                    infos: list[tuple[Path, float]] = []
                    for fp in items:
                        if _CANCEL_EV.is_set() or (not _active()):
                            break
                        d = _duration(fp)
                        if d is None:
                            continue
                        infos.append((fp, float(d)))
                    if len(infos) <= 1:
                        continue
                    infos.sort(key=lambda t: t[1])
                    cluster: list[tuple[Path, float]] = [infos[0]]
                    clusters: list[list[tuple[Path, float]]] = []
                    for fp, d in infos[1:]:
                        if abs(d - cluster[-1][1]) <= tol_s:
                            cluster.append((fp, d))
                        else:
                            clusters.append(cluster)
                            cluster = [(fp, d)]
                    clusters.append(cluster)
                    for cl in clusters:
                        if len(cl) <= 1:
                            continue
                        cl_paths = [x for x, _d in cl]

                        # Existing subtitle has priority over dedup: if any member already has subtitle,
                        # skip transcription for the whole cluster and copy that subtitle to others.
                        src_media: Path | None = None
                        src_sub: Path | None = None
                        for fp in cl_paths:
                            ex_s = existing_subtitle.get(str(fp))
                            if ex_s:
                                src_media = fp
                                src_sub = Path(ex_s)
                                break
                        if src_media is not None and src_sub is not None:
                            for fp in cl_paths:
                                done_media.add(fp)
                                if fp == src_media:
                                    continue
                                precopy_actions.append((src_media, src_sub, fp))
                            continue

                        # Otherwise: normal dedup => choose one canonical to transcribe.
                        canon = _choose_canon(cl_paths)
                        for fp in cl_paths:
                            if fp == canon:
                                continue
                            canonical_of[fp] = canon
                            dups_of.setdefault(canon, []).append(fp)

            # 3) Canonical transcribe list (skip duplicates and existing subtitles).
            worker_files: list[Path] = []
            for fp in folder_files:
                if _CANCEL_EV.is_set() or (not _active()):
                    break
                if fp in canonical_of:
                    continue
                if existing_subtitle.get(str(fp)):
                    done_media.add(fp)
                    continue
                if fp in done_media:
                    continue
                worker_files.append(fp)

            return worker_files, dups_of, existing_subtitle, canonical_of, done_media, precopy_actions

        def _touch_album_root(fp: Path) -> None:
            if not touch_album_root_mtime:
                return
            ar = _album_root_dir(fp, input_mode=input_mode, input_dir=input_dir)
            if ar is None:
                return
            key = str(ar)
            # Touch at most once per album to reduce NAS metadata churn.
            if key in touched_albums:
                return
            try:
                os.utime(ar, None)
                touched_albums.add(key)
            except Exception:
                pass

        def maybe_folder_log(fp: Path) -> None:
            folder = _folder_label(p=fp, input_mode=input_mode, input_dir=input_dir)
            if folder not in seen_folders:
                seen_folders.add(folder)
                _emit(f"[处理文件夹] {folder}\n")
        # Note: "跳过已存在字幕" 和 "同名同长度去重" 都按文件夹批次处理（见后续 backend 分支）。

        if backend_n == "vllm":
            ctx = mp.get_context("spawn")
            gpu_util = float(vllm_gpu_memory_utilization)
            try:
                rtf_est = float(os.getenv("ASR_PROGRESS_RTF", "0.18"))
            except Exception:
                rtf_est = 0.18

            file_t0: float | None = None
            file_expected_s: float | None = None
            base_pct: int = 0
            last_emit = 0.0

            # Start ONE persistent worker; main thread feeds folder batches.
            def _start_queue_worker():
                mp_cancel = ctx.Event()
                cmd_q = ctx.Queue()
                out_q = ctx.Queue()
                cfg_d = asdict(asr_cfg)
                try:
                    vk = dict(cfg_d.get("vllm_kwargs") or {})
                    vk["gpu_memory_utilization"] = float(gpu_util)
                    cfg_d["vllm_kwargs"] = vk
                except Exception:
                    pass
                proc = ctx.Process(
                    target=vllm_worker_queue,
                    args=(cmd_q, out_q, mp_cancel),
                    kwargs=dict(
                        asr_cfg_d=cfg_d,
                        lang_s=_normalize_language(language),
                        cap_cfg_d=asdict(cap_cfg),
                        transcribe_kwargs_obj=(transcribe_kwargs or None),
                        vad_cfg_d=asdict(vad_cfg),
                        hallucination_cfg_obj=(hallucination_cfg_obj or None),
                        toolkit_post_process_enabled_b=bool(toolkit_post_process_enabled),
                        toolkit_post_process_threshold_i=int(toolkit_post_process_threshold),
                        output_dir_mode_s=output_dir_mode,
                        custom_output_dir_s=(custom_output_dir.strip() or None),
                        overwrite_b=bool(overwrite),
                        default_out_dir_s=str(default_out_dir),
                    ),
                    daemon=False,
                )
                proc.start()
                return proc, cmd_q, out_q, mp_cancel

            proc, cmd_q, out_q, mp_cancel = _start_queue_worker()
            ct = CancelToken()
            ct.attach_worker(proc=proc, mp_cancel=mp_cancel)

            batch_id = 0
            for folder_label, folder_files in _iter_folder_batches():
                if _CANCEL_EV.is_set() or (not _active()):
                    _emit("已取消：停止后续任务。\n")
                    try:
                        ct.terminate_worker()
                    except Exception:
                        pass
                    if _active():
                        js_update_state(dict(running=False, cancel_requested=True, done=True, output_files=list(output_files.values())))
                    return
                if not folder_files:
                    continue

                _emit(f"[处理文件夹] {folder_label}\n")
                existing_subtitle_pre: dict[str, str] = {}
                _emit(f"[开关] 跳过已存在字幕={'ON' if skip_if_subtitle_exists else 'OFF'} 覆盖={'ON' if overwrite else 'OFF'} 去重={'ON' if dedup_same_name_same_duration else 'OFF'}\n")
                if skip_if_subtitle_exists:
                    for fp in folder_files:
                        ex = _find_existing_subtitle(fp)
                        if ex is not None:
                            existing_subtitle_pre[str(fp)] = str(ex)
                    found_n = int(len(existing_subtitle_pre))
                    total_n = int(len(folder_files))
                    if found_n >= total_n and total_n > 0:
                        _emit(f"[检查字幕] 已找到 {found_n}/{total_n}，跳过去重与转录。\n")
                        _emit(f"[跳过文件夹] {folder_label}：全部已存在字幕。\n")
                        continue
                    else:
                        miss = [fp for fp in folder_files if str(fp) not in existing_subtitle_pre]
                        sample = ", ".join([_fmt_media(m) for m in miss[:3]])
                        _emit(f"[检查字幕] 已找到 {found_n}/{total_n}（示例缺失：{sample}）\n")
                (
                    worker_files,
                    dups_of,
                    existing_subtitle,
                    canonical_of,
                    done_media,
                    precopy_actions,
                ) = _prepare_folder(folder_files, existing_subtitle_pre=existing_subtitle_pre)

                # 1) Existing subtitle wins: handle group-level precopies first (and log them).
                logged_existing: set[str] = set()
                for src_media, src_sub, dst_media in precopy_actions:
                    if _CANCEL_EV.is_set() or (not _active()):
                        _emit("已取消：停止后续任务。\n")
                        try:
                            ct.terminate_worker()
                        except Exception:
                            pass
                        if _active():
                            js_update_state(dict(running=False, cancel_requested=True, done=True, output_files=list(output_files.values())))
                        return
                    k = str(src_media)
                    if k not in logged_existing:
                        logged_existing.add(k)
                        _emit(f"[跳过] {_fmt_media(src_media)}：已存在字幕 `{src_sub.name}`，跳过转录。\n")
                        ui0 = _copy_to_ui(
                            src_sub,
                            audio_path=src_media,
                            ui_output_dir=ui_output_dir,
                            ext=src_sub.suffix,
                            idx=len(output_files) + 1,
                            input_mode=input_mode,
                            input_dir=input_dir,
                        )
                        output_files[str(ui0)] = str(ui0)
                        _touch_album_root(src_media)

                    _emit(f"[跳过] {_fmt_media(dst_media)}：同组已有字幕，复制自 `{_fmt_media(src_media)}`。\n")
                    dst = _same_dir_sub_path(dst_media, src_sub.suffix)
                    written = _copy_subtitle(src_sub, dst)
                    if written is None:
                        continue
                    _emit(f"[复制] {_fmt_media(dst_media)}：从 `{_fmt_media(src_media)}` 复制字幕 -> `{written.name}`\n")
                    ui = _copy_to_ui(
                        written,
                        audio_path=dst_media,
                        ui_output_dir=ui_output_dir,
                        ext=written.suffix,
                        idx=len(output_files) + 1,
                        input_mode=input_mode,
                        input_dir=input_dir,
                    )
                    output_files[str(ui)] = str(ui)
                    _touch_album_root(dst_media)

                # 2) Individual existing subtitles (not part of group precopy).
                for fp in folder_files:
                    ex_s = existing_subtitle.get(str(fp))
                    if not ex_s:
                        continue
                    if str(fp) in logged_existing:
                        continue
                    src = Path(ex_s)
                    _emit(f"[跳过] {_fmt_media(fp)}：已存在字幕 `{src.name}`，跳过转录。\n")
                    ui = _copy_to_ui(
                        src,
                        audio_path=fp,
                        ui_output_dir=ui_output_dir,
                        ext=src.suffix,
                        idx=len(output_files) + 1,
                        input_mode=input_mode,
                        input_dir=input_dir,
                    )
                    output_files[str(ui)] = str(ui)
                    _touch_album_root(fp)
                    logged_existing.add(str(fp))

                # 3) Dedup logs (only for items that are NOT already handled by existing subtitles).
                for fp in folder_files:
                    if fp in done_media:
                        continue
                    canon = canonical_of.get(fp)
                    if canon is not None:
                        _emit(f"[去重] {_fmt_media(fp)}：同名同长度，使用 `{_fmt_media(canon)}` 的字幕（完成后自动复制）。\n")

                if not worker_files:
                    _emit(f"[跳过文件夹] {folder_label}：无需转录（已存在字幕/全部去重）。\n")
                    continue

                # Run this folder immediately (but worker stays alive).
                pending = list(worker_files)
                retry_same = 0
                while pending:
                    if not proc.is_alive():
                        _emit("[警告] vLLM 进程异常退出，正在重启并继续…\n")
                        try:
                            ct.terminate_worker()
                        except Exception:
                            pass
                        proc, cmd_q, out_q, mp_cancel = _start_queue_worker()
                        ct = CancelToken()
                        ct.attach_worker(proc=proc, mp_cancel=mp_cancel)
                        retry_same = 0

                    batch_id += 1
                    cur_batch = batch_id
                    cmd_q.put(("run_files", (cur_batch, [str(x) for x in pending])))

                    file_t0 = None
                    file_expected_s = None
                    base_pct = 0
                    last_emit = 0.0
                    cur_idx = 0
                    cur_name: str | None = None
                    cur_fp: Path | None = None
                    last_handled_i1 = 0
                    oom_restart: tuple[int, str] | None = None  # (idx1, msg)

                    while True:
                        if _CANCEL_EV.is_set() or (not _active()):
                            _emit("已取消：停止后续任务。\n")
                            try:
                                mp_cancel.set()
                            except Exception:
                                pass
                            try:
                                ct.terminate_worker()
                            except Exception:
                                pass
                            if _active():
                                js_update_state(dict(running=False, cancel_requested=True, done=True, output_files=list(output_files.values())))
                            return

                        try:
                            msg_type, payload = out_q.get(timeout=0.2)
                        except Exception:
                            if not proc.is_alive():
                                _emit("[警告] vLLM 进程异常退出，正在重启并继续…\n")
                                break
                            now = time.perf_counter()
                            if file_t0 is not None and (now - last_emit) >= 0.5:
                                last_emit = now
                                elapsed = max(0.0, now - file_t0)
                                expected = max(1.0, float(file_expected_s or 30.0))
                                target = 9 if (vad_cfg.enabled and base_pct < 12) else 90
                                span = max(1, target - int(base_pct))
                                frac = min(0.98, elapsed / expected)
                                pct_sim = max(int(base_pct), min(target, int(base_pct) + int(round(frac * span))))
                                try:
                                    js_update_state(dict(progress_pct=int(pct_sim), current_file=cur_name, current_idx=cur_idx, total=n))
                                except Exception:
                                    pass
                            continue

                        if msg_type == "init":
                            continue
                        if msg_type == "cancelled":
                            _emit("已取消：停止后续任务。\n")
                            js_update_state(dict(running=False, cancel_requested=True, done=True, output_files=list(output_files.values())))
                            return

                        if msg_type == "file_start":
                            b_id, idx1, total, fp_s, name = payload
                            if int(b_id) != int(cur_batch):
                                continue
                            last_handled_i1 = max(last_handled_i1, int(idx1))
                            cur_fp = Path(str(fp_s))
                            cur_name = str(name)
                            cur_idx = int(orig_idx_of.get(str(cur_fp), int(idx1)))
                            if _active():
                                _emit(f"[处理中] {cur_name} ({cur_idx}/{n})\n")
                                js_update_state(dict(progress_pct=0, current_file=cur_name, current_idx=cur_idx, total=n))
                            file_t0 = time.perf_counter()
                            base_pct = 0
                            dur_s = _get_media_duration_s(cur_fp)
                            file_expected_s = max(3.0, (dur_s or 30.0) * max(0.1, rtf_est))
                            continue

                        if msg_type == "progress":
                            b_id, fp_s, pct = payload
                            if int(b_id) != int(cur_batch):
                                continue
                            base_pct = max(int(base_pct), int(pct))
                            try:
                                if _active():
                                    js_update_state(dict(progress_pct=int(pct), current_file=cur_name, current_idx=cur_idx, total=n))
                            except Exception:
                                pass
                            continue

                        if msg_type == "file_done":
                            b_id, idx1, total, fp_s, log_line, out_path_s = payload
                            if int(b_id) != int(cur_batch):
                                continue
                            last_handled_i1 = max(last_handled_i1, int(idx1))
                            if _active():
                                _emit(str(log_line))
                            if out_path_s:
                                outp = Path(str(out_path_s))
                                src_fp = Path(str(fp_s))
                                ui = _copy_to_ui(
                                    outp,
                                    audio_path=src_fp,
                                    ui_output_dir=ui_output_dir,
                                    ext=outp.suffix,
                                    idx=len(output_files) + 1,
                                    input_mode=input_mode,
                                    input_dir=input_dir,
                                )
                                output_files[str(ui)] = str(ui)
                                _touch_album_root(src_fp)
                                for dup in (dups_of.get(src_fp) or []):
                                    dst = _same_dir_sub_path(dup, outp.suffix)
                                    written = _copy_subtitle(outp, dst)
                                    if written is None:
                                        continue
                                    _emit(f"[复制] {dup.name}：从 `{src_fp.name}` 复制字幕 -> `{written.name}`\n")
                                    ui2 = _copy_to_ui(
                                        written,
                                        audio_path=dup,
                                        ui_output_dir=ui_output_dir,
                                        ext=written.suffix,
                                        idx=len(output_files) + 1,
                                        input_mode=input_mode,
                                        input_dir=input_dir,
                                    )
                                    output_files[str(ui2)] = str(ui2)
                                    _touch_album_root(dup)
                            file_t0 = None
                            base_pct = 0
                            if _active():
                                js_update_state(
                                    dict(
                                        progress_pct=100,
                                        current_file=cur_name,
                                        current_idx=cur_idx,
                                        total=n,
                                        last_file_done_at=time.time(),
                                        last_file_done_name=cur_name,
                                        last_file_done_idx=int(cur_idx),
                                    )
                                )
                            continue

                        if msg_type == "file_error":
                            b_id, idx1, total, fp_s, msg = payload
                            if int(b_id) != int(cur_batch):
                                continue
                            last_handled_i1 = max(last_handled_i1, int(idx1))
                            name = Path(str(fp_s)).name if fp_s else (cur_name or "unknown")
                            if _active():
                                _emit(f"[失败] {name}: {msg}\n")
                            if isinstance(msg, str) and ("CUDA out of memory" in msg or "out of memory" in msg.lower()):
                                oom_restart = (int(idx1), str(msg))
                                break
                            continue

                        if msg_type == "batch_done":
                            b_id = int(payload[0]) if payload else -1
                            if b_id != int(cur_batch):
                                continue
                            break

                    # Finished this batch or need restart.
                    if oom_restart is not None:
                        idx1, msg = oom_restart
                        _emit("[警告] 检测到 OOM：重启 vLLM 并下调 gpu_memory_utilization 后继续…\n")
                        try:
                            ct.terminate_worker()
                        except Exception:
                            pass
                        if retry_same < 1:
                            retry_same += 1
                            start_from = max(0, int(idx1) - 1)
                        else:
                            start_from = int(idx1)
                        pending = pending[start_from:]
                        gpu_util = max(0.55, gpu_util - 0.05)
                        proc, cmd_q, out_q, mp_cancel = _start_queue_worker()
                        ct = CancelToken()
                        ct.attach_worker(proc=proc, mp_cancel=mp_cancel)
                        continue

                    # Normal completion of this pending list.
                    pending = []

            # shutdown worker
            try:
                cmd_q.put(("shutdown", None))
            except Exception:
                pass
            try:
                ct.terminate_worker()
            except Exception:
                pass

            if _active():
                js_update_state(dict(running=False, done=True, progress_pct=100, output_files=list(output_files.values())))
                _emit("全部完成。\n")
            return

        # transformers backend (in-process)
        try:
            rtf_est_tf = float(os.getenv("ASR_PROGRESS_RTF", "0.18"))
        except Exception:
            rtf_est_tf = 0.18
        for folder_label, folder_files in _iter_folder_batches():
            if _CANCEL_EV.is_set() or (not _active()):
                _emit("已取消：停止后续任务。\n")
                if _active():
                    js_update_state(dict(running=False, cancel_requested=True, done=True, output_files=list(output_files.values())))
                return
            if not folder_files:
                continue

            _emit(f"[处理文件夹] {folder_label}\n")
            _emit(f"[开关] 跳过已存在字幕={'ON' if skip_if_subtitle_exists else 'OFF'} 覆盖={'ON' if overwrite else 'OFF'} 去重={'ON' if dedup_same_name_same_duration else 'OFF'}\n")
            (
                worker_files,
                dups_of,
                existing_subtitle,
                canonical_of,
                done_media,
                precopy_actions,
            ) = _prepare_folder(folder_files)

            # Always print coverage when skip switch enabled (helps diagnose).
            if skip_if_subtitle_exists:
                try:
                    found_n = int(len(existing_subtitle))
                    total_n = int(len(folder_files))
                    if found_n >= total_n and total_n > 0:
                        _emit(f"[检查字幕] 已找到 {found_n}/{total_n}，跳过去重与转录。\n")
                    else:
                        miss = [fp for fp in folder_files if str(fp) not in existing_subtitle]
                        sample = ", ".join([_fmt_media(m) for m in miss[:3]])
                        _emit(f"[检查字幕] 已找到 {found_n}/{total_n}（示例缺失：{sample}）\n")
                except Exception:
                    pass

            # 1) Existing subtitle wins: handle group-level precopies first (and log them).
            logged_existing: set[str] = set()
            for src_media, src_sub, dst_media in precopy_actions:
                if _CANCEL_EV.is_set() or (not _active()):
                    _emit("已取消：停止后续任务。\n")
                    if _active():
                        js_update_state(dict(running=False, cancel_requested=True, done=True, output_files=list(output_files.values())))
                    return
                k = str(src_media)
                if k not in logged_existing:
                    logged_existing.add(k)
                    _emit(f"[跳过] {src_media.name}：已存在字幕 `{src_sub.name}`，跳过转录。\n")
                    ui0 = _copy_to_ui(
                        src_sub,
                        audio_path=src_media,
                        ui_output_dir=ui_output_dir,
                        ext=src_sub.suffix,
                        idx=len(output_files) + 1,
                        input_mode=input_mode,
                        input_dir=input_dir,
                    )
                    output_files[str(ui0)] = str(ui0)
                    _touch_album_root(src_media)

                _emit(f"[跳过] {dst_media.name}：同组已有字幕，复制自 `{src_media.name}`。\n")
                dst = _same_dir_sub_path(dst_media, src_sub.suffix)
                written = _copy_subtitle(src_sub, dst)
                if written is None:
                    continue
                _emit(f"[复制] {dst_media.name}：从 `{src_media.name}` 复制字幕 -> `{written.name}`\n")
                ui = _copy_to_ui(
                    written,
                    audio_path=dst_media,
                    ui_output_dir=ui_output_dir,
                    ext=written.suffix,
                    idx=len(output_files) + 1,
                    input_mode=input_mode,
                    input_dir=input_dir,
                )
                output_files[str(ui)] = str(ui)
                _touch_album_root(dst_media)

            # 2) Individual existing subtitles (not part of group precopy).
            for fp in folder_files:
                ex_s = existing_subtitle.get(str(fp))
                if not ex_s:
                    continue
                if str(fp) in logged_existing:
                    continue
                src = Path(ex_s)
                _emit(f"[跳过] {fp.name}：已存在字幕 `{src.name}`，跳过转录。\n")
                ui = _copy_to_ui(
                    src,
                    audio_path=fp,
                    ui_output_dir=ui_output_dir,
                    ext=src.suffix,
                    idx=len(output_files) + 1,
                    input_mode=input_mode,
                    input_dir=input_dir,
                )
                output_files[str(ui)] = str(ui)
                _touch_album_root(fp)
                logged_existing.add(str(fp))

            # 3) Dedup logs (only for items that are NOT already handled by existing subtitles).
            for fp in folder_files:
                if fp in done_media:
                    continue
                canon = canonical_of.get(fp)
                if canon is not None:
                    _emit(f"[去重] {_fmt_media(fp)}：同名同长度，使用 `{_fmt_media(canon)}` 的字幕（完成后自动复制）。\n")

            if not worker_files:
                _emit(f"[跳过文件夹] {folder_label}：无需转录（已存在字幕/全部去重）。\n")
                continue

            for fp in worker_files:
                i = int(orig_idx_of.get(str(fp), 1))
                if _CANCEL_EV.is_set() or (not _active()):
                    _emit("已取消：停止后续任务。\n")
                    if _active():
                        js_update_state(dict(running=False, cancel_requested=True, done=True, output_files=list(output_files.values())))
                    return

                if _active():
                    _emit(f"[处理中] {fp.name} ({i}/{n})\n")
                    js_update_state(dict(progress_pct=0, current_file=fp.name, current_idx=i, total=n))

                # Smooth progress ticker (keeps transformers progress consistent with vLLM UX).
                base_pct = 0
                ticker_stop = threading.Event()
                t0 = time.perf_counter()
                dur_s = _get_media_duration_s(fp)
                expected_s = max(3.0, (dur_s or 30.0) * max(0.1, float(rtf_est_tf)))

                def _ticker() -> None:
                    nonlocal base_pct
                    last_sent = -1
                    while (not ticker_stop.is_set()) and _active() and (not _CANCEL_EV.is_set()):
                        try:
                            now = time.perf_counter()
                            elapsed = max(0.0, now - t0)
                            if vad_cfg.enabled and base_pct < 12:
                                target = 9
                            else:
                                target = 90
                            span = max(1, target - int(base_pct))
                            frac = min(0.98, elapsed / max(1.0, float(expected_s)))
                            pct_sim = max(int(base_pct), min(target, int(base_pct) + int(round(frac * span))))
                            if pct_sim > last_sent:
                                last_sent = pct_sim
                                js_update_state(dict(progress_pct=int(pct_sim), current_file=fp.name, current_idx=i, total=n))
                        except Exception:
                            pass
                        ticker_stop.wait(0.5)

                ticker_th = threading.Thread(target=_ticker, daemon=True)
                ticker_th.start()

                def _progress_cb(pct: int, _stage: str = "") -> None:
                    nonlocal base_pct
                    try:
                        if _active():
                            base_pct = max(int(base_pct), int(pct))
                            js_update_state(dict(progress_pct=int(pct), current_file=fp.name, current_idx=i, total=n))
                    except Exception:
                        pass

                try:
                    log_line, content, ext = generate_for_one_audio(
                        fp,
                        asr_cfg=asr_cfg,
                        language=_normalize_language(language),
                        caption_cfg=cap_cfg,
                        transcribe_kwargs=transcribe_kwargs or None,
                        vad_cfg=vad_cfg,
                        hallucination_cfg=(hallucination_cfg_obj or None),
                        toolkit_post_process_enabled=bool(toolkit_post_process_enabled),
                        toolkit_post_process_threshold=int(toolkit_post_process_threshold),
                        progress_cb=_progress_cb,
                    )
                finally:
                    ticker_stop.set()
                    try:
                        ticker_th.join(timeout=1.0)
                    except Exception:
                        pass
                if _active():
                    _emit(str(log_line))

                out_dir = resolve_output_dir(
                    input_audio_path=fp,
                    mode=output_dir_mode,
                    custom_dir=(custom_output_dir.strip() or None),
                    default_output_dir=default_out_dir,
                )
                outp = write_output(audio_path=fp, content=content, ext=ext, output_dir=out_dir, overwrite=overwrite)
                ui = _copy_to_ui(
                    outp,
                    audio_path=fp,
                    ui_output_dir=ui_output_dir,
                    ext=outp.suffix,
                    idx=len(output_files) + 1,
                    input_mode=input_mode,
                    input_dir=input_dir,
                )
                output_files[str(ui)] = str(ui)
                _touch_album_root(fp)

                for dup in (dups_of.get(fp) or []):
                    dst = _same_dir_sub_path(dup, outp.suffix)
                    written = _copy_subtitle(outp, dst)
                    if written is None:
                        continue
                    _emit(f"[复制] {dup.name}：从 `{fp.name}` 复制字幕 -> `{written.name}`\n")
                    ui2 = _copy_to_ui(
                        written,
                        audio_path=dup,
                        ui_output_dir=ui_output_dir,
                        ext=written.suffix,
                        idx=len(output_files) + 1,
                        input_mode=input_mode,
                        input_dir=input_dir,
                    )
                    output_files[str(ui2)] = str(ui2)
                    _touch_album_root(dup)

                if _active():
                    js_update_state(
                        dict(
                            progress_pct=100,
                            current_file=fp.name,
                            current_idx=i,
                            total=n,
                            last_file_done_at=time.time(),
                            last_file_done_name=fp.name,
                            last_file_done_idx=int(i),
                        )
                    )

        if _active():
            js_update_state(dict(running=False, done=True, progress_pct=100, output_files=list(output_files.values())))
            _emit("全部完成。\n")
    except Exception as e:
        if _active():
            _emit(f"[失败] {e}\n")
        try:
            if _active():
                js_update_state(dict(running=False, done=True, last_error=str(e)))
        except Exception:
            pass
    finally:
        try:
            unload_model()
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

