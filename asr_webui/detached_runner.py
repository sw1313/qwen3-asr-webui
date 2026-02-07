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
from .vllm_subprocess_batch_worker import vllm_worker_batch


_JOB_LOCK = threading.Lock()
_JOB_THREAD: threading.Thread | None = None
_CANCEL_EV = threading.Event()
_CURRENT_JOB_ID: str | None = None


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
    # Try ffprobe first (works for audio/video), fallback to torchaudio.
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
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
    try:
        import torchaudio  # type: ignore

        info = torchaudio.info(str(p))
        if getattr(info, "num_frames", 0) and getattr(info, "sample_rate", 0):
            return float(info.num_frames) / float(info.sample_rate)
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

        # Prepare inputs
        if input_mode == "dir":
            exts = [x.strip() for x in (exts_csv or "").split(",") if x.strip()]
            files = list_audio_files(input_dir, recursive=recursive, exts=exts or MEDIA_EXTS_DEFAULT)
        else:
            files = _normalize_uploads(uploads)

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
        _emit(f"共 {n} 个音频，开始批量生成…\n")
        js_update_state(dict(running=True, done=False, total=n, progress_pct=0, current_file=None, current_idx=0))

        ui_output_dir = Path(os.getenv("WEBUI_OUTPUT_DIR") or os.getenv("OUTPUT_DIR") or str(Path.cwd() / "output"))
        default_out_dir = Path(os.getenv("OUTPUT_DIR", str(Path.cwd() / "output")))

        output_files: dict[str, str] = {}
        seen_folders: set[str] = set()
        touched_albums: set[str] = set()

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

        if backend_n == "vllm":
            ctx = mp.get_context("spawn")
            gpu_util = float(vllm_gpu_memory_utilization)
            start_idx0 = 0
            retry_same = 0
            try:
                rtf_est = float(os.getenv("ASR_PROGRESS_RTF", "0.18"))
            except Exception:
                rtf_est = 0.18

            file_t0: float | None = None
            file_expected_s: float | None = None
            base_pct: int = 0
            last_emit = 0.0

            def start_worker():
                mp_cancel = ctx.Event()
                q = ctx.Queue()
                cfg_d = asdict(asr_cfg)
                try:
                    vk = dict(cfg_d.get("vllm_kwargs") or {})
                    vk["gpu_memory_utilization"] = float(gpu_util)
                    cfg_d["vllm_kwargs"] = vk
                except Exception:
                    pass
                proc = ctx.Process(
                    target=vllm_worker_batch,
                    args=(q, mp_cancel),
                    kwargs=dict(
                        files=[str(x) for x in files],
                        start_idx0=int(start_idx0),
                        asr_cfg_d=cfg_d,
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
                proc.start()
                return proc, q, mp_cancel

            proc, q, mp_cancel = start_worker()

            cur_idx = 0
            cur_name: str | None = None
            while True:
                if not _active():
                    # best-effort stop
                    try:
                        ct = CancelToken()
                        ct.attach_worker(proc=proc, mp_cancel=mp_cancel)
                        ct.terminate_worker()
                    except Exception:
                        pass
                    return
                if _CANCEL_EV.is_set():
                    try:
                        mp_cancel.set()
                    except Exception:
                        pass
                    # kill whole tree via CancelToken helper
                    ct = CancelToken()
                    ct.attach_worker(proc=proc, mp_cancel=mp_cancel)
                    ct.terminate_worker()
                    _emit("已取消：停止后续任务。\n")
                    js_update_state(dict(running=False, cancel_requested=True, done=True, output_files=list(output_files.values())))
                    return

                try:
                    msg_type, payload = q.get(timeout=0.2)
                except Exception:
                    # worker died?
                    if not proc.is_alive():
                        _emit("[警告] vLLM 进程异常退出，正在重启并继续…\n")
                        proc, q, mp_cancel = start_worker()
                        file_t0 = None
                        base_pct = 0
                        last_emit = 0.0
                    # Even if no queue messages, we still want progress to move.
                    now = time.perf_counter()
                    if file_t0 is not None and (now - last_emit) >= 0.5:
                        last_emit = now
                        elapsed = max(0.0, now - file_t0)
                        expected = max(1.0, float(file_expected_s or 30.0))
                        if vad_cfg.enabled and base_pct < 12:
                            target = 9
                        else:
                            target = 90
                        span = max(1, target - int(base_pct))
                        frac = min(0.98, elapsed / expected)
                        pct_sim = max(int(base_pct), min(target, int(base_pct) + int(round(frac * span))))
                        try:
                            js_update_state(dict(progress_pct=int(pct_sim), current_file=cur_name, current_idx=cur_idx, total=n))
                        except Exception:
                            pass
                    continue

                if msg_type == "file_start":
                    cur_idx, total, cur_name = payload
                    start_idx0 = int(cur_idx - 1)
                    retry_same = 0
                    fp = Path(files[cur_idx - 1])
                    if _active():
                        maybe_folder_log(fp)
                        _emit(f"[处理中] {cur_name} ({cur_idx}/{total})\n")
                        js_update_state(dict(progress_pct=0, current_file=cur_name, current_idx=cur_idx, total=total))
                    file_t0 = time.perf_counter()
                    base_pct = 0
                    dur_s = _get_media_duration_s(fp)
                    file_expected_s = max(3.0, (dur_s or 30.0) * max(0.1, rtf_est))
                elif msg_type == "progress":
                    pct = int(payload)
                    base_pct = max(base_pct, pct)
                    if _active():
                        js_update_state(dict(progress_pct=pct, current_file=cur_name, current_idx=cur_idx, total=n))
                elif msg_type == "file_done":
                    idx1, total, log_line, out_path_s = payload
                    if _active():
                        _emit(str(log_line))
                    if out_path_s:
                        outp = Path(out_path_s)
                        ui = _copy_to_ui(outp, audio_path=Path(files[idx1 - 1]), ui_output_dir=ui_output_dir, ext=outp.suffix, idx=len(output_files) + 1, input_mode=input_mode, input_dir=input_dir)
                        output_files[str(ui)] = str(ui)
                        _touch_album_root(Path(files[idx1 - 1]))
                    start_idx0 = int(idx1)
                    file_t0 = None
                    base_pct = 0
                    if _active():
                        js_update_state(
                        dict(
                            progress_pct=100,
                            current_file=cur_name,
                            current_idx=idx1,
                            total=total,
                            last_file_done_at=time.time(),
                            last_file_done_name=cur_name,
                            last_file_done_idx=int(idx1),
                        )
                    )
                elif msg_type == "file_error":
                    idx1, total, msg = payload
                    name = Path(files[idx1 - 1]).name if idx1 else (cur_name or "unknown")
                    if _active():
                        _emit(f"[失败] {name}: {msg}\n")
                    if isinstance(msg, str) and ("CUDA out of memory" in msg or "out of memory" in msg.lower()):
                        _emit("[警告] 检测到 OOM：重启 vLLM 并下调 gpu_memory_utilization 后继续…\n")
                        # terminate current worker tree
                        ct = CancelToken()
                        ct.attach_worker(proc=proc, mp_cancel=mp_cancel)
                        ct.terminate_worker()
                        if retry_same < 1:
                            retry_same += 1
                            start_idx0 = max(0, idx1 - 1)
                        else:
                            start_idx0 = idx1
                        gpu_util = max(0.55, gpu_util - 0.05)
                        proc, q, mp_cancel = start_worker()
                        file_t0 = None
                        base_pct = 0
                        continue
                elif msg_type == "done":
                    break

            if _active():
                js_update_state(dict(running=False, done=True, progress_pct=100, output_files=list(output_files.values())))
                _emit("全部完成。\n")
            return

        # transformers backend (in-process)
        try:
            rtf_est_tf = float(os.getenv("ASR_PROGRESS_RTF", "0.18"))
        except Exception:
            rtf_est_tf = 0.18
        for i, fp in enumerate(files, start=1):
            if _CANCEL_EV.is_set() or (not _active()):
                _emit("已取消：停止后续任务。\n")
                if _active():
                    js_update_state(dict(running=False, cancel_requested=True, done=True, output_files=list(output_files.values())))
                return

            if _active():
                maybe_folder_log(fp)
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
            ui = _copy_to_ui(outp, audio_path=fp, ui_output_dir=ui_output_dir, ext=outp.suffix, idx=len(output_files) + 1, input_mode=input_mode, input_dir=input_dir)
            output_files[str(ui)] = str(ui)
            _touch_album_root(fp)
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

