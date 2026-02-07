from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .file_utils import resolve_output_dir
from .qwen_runner import ASRConfig, transcribe_with_timestamps
from .segmenter import iter_tokens, tokens_to_segments
from .subtitle_formats import segments_to_lrc, segments_to_srt
from .vad import VadConfig, cut_waveform_segments, detect_speech_segments


_VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".ts"}


def _is_video_file(p: Path) -> bool:
    return p.suffix.lower() in _VIDEO_EXTS


def _ffprobe_has_audio_stream(p: Path) -> bool:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        # If we cannot probe, assume it has audio and let downstream handle.
        return True
    try:
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(p),
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        return bool(out)
    except Exception:
        return True


def _decode_head_f32_mono(p: Path, *, sr: int = 16000, seconds: float = 10.0) -> np.ndarray | None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    try:
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(p),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(int(sr)),
            "-t",
            str(float(seconds)),
            "-f",
            "f32le",
            "pipe:1",
        ]
        raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        if not raw:
            return np.zeros((0,), dtype=np.float32)
        return np.frombuffer(raw, dtype=np.float32)
    except Exception:
        return None


def _is_effectively_silent(p: Path) -> bool:
    # Quick check: decode short head and estimate RMS/mean abs.
    wav = _decode_head_f32_mono(p)
    if wav is None:
        return False
    if wav.size == 0:
        return True
    mean_abs = float(np.mean(np.abs(wav)))
    return mean_abs < 1e-4


def _extract_audio_to_wav(src: Path, *, sr: int = 16000) -> Path | None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    try:
        tmp = tempfile.NamedTemporaryFile(prefix="asr_", suffix=".wav", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(int(sr)),
            "-f",
            "wav",
            str(tmp_path),
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if p.returncode != 0 or (not tmp_path.exists()) or tmp_path.stat().st_size == 0:
            try:
                tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass
            return None
        return tmp_path
    except Exception:
        return None


@dataclass(frozen=True)
class CaptionConfig:
    output_format: Literal["srt", "lrc"] = "srt"
    max_chars_per_line: int = 40
    gap_break_s: float = 0.8
    break_on_punct: bool = True


def generate_for_one_audio(
    audio_path: Path,
    *,
    asr_cfg: ASRConfig,
    language: str | None,
    caption_cfg: CaptionConfig,
    transcribe_kwargs: dict | None = None,
    vad_cfg: VadConfig | None = None,
    progress_cb: Any | None = None,
) -> tuple[str, str, str]:
    t0 = time.perf_counter()
    vad_cfg = vad_cfg or VadConfig(enabled=False)
    all_token_dicts: list[dict] = []
    text_parts: list[str] = []
    lang = None

    # Decide output ext early, even for "empty output" cases.
    if caption_cfg.output_format == "srt":
        out_ext = ".srt"
    else:
        out_ext = ".lrc"

    # Video support: if input is a video, extract audio via ffmpeg first.
    work_path = audio_path
    tmp_wav: Path | None = None
    try:
        if progress_cb:
            progress_cb(0, "开始")
    except Exception:
        pass
    if _is_video_file(audio_path):
        try:
            if progress_cb:
                progress_cb(2, "提取音频")
        except Exception:
            pass
        if not _ffprobe_has_audio_stream(audio_path):
            return f"[无音轨] {audio_path.name}：视频无音轨，输出空文件\n", "", out_ext
        tmp_wav = _extract_audio_to_wav(audio_path, sr=int(getattr(vad_cfg, "target_sr", 16000)))
        if tmp_wav is None:
            # treat as silent/no audio
            return f"[无声] {audio_path.name}：无法提取音频或音频为空，输出空文件\n", "", out_ext
        work_path = tmp_wav

    # Silence handling: audio/video with silence should produce empty output (not error).
    if _is_effectively_silent(work_path):
        if tmp_wav is not None:
            try:
                tmp_wav.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass
        return f"[无声] {audio_path.name}：检测到音频无声，输出空文件\n", "", out_ext

    try:
        if progress_cb:
            progress_cb(5, "预处理完成")
    except Exception:
        pass

    if vad_cfg.enabled:
        segs = detect_speech_segments(work_path, cfg=vad_cfg)
        try:
            if progress_cb:
                # user's expectation: VAD done should land around 10-15%
                progress_cb(12, "VAD 完成")
        except Exception:
            pass
        if not segs:
            if tmp_wav is not None:
                try:
                    tmp_wav.unlink(missing_ok=True)  # type: ignore[arg-type]
                except Exception:
                    pass
            return f"[跳过] {audio_path.name}：VAD 未检测到语音段（输出空文件）\n", "", out_ext

        cuts = cut_waveform_segments(work_path, segs, sr=vad_cfg.target_sr)
        audios = [(w, sr) for (w, sr, _off) in cuts]
        offsets = [off for (_w, _sr, off) in cuts]

        # batch transcribe segments
        results = transcribe_with_timestamps(
            audios,
            cfg=asr_cfg,
            language=language,
            return_time_stamps=True,
            transcribe_kwargs=transcribe_kwargs,
        )
        for r, off in zip(results, offsets):
            lang = lang or getattr(r, "language", None)
            text_parts.append((getattr(r, "text", "") or "").strip())
            ts = getattr(r, "time_stamps", None)
            toks = iter_tokens(ts)
            for t in toks:
                all_token_dicts.append(
                    dict(
                        text=t.text,
                        start_time=float(t.start_time) + float(off),
                        end_time=float(t.end_time) + float(off),
                    )
                )
        text = " ".join([x for x in text_parts if x])
        tokens = iter_tokens(all_token_dicts)
    else:
        results = transcribe_with_timestamps(
            str(work_path),
            cfg=asr_cfg,
            language=language,
            return_time_stamps=True,
            transcribe_kwargs=transcribe_kwargs,
        )
        if not results:
            if tmp_wav is not None:
                try:
                    tmp_wav.unlink(missing_ok=True)  # type: ignore[arg-type]
                except Exception:
                    pass
            return f"[跳过] {audio_path.name}：无识别结果（输出空文件）\n", "", out_ext

        r0 = results[0]
        text = getattr(r0, "text", "") or ""
        lang = getattr(r0, "language", None)
        ts = getattr(r0, "time_stamps", None)
        tokens = iter_tokens(ts)

    try:
        if progress_cb:
            progress_cb(85, "ASR 完成")
    except Exception:
        pass

    if tokens:
        segments = tokens_to_segments(
            tokens,
            max_chars_per_line=max(5, int(caption_cfg.max_chars_per_line)),
            gap_break_s=max(0.0, float(caption_cfg.gap_break_s)),
            break_on_punct=bool(caption_cfg.break_on_punct),
        )
    else:
        # fallback: single segment without timestamps
        segments = []

    # If ASR/VAD produced no meaningful text, we still output an empty subtitle file (no error).
    if not (text or "").strip() and not segments:
        vad_note = " VAD=on" if vad_cfg.enabled else ""
        elapsed_s = time.perf_counter() - t0
        if tmp_wav is not None:
            try:
                tmp_wav.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass
        return (
            f"[空] {audio_path.name}{vad_note}：无有效文字，输出空文件 时间={elapsed_s:.2f}s\n",
            "",
            out_ext,
        )

    try:
        if progress_cb:
            progress_cb(92, "生成字幕")
    except Exception:
        pass

    if caption_cfg.output_format == "srt":
        content = segments_to_srt(segments) if segments else (text.strip() + "\n" if text.strip() else "")
        ext = ".srt"
    else:
        content = segments_to_lrc(segments) if segments else (text.strip() + "\n" if text.strip() else "")
        ext = ".lrc"

    vad_note = " VAD=on" if vad_cfg.enabled else ""
    elapsed_s = time.perf_counter() - t0
    out = (
        f"[完成] {audio_path.name}{vad_note} 语言={lang} 字符={len(text)} 段落={len(segments)} 时间={elapsed_s:.2f}s\n",
        content,
        ext,
    )
    if tmp_wav is not None:
        try:
            tmp_wav.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass
    return out


def write_output(
    *,
    audio_path: Path,
    content: str,
    ext: str,
    output_dir: Path,
    overwrite: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{audio_path.stem}{ext}"
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"输出文件已存在（未允许覆盖）: {out_path}")
    # Write atomically: helps avoid partial files and also updates directory mtime on overwrite
    # (rename/replace updates directory entry, while in-place overwrite may not).
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(out_path)
    return out_path


def batch_generate(
    audio_files: list[Path],
    *,
    asr_cfg: ASRConfig,
    language: str | None,
    caption_cfg: CaptionConfig,
    output_dir_mode: Literal["output", "same", "custom"] = "output",
    custom_output_dir: str | None = None,
    overwrite: bool = False,
    default_output_dir: Path | None = None,
    transcribe_kwargs: dict | None = None,
    vad_cfg: VadConfig | None = None,
    debug_traceback: bool = False,
) -> tuple[str, list[Path]]:
    default_output_dir = default_output_dir or (Path.cwd() / "output")
    logs: list[str] = []
    outputs: list[Path] = []

    for audio_path in audio_files:
        try:
            log, content, ext = generate_for_one_audio(
                audio_path,
                asr_cfg=asr_cfg,
                language=language,
                caption_cfg=caption_cfg,
                transcribe_kwargs=transcribe_kwargs,
                vad_cfg=vad_cfg,
            )
            logs.append(log)

            out_dir = resolve_output_dir(
                input_audio_path=audio_path,
                mode=output_dir_mode,
                custom_dir=custom_output_dir,
                default_output_dir=default_output_dir,
            )
            out_path = write_output(
                audio_path=audio_path,
                content=content,
                ext=ext,
                output_dir=out_dir,
                overwrite=overwrite,
            )
            outputs.append(out_path)
        except Exception as e:
            if debug_traceback:
                tb = traceback.format_exc()
                logs.append(f"[失败] {audio_path}: {e}\n{tb}\n")
            else:
                logs.append(f"[失败] {audio_path}: {e}\n")

    return "".join(logs), outputs

