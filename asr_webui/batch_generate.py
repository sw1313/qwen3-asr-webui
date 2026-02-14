from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
import traceback
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .file_utils import resolve_output_dir
from .qwen_runner import ASRConfig, transcribe_with_timestamps
from .segmenter import iter_tokens, tokens_to_segments
from .subtitle_formats import Segment, segments_to_lrc, segments_to_srt
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


def _toolkit_fix_char_repeats(s: str, thresh: int) -> str:
    # Mirror Qwen3-ASR-Toolkit logic.
    res: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        count = 1
        while i + count < n and s[i + count] == s[i]:
            count += 1
        if count > thresh:
            res.append(s[i])
            i += count
        else:
            res.append(s[i : i + count])
            i += count
    return "".join(res)


def _toolkit_fix_pattern_repeats(s: str, thresh: int, max_len: int = 20) -> str:
    # Mirror Qwen3-ASR-Toolkit logic.
    n = len(s)
    min_repeat_chars = thresh * 2
    if n < min_repeat_chars:
        return s

    i = 0
    result: list[str] = []
    while i <= n - min_repeat_chars:
        found = False
        for k in range(1, max_len + 1):
            if i + k * thresh > n:
                break
            pattern = s[i : i + k]
            valid = True
            for rep in range(1, thresh):
                start_idx = i + rep * k
                if s[start_idx : start_idx + k] != pattern:
                    valid = False
                    break
            if valid:
                total_rep = thresh
                end_index = i + thresh * k
                while end_index + k <= n and s[end_index : end_index + k] == pattern:
                    total_rep += 1
                    end_index += k
                result.append(pattern)
                result.append(_toolkit_fix_pattern_repeats(s[end_index:], thresh, max_len))
                i = n
                found = True
                break
        if found:
            break
        result.append(s[i])
        i += 1
    if not found:
        result.append(s[i:])
    return "".join(result)


def toolkit_post_text_process(text: str, threshold: int = 20) -> str:
    # Mirror Qwen3-ASR-Toolkit QwenASR.post_text_process.
    t = _toolkit_fix_char_repeats(text, int(threshold))
    return _toolkit_fix_pattern_repeats(t, int(threshold))


def _clean_text_keep_jp(text: str) -> str:
    s = (text or "").strip()
    # Normalize only whitespace/case; keep punctuation as hard boundaries.
    # This avoids treating non-consecutive repeats as consecutive.
    s = re.sub(r"\s+", "", s)
    return s.lower()


def _is_entire_nplus_repeat(text: str, n: int, min_unit_len: int = 1) -> bool:
    n = max(2, int(n))
    min_unit_len = max(1, int(min_unit_len))
    cleaned = _clean_text_keep_jp(text)
    if len(cleaned) < n:
        return False
    pattern = rf"^(.{{{min_unit_len},}}?)\1{{{n - 1},}}$"
    m = re.match(pattern, cleaned)
    if not m:
        return False
    unit = m.group(1)
    return _valid_repeat_unit(unit)


def _find_entire_repeat_match(text: str, n: int, min_unit_len: int = 1) -> tuple[str, int] | None:
    n = max(2, int(n))
    min_unit_len = max(1, int(min_unit_len))
    cleaned = _clean_text_keep_jp(text)
    if len(cleaned) < n:
        return None
    pattern = rf"^(.{{{min_unit_len},}}?)\1{{{n - 1},}}$"
    m = re.match(pattern, cleaned)
    if not m:
        return None
    unit = m.group(1)
    if not _valid_repeat_unit(unit):
        return None
    try:
        rep = int(len(cleaned) // max(1, len(unit)))
    except Exception:
        rep = n
    return unit, rep


def _has_substring_repeat_n_times(text: str, n: int, min_unit_len: int = 1) -> bool:
    return _find_substring_repeat_match(text, n=n, min_unit_len=min_unit_len) is not None


def _find_substring_repeat_match(text: str, n: int, min_unit_len: int = 1) -> tuple[str, int] | None:
    n = max(2, int(n))
    min_unit_len = max(1, int(min_unit_len))
    cleaned = _clean_text_keep_jp(text)
    if len(cleaned) < n:
        return None
    pattern = rf"(.{{{min_unit_len},}}?)\1{{{n - 1},}}"
    pos = 0
    while pos < len(cleaned):
        m = re.search(pattern, cleaned[pos:])
        if not m:
            return None
        unit = m.group(1)
        if _valid_repeat_unit(unit):
            try:
                rep = int(len(m.group(0)) // max(1, len(unit)))
            except Exception:
                rep = n
            return unit, rep
        pos += max(1, m.start() + 1)
    return None


def _valid_repeat_unit(unit: str) -> bool:
    u = (unit or "").strip()
    if not u:
        return False
    # Reject units that are just one repeated char (e.g. "...", "ーーー", "aaaa").
    if len(set(u)) <= 1:
        return False
    # Must contain at least one meaningful char (JP/CJK/latin/digit), not punctuation-only.
    if not re.search(r"[0-9A-Za-z\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]", u):
        return False
    return True


def _contains_keywords(text: str, keywords: list[str]) -> bool:
    if not keywords:
        return False
    t = (text or "").lower()
    for kw in keywords:
        k = (kw or "").strip().lower()
        if k and k in t:
            return True
    return False


def _segment_hallucinated(text: str, cfg: dict[str, Any]) -> bool:
    return bool(_segment_hallucination_reasons(text, cfg))


def _segment_hallucination_reasons(text: str, cfg: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    kws = [str(x) for x in (cfg.get("segment_keywords") or []) if str(x).strip()]
    if _contains_keywords(text, kws):
        hit = ""
        t = (text or "").lower()
        for kw in kws:
            kk = (kw or "").strip().lower()
            if kk and kk in t:
                hit = kk
                break
        reasons.append(f"关键词命中({hit or 'keyword'})")
    # NOTE:
    # Repetition-based hallucination checks are intentionally moved to "whole subtitle retry".
    # Segment-level retry only uses explicit keyword hits to avoid over-triggering before
    # final subtitle segmentation is available.
    return reasons


def _preview_text(t: str, n: int = 120) -> str:
    p = (t or "").replace("\n", " ").strip()
    if len(p) > n:
        p = p[:n] + "..."
    return p


def _collect_repeat_rule_hits(
    *,
    segments: list[Segment],
    full_text: str,
    matcher: Any,
    reason_prefix: str,
) -> list[tuple[int, str, str]]:
    """
    Collect per-segment hits for a repeat rule.
    Return (segment_index_1based_or_0_for_full_text, reason, preview).
    """
    hits: list[tuple[int, str, str]] = []
    if segments:
        for i, seg in enumerate(segments, start=1):
            t = (getattr(seg, "text", "") or "").strip()
            if not t:
                continue
            m = matcher(t)
            if m is None:
                continue
            unit, rep = m
            u = unit if len(unit) <= 24 else (unit[:24] + "...")
            reason = f"{reason_prefix},命中单元=`{u}`,重复≈{rep}次"
            hits.append((i, reason, _preview_text(t)))
        return hits
    # Fallback (no segments): evaluate on whole text once.
    m = matcher(full_text or "")
    if m is None:
        return []
    unit, rep = m
    u = unit if len(unit) <= 24 else (unit[:24] + "...")
    reason = f"{reason_prefix},命中单元=`{u}`,重复≈{rep}次"
    return [(0, reason, _preview_text(full_text or ""))]


def _format_rule_hits_for_log(hits: list[tuple[int, str, str]], *, title: str) -> str:
    """
    Format all matched hits into compact log text.
    Each subtitle segment (timestamp block) counts as one hit.
    """
    if not hits:
        return ""
    parts: list[str] = []
    for seg_idx, reason, preview in hits:
        if seg_idx > 0:
            parts.append(f"段#{seg_idx} 规则={reason} 文本片段=`{preview}`")
        else:
            parts.append(f"整段文本 规则={reason} 文本片段=`{preview}`")
    return f"{title}（共{len(hits)}次）: " + "；".join(parts)


def _merge_adjacent_same_text_with_count(segments: list[Segment]) -> tuple[list[Segment], int]:
    if not segments:
        return [], 0
    out: list[Segment] = [segments[0]]
    merge_count = 0
    for seg in segments[1:]:
        prev = out[-1]
        if (prev.text or "").strip() == (seg.text or "").strip():
            out[-1] = Segment(start_s=float(prev.start_s), end_s=float(seg.end_s), text=str(prev.text))
            merge_count += 1
        else:
            out.append(seg)
    return out, merge_count


def _media_duration_s(p: Path) -> float | None:
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


def _default_hallucination_cfg() -> dict[str, Any]:
    return dict(
        enabled=False,
        # retry control
        retry_segment=True,
        retry_segment_times=1,
        retry_whole_subtitle=False,
        retry_whole_times=1,
        # VAD segment-level rules
        segment_keywords=[],
        # whole repeat rules (per-segment counting)
        whole_entire_repeat_enabled=True,
        whole_entire_repeat_min_unit_len=1,
        whole_entire_repeat_min_repeats=3,
        whole_entire_repeat_min_hits=1,
        whole_any_repeat_enabled=True,
        whole_any_repeat_min_unit_len=1,
        whole_any_repeat_min_repeats=4,
        whole_any_repeat_min_hits=1,
        # whole subtitle rules
        whole_tail_gap_enabled=True,
        whole_tail_gap_s=15.0,
        whole_long_segment_enabled=True,
        whole_long_segment_s=20.0,
        whole_long_segment_count=2,
        whole_merged_duration_enabled=True,
        merge_adjacent_same_text=True,
        whole_merged_duration_s=20.0,
        whole_empty_output_enabled=False,
        # whole retry VAD mode: none/base/other
        whole_retry_vad_mode="base",
        # backward compatibility key
        whole_retry_use_other_vad=False,
        whole_retry_vad_threshold=0.5,
        whole_retry_vad_min_speech_ms=250,
        whole_retry_vad_max_speech_s=60.0,
        whole_retry_vad_min_silence_ms=100,
        whole_retry_vad_speech_pad_ms=30,
        whole_retry_vad_window_size_samples=512,
        whole_retry_vad_merge_gap_ms=120,
        whole_retry_vad_min_segment_ms=300,
    )


def _normalize_hallucination_cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    out = _default_hallucination_cfg()
    if isinstance(cfg, dict):
        out.update(cfg)
        # Backward compatibility with older key names.
        if "whole_entire_repeat_min_repeats" not in cfg and "segment_entire_repeat_n" in cfg:
            out["whole_entire_repeat_min_repeats"] = cfg.get("segment_entire_repeat_n")
        if "whole_any_repeat_min_repeats" not in cfg and "segment_any_repeat_n" in cfg:
            out["whole_any_repeat_min_repeats"] = cfg.get("segment_any_repeat_n")
        if "whole_entire_repeat_min_unit_len" not in cfg and "segment_repeat_min_unit_len" in cfg:
            out["whole_entire_repeat_min_unit_len"] = cfg.get("segment_repeat_min_unit_len")
        if "whole_any_repeat_min_unit_len" not in cfg and "segment_repeat_min_unit_len" in cfg:
            out["whole_any_repeat_min_unit_len"] = cfg.get("segment_repeat_min_unit_len")
        if "whole_retry_vad_mode" not in cfg:
            out["whole_retry_vad_mode"] = "other" if bool(cfg.get("whole_retry_use_other_vad", False)) else "base"
    return out


def _apply_whole_retry_vad_cfg(base: VadConfig, hcfg: dict[str, Any]) -> tuple[VadConfig, str]:
    """
    Whole-retry VAD mode selector.
    Returns (vad_cfg, mode) where mode in {"none","base","other"}.
    """
    mode = str(hcfg.get("whole_retry_vad_mode", "") or "").strip().lower()
    if mode not in {"none", "base", "other"}:
        mode = "other" if bool(hcfg.get("whole_retry_use_other_vad", False)) else "base"
    if mode == "none":
        return VadConfig(enabled=False), "none"
    if mode == "base":
        return base, "base"
    try:
        return (
            VadConfig(
            enabled=bool(base.enabled),
            backend=base.backend,
            source=base.source,
            local_repo_dir=base.local_repo_dir,
            target_sr=int(base.target_sr),
            threshold=float(hcfg.get("whole_retry_vad_threshold", base.threshold)),
            min_speech_duration_ms=int(hcfg.get("whole_retry_vad_min_speech_ms", base.min_speech_duration_ms)),
            max_speech_duration_s=float(hcfg.get("whole_retry_vad_max_speech_s", base.max_speech_duration_s)),
            min_silence_duration_ms=int(hcfg.get("whole_retry_vad_min_silence_ms", base.min_silence_duration_ms)),
            speech_pad_ms=int(hcfg.get("whole_retry_vad_speech_pad_ms", base.speech_pad_ms)),
            window_size_samples=int(hcfg.get("whole_retry_vad_window_size_samples", base.window_size_samples)),
            merge_gap_ms=int(hcfg.get("whole_retry_vad_merge_gap_ms", base.merge_gap_ms)),
            min_segment_ms=int(hcfg.get("whole_retry_vad_min_segment_ms", base.min_segment_ms)),
            vad_kwargs=base.vad_kwargs,
            ),
            "other",
        )
    except Exception:
        return base, "base"


def generate_for_one_audio(
    audio_path: Path,
    *,
    asr_cfg: ASRConfig,
    language: str | None,
    caption_cfg: CaptionConfig,
    transcribe_kwargs: dict | None = None,
    vad_cfg: VadConfig | None = None,
    hallucination_cfg: dict[str, Any] | None = None,
    toolkit_post_process_enabled: bool = False,
    toolkit_post_process_threshold: int = 20,
    progress_cb: Any | None = None,
    _whole_retry_left: int | None = None,
    _retry_logs: list[str] | None = None,
    _whole_recheck_rule_ids: list[int] | None = None,
) -> tuple[str, str, str]:
    t0 = time.perf_counter()
    vad_cfg = vad_cfg or VadConfig(enabled=False)
    hcfg = _normalize_hallucination_cfg(hallucination_cfg)
    all_token_dicts: list[dict] = []
    text_parts: list[str] = []
    lang = None
    hallu_notes: list[str] = []
    retry_logs: list[str] = list(_retry_logs or [])

    def _maybe_retry_on_empty_output(reason: str) -> tuple[str, str, str] | None:
        if not bool(hcfg.get("enabled")):
            return None
        if not bool(hcfg.get("whole_empty_output_enabled", False)):
            return None
        if not bool(hcfg.get("retry_whole_subtitle", False)):
            return None
        whole_retry_times = max(0, int(hcfg.get("retry_whole_times", 1) or 1))
        cur_left = whole_retry_times if _whole_retry_left is None else int(_whole_retry_left)
        if cur_left <= 0:
            return None
        attempt_idx = max(1, whole_retry_times - cur_left + 1)
        retry_vad_cfg, retry_vad_mode = _apply_whole_retry_vad_cfg(vad_cfg, hcfg)
        start_line = (
            f"[重试] {audio_path.name} 整段触发重试#{attempt_idx}（剩余 {cur_left} 次），原因：无输出结果({reason})\n"
        )
        if retry_vad_mode == "none":
            start_line += f"[重试] {audio_path.name} 整段重试不使用VAD\n"
        elif retry_vad_mode == "other":
            start_line += (
                f"[重试] {audio_path.name} 整段重试使用替代VAD参数："
                f"threshold={retry_vad_cfg.threshold},"
                f"min_speech_ms={retry_vad_cfg.min_speech_duration_ms},"
                f"max_speech_s={retry_vad_cfg.max_speech_duration_s},"
                f"min_silence_ms={retry_vad_cfg.min_silence_duration_ms},"
                f"speech_pad_ms={retry_vad_cfg.speech_pad_ms},"
                f"window={retry_vad_cfg.window_size_samples},"
                f"merge_gap_ms={retry_vad_cfg.merge_gap_ms},"
                f"min_segment_ms={retry_vad_cfg.min_segment_ms}\n"
            )
        t_retry = time.perf_counter()
        log_line2, content2, ext2 = generate_for_one_audio(
            audio_path,
            asr_cfg=asr_cfg,
            language=language,
            caption_cfg=caption_cfg,
            transcribe_kwargs=transcribe_kwargs,
            vad_cfg=retry_vad_cfg,
            hallucination_cfg=hcfg,
            toolkit_post_process_enabled=toolkit_post_process_enabled,
            toolkit_post_process_threshold=toolkit_post_process_threshold,
            progress_cb=progress_cb,
            _whole_retry_left=cur_left - 1,
            _retry_logs=retry_logs,
            _whole_recheck_rule_ids=[6],
        )
        elapsed_retry = time.perf_counter() - t_retry
        end_line = f"[重试] {audio_path.name} 整段重试#{attempt_idx}完成，耗时={elapsed_retry:.2f}s\n"
        return start_line + end_line + log_line2, content2, ext2

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
            retried = _maybe_retry_on_empty_output("VAD未检测到语音段")
            if retried is not None:
                return retried
            return f"[跳过] {audio_path.name}：VAD 未检测到语音段（输出空文件）\n", "", out_ext

        cuts = cut_waveform_segments(work_path, segs, sr=vad_cfg.target_sr)
        audios = [(w, sr) for (w, sr, _off) in cuts]
        offsets = [off for (_w, _sr, off) in cuts]

        # batch transcribe segments (with optional per-segment retry on hallucination)
        results = transcribe_with_timestamps(
            audios,
            cfg=asr_cfg,
            language=language,
            return_time_stamps=True,
            transcribe_kwargs=transcribe_kwargs,
        )
        if bool(hcfg.get("enabled")) and bool(hcfg.get("retry_segment", True)):
            seg_retry_times = max(0, int(hcfg.get("retry_segment_times", 1) or 1))
            fixed_results = list(results)
            for idx, (r0, aud0) in enumerate(zip(results, audios)):
                rt = (getattr(r0, "text", "") or "").strip()
                reasons0 = _segment_hallucination_reasons(rt, hcfg)
                if not reasons0:
                    continue
                retried = False
                attempts = 0
                for _ in range(seg_retry_times):
                    attempts += 1
                    one = transcribe_with_timestamps(
                        [aud0],
                        cfg=asr_cfg,
                        language=language,
                        return_time_stamps=True,
                        transcribe_kwargs=transcribe_kwargs,
                    )
                    if not one:
                        continue
                    cand = one[0]
                    cand_text = (getattr(cand, "text", "") or "").strip()
                    fixed_results[idx] = cand
                    retried = True
                    if not _segment_hallucinated(cand_text, hcfg):
                        break
                if retried:
                    hallu_notes.append(f"VAD段重试#{idx+1}")
                    final_txt = (getattr(fixed_results[idx], "text", "") or "").strip()
                    reasons_final = _segment_hallucination_reasons(final_txt, hcfg)
                    still = bool(reasons_final)
                    preview = (rt or "").replace("\n", " ").strip()
                    if len(preview) > 120:
                        preview = preview[:120] + "..."
                    retry_logs.append(
                        f"[重试] {audio_path.name} VAD段#{idx+1} 命中规则[{';'.join(reasons0)}] 文本片段=`{preview}`，已重试 {attempts} 次，"
                        + ("仍疑似幻觉\n" if still else "已恢复正常\n")
                    )
                    if still:
                        retry_logs.append(
                            f"[重试] {audio_path.name} VAD段#{idx+1} 重试后仍命中[{';'.join(reasons_final)}]\n"
                        )
            results = fixed_results
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
            retried = _maybe_retry_on_empty_output("无识别结果")
            if retried is not None:
                return retried
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

    # Optional post-processing compatible with Qwen3-ASR-Toolkit.
    # Apply to both plain text and segment text so SRT/LRC output is consistent.
    if toolkit_post_process_enabled:
        try:
            th = max(2, int(toolkit_post_process_threshold))
        except Exception:
            th = 20
        try:
            text = toolkit_post_text_process(text or "", threshold=th)
        except Exception:
            pass
        if segments:
            new_segments = []
            for seg in segments:
                try:
                    seg_txt = toolkit_post_text_process(getattr(seg, "text", "") or "", threshold=th)
                    new_segments.append(
                        Segment(
                            start_s=float(getattr(seg, "start_s")),
                            end_s=float(getattr(seg, "end_s")),
                            text=seg_txt,
                        )
                    )
                except Exception:
                    # Keep original segment on any conversion mismatch.
                    new_segments.append(seg)
            segments = new_segments

    # Optional whole-subtitle hallucination check + retry whole audio.
    if bool(hcfg.get("enabled")):
        whole_hit = False
        reasons: list[str] = []
        hit_rule_ids: list[int] = []
        # rule 0a: 整段文本由重复子串构成（按字幕段统计命中次数）
        if bool(hcfg.get("whole_entire_repeat_enabled", True)):
            whole_entire_min_unit = int(hcfg.get("whole_entire_repeat_min_unit_len", 1) or 1)
            whole_entire_min_rep = int(hcfg.get("whole_entire_repeat_min_repeats", 3) or 3)
            whole_entire_min_hits = max(1, int(hcfg.get("whole_entire_repeat_min_hits", 1) or 1))
            entire_hits = _collect_repeat_rule_hits(
                segments=segments,
                full_text=(text or ""),
                matcher=lambda t: _find_entire_repeat_match(
                    t, n=whole_entire_min_rep, min_unit_len=whole_entire_min_unit
                ),
                reason_prefix=(
                    f"整段文本由重复子串构成>=N({whole_entire_min_rep}),"
                    f"最少子串长度={whole_entire_min_unit}"
                ),
            )
            if len(entire_hits) >= whole_entire_min_hits:
                whole_hit = True
                if 1 not in hit_rule_ids:
                    hit_rule_ids.append(1)
                reasons.append(
                    f"整段文本由重复子串构成规则命中{len(entire_hits)}次(阈值>={whole_entire_min_hits})"
                )

        # rule 0b: 文本中有连续重复子串（按字幕段统计命中次数）
        if bool(hcfg.get("whole_any_repeat_enabled", True)):
            whole_any_min_unit = int(hcfg.get("whole_any_repeat_min_unit_len", 1) or 1)
            whole_any_min_rep = int(hcfg.get("whole_any_repeat_min_repeats", 4) or 4)
            whole_any_min_hits = max(1, int(hcfg.get("whole_any_repeat_min_hits", 1) or 1))
            any_hits = _collect_repeat_rule_hits(
                segments=segments,
                full_text=(text or ""),
                matcher=lambda t: _find_substring_repeat_match(
                    t, n=whole_any_min_rep, min_unit_len=whole_any_min_unit
                ),
                reason_prefix=(
                    f"文本中有连续重复子串>=N({whole_any_min_rep}),"
                    f"最少子串长度={whole_any_min_unit}"
                ),
            )
            if len(any_hits) >= whole_any_min_hits:
                whole_hit = True
                if 2 not in hit_rule_ids:
                    hit_rule_ids.append(2)
                reasons.append(f"文本中有连续重复子串规则命中{len(any_hits)}次(阈值>={whole_any_min_hits})")

        # rule 1: tail gap (media_duration - last_segment_end) > n
        if bool(hcfg.get("whole_tail_gap_enabled", True)) and segments:
            dur = _media_duration_s(audio_path)
            if dur is not None:
                tail_gap = float(dur) - float(getattr(segments[-1], "end_s", 0.0))
                if tail_gap > float(hcfg.get("whole_tail_gap_s", 15.0) or 15.0):
                    whole_hit = True
                    if 3 not in hit_rule_ids:
                        hit_rule_ids.append(3)
                    reasons.append(f"尾段差值过大({tail_gap:.2f}s)")
        # rule 2: long segments count >= n
        if bool(hcfg.get("whole_long_segment_enabled", True)) and segments:
            long_s = float(hcfg.get("whole_long_segment_s", 20.0) or 20.0)
            long_n = int(hcfg.get("whole_long_segment_count", 2) or 2)
            long_count = sum(1 for s in segments if (float(getattr(s, "end_s", 0.0)) - float(getattr(s, "start_s", 0.0))) > long_s)
            if long_count >= max(1, long_n):
                whole_hit = True
                if 4 not in hit_rule_ids:
                    hit_rule_ids.append(4)
                reasons.append(f"超长段数量异常({long_count})")
        # rule 3: merge adjacent same text, then if merged segment duration > n and merged_count>1
        if bool(hcfg.get("whole_merged_duration_enabled", True)) and bool(hcfg.get("merge_adjacent_same_text", True)) and segments:
            merged, merge_count = _merge_adjacent_same_text_with_count(segments)
            mdur_s = float(hcfg.get("whole_merged_duration_s", 20.0) or 20.0)
            if merge_count > 1 and any((float(s.end_s) - float(s.start_s)) > mdur_s for s in merged):
                whole_hit = True
                if 5 not in hit_rule_ids:
                    hit_rule_ids.append(5)
                reasons.append("相邻相同段合并后超长")
            # optional: when enabled, keep merged text blocks in final subtitle
            if merge_count > 0:
                segments = merged

        # whole retry / post-retry verification
        whole_retry_times = max(0, int(hcfg.get("retry_whole_times", 1) or 1))
        if _whole_retry_left is None:
            _whole_retry_left = whole_retry_times
        can_whole_retry = bool(hcfg.get("retry_whole_subtitle", False)) and int(_whole_retry_left) > 0
        had_whole_retry_before = int(_whole_retry_left) < int(whole_retry_times)
        expected_rule_ids = [int(x) for x in (_whole_recheck_rule_ids or []) if str(x).strip()]
        def _rule_label(i: int) -> str:
            if i == 1:
                n = max(1, int(hcfg.get("whole_entire_repeat_min_hits", 1) or 1))
                return f"整段文本由重复子串构成≥{n}次"
            if i == 2:
                n = max(1, int(hcfg.get("whole_any_repeat_min_hits", 1) or 1))
                return f"文本中有连续重复子串≥{n}次"
            if i == 3:
                s = float(hcfg.get("whole_tail_gap_s", 15.0) or 15.0)
                return f"字幕尾时间戳与音频时长差值>{s:g}s"
            if i == 4:
                n = max(1, int(hcfg.get("whole_long_segment_count", 2) or 2))
                return f"超长段数量≥{n}"
            if i == 5:
                s = float(hcfg.get("whole_merged_duration_s", 20.0) or 20.0)
                return f"相邻同文本合并后超长>{s:g}s"
            if i == 6:
                return "无输出结果"
            return "未知规则"

        def _fmt_rules(ids: list[int]) -> str:
            uniq = sorted(set(int(x) for x in ids))
            if not uniq:
                return "（无）"
            return ",".join(f"{i}[{_rule_label(i)}]" for i in uniq)

        ids_text = _fmt_rules(hit_rule_ids)
        expected_ids_text = _fmt_rules(expected_rule_ids)
        # If we are in a post-retry pass, explicitly report re-check result.
        if had_whole_retry_before:
            if whole_hit:
                retry_logs.append(
                    f"[重试] {audio_path.name} 重试后复检：仍命中{ids_text}\n"
                )
            else:
                if expected_rule_ids:
                    retry_logs.append(
                        f"[重试] {audio_path.name} 重试后复检：已恢复正常（未再命中{expected_ids_text}）\n"
                    )
                else:
                    retry_logs.append(f"[重试] {audio_path.name} 重试后复检：已恢复正常（未再命中规则）\n")
        if whole_hit and can_whole_retry:
            hallu_notes.append("整段重试触发:" + ",".join(reasons))
            try:
                whole_total = max(0, int(hcfg.get("retry_whole_times", 1) or 1))
            except Exception:
                whole_total = 1
            try:
                cur_left = int(_whole_retry_left)
            except Exception:
                cur_left = 1
            attempt_idx = max(1, whole_total - cur_left + 1)
            reason_compact = f"命中{ids_text}" if hit_rule_ids else ",".join(reasons)
            start_line = (
                f"[重试] {audio_path.name} 整段触发重试#{attempt_idx}（剩余 {cur_left} 次），原因：{reason_compact}\n"
            )
            retry_vad_cfg, retry_vad_mode = _apply_whole_retry_vad_cfg(vad_cfg, hcfg)
            if retry_vad_mode == "none":
                start_line += f"[重试] {audio_path.name} 整段重试不使用VAD\n"
            elif retry_vad_mode == "other":
                start_line += (
                    f"[重试] {audio_path.name} 整段重试使用替代VAD参数："
                    f"threshold={retry_vad_cfg.threshold},"
                    f"min_speech_ms={retry_vad_cfg.min_speech_duration_ms},"
                    f"max_speech_s={retry_vad_cfg.max_speech_duration_s},"
                    f"min_silence_ms={retry_vad_cfg.min_silence_duration_ms},"
                    f"speech_pad_ms={retry_vad_cfg.speech_pad_ms},"
                    f"window={retry_vad_cfg.window_size_samples},"
                    f"merge_gap_ms={retry_vad_cfg.merge_gap_ms},"
                    f"min_segment_ms={retry_vad_cfg.min_segment_ms}\n"
                )
            t_retry = time.perf_counter()
            log_line2, content2, ext2 = generate_for_one_audio(
                audio_path,
                asr_cfg=asr_cfg,
                language=language,
                caption_cfg=caption_cfg,
                transcribe_kwargs=transcribe_kwargs,
                vad_cfg=retry_vad_cfg,
                hallucination_cfg=hcfg,
                toolkit_post_process_enabled=toolkit_post_process_enabled,
                toolkit_post_process_threshold=toolkit_post_process_threshold,
                progress_cb=progress_cb,
                _whole_retry_left=int(_whole_retry_left) - 1,
                _retry_logs=retry_logs,
                _whole_recheck_rule_ids=sorted(set(hit_rule_ids)),
            )
            elapsed_retry = time.perf_counter() - t_retry
            end_line = f"[重试] {audio_path.name} 整段重试#{attempt_idx}完成，耗时={elapsed_retry:.2f}s\n"
            return start_line + end_line + log_line2, content2, ext2

    # If ASR/VAD produced no meaningful text, we still output an empty subtitle file (no error).
    if not (text or "").strip() and not segments:
        retried = _maybe_retry_on_empty_output("无有效文字")
        if retried is not None:
            if tmp_wav is not None:
                try:
                    tmp_wav.unlink(missing_ok=True)  # type: ignore[arg-type]
                except Exception:
                    pass
            return retried
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
    prefix = "".join(retry_logs)
    out = (
        prefix
        + f"[完成] {audio_path.name}{vad_note} 语言={lang} 字符={len(text)} 段落={len(segments)} 时间={elapsed_s:.2f}s"
        + (f" 后处理重试={';'.join(hallu_notes)}" if hallu_notes else "")
        + "\n",
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

