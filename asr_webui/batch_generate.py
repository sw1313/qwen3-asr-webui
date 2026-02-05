from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .file_utils import resolve_output_dir
from .qwen_runner import ASRConfig, transcribe_with_timestamps
from .segmenter import iter_tokens, tokens_to_segments
from .subtitle_formats import segments_to_lrc, segments_to_srt
from .vad import VadConfig, cut_waveform_segments, detect_speech_segments


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
) -> tuple[str, str, str]:
    t0 = time.perf_counter()
    vad_cfg = vad_cfg or VadConfig(enabled=False)
    all_token_dicts: list[dict] = []
    text_parts: list[str] = []
    lang = None

    if vad_cfg.enabled:
        segs = detect_speech_segments(audio_path, cfg=vad_cfg)
        if not segs:
            return f"[跳过] {audio_path.name}：VAD 未检测到语音段\n", "", ""

        cuts = cut_waveform_segments(audio_path, segs, sr=vad_cfg.target_sr)
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
            str(audio_path),
            cfg=asr_cfg,
            language=language,
            return_time_stamps=True,
            transcribe_kwargs=transcribe_kwargs,
        )
        if not results:
            return f"[跳过] {audio_path}：无识别结果\n", "", ""

        r0 = results[0]
        text = getattr(r0, "text", "") or ""
        lang = getattr(r0, "language", None)
        ts = getattr(r0, "time_stamps", None)
        tokens = iter_tokens(ts)

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

    if caption_cfg.output_format == "srt":
        content = segments_to_srt(segments) if segments else text.strip() + "\n"
        ext = ".srt"
    else:
        content = segments_to_lrc(segments) if segments else text.strip() + "\n"
        ext = ".lrc"

    vad_note = " VAD=on" if vad_cfg.enabled else ""
    elapsed_s = time.perf_counter() - t0
    return (
        f"[完成] {audio_path.name}{vad_note} 语言={lang} 字符={len(text)} 段落={len(segments)} 时间={elapsed_s:.2f}s\n",
        content,
        ext,
    )


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
    out_path.write_text(content, encoding="utf-8")
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

