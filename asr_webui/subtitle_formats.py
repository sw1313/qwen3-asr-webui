from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    start_s: float
    end_s: float
    text: str


def _clamp_nonneg(x: float) -> float:
    return x if x > 0 else 0.0


def format_srt_timestamp(seconds: float) -> str:
    seconds = _clamp_nonneg(seconds)
    ms_total = int(round(seconds * 1000.0))
    ms = ms_total % 1000
    s_total = ms_total // 1000
    s = s_total % 60
    m_total = s_total // 60
    m = m_total % 60
    h = m_total // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_lrc_timestamp(seconds: float) -> str:
    # Common LRC uses mm:ss.xx (hundredths)
    seconds = _clamp_nonneg(seconds)
    cs_total = int(round(seconds * 100.0))  # centiseconds
    cs = cs_total % 100
    s_total = cs_total // 100
    s = s_total % 60
    m = s_total // 60
    return f"{m:02d}:{s:02d}.{cs:02d}"


def segments_to_srt(segments: list[Segment]) -> str:
    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{format_srt_timestamp(seg.start_s)} --> {format_srt_timestamp(seg.end_s)}")
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def segments_to_lrc(segments: list[Segment]) -> str:
    lines: list[str] = []
    for seg in segments:
        t = format_lrc_timestamp(seg.start_s)
        lines.append(f"[{t}]{seg.text.strip()}")
    return "\n".join(lines).rstrip() + "\n"

