from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .subtitle_formats import Segment


PUNCTUATION = set(list("。！？!?；;，,、…：:"))


@dataclass(frozen=True)
class TokenTS:
    text: str
    start_time: float
    end_time: float


def _token_from_any(x: Any) -> TokenTS:
    # qwen-asr returns objects with attributes: text/start_time/end_time
    # Be defensive so UI doesn't hard-crash on minor API diffs.
    if hasattr(x, "text") and hasattr(x, "start_time") and hasattr(x, "end_time"):
        return TokenTS(str(x.text), float(x.start_time), float(x.end_time))
    if isinstance(x, dict) and {"text", "start_time", "end_time"} <= set(x.keys()):
        return TokenTS(str(x["text"]), float(x["start_time"]), float(x["end_time"]))
    raise TypeError(f"Unsupported timestamp token type: {type(x)}")


def iter_tokens(time_stamps: Any) -> list[TokenTS]:
    """
    Normalize `result.time_stamps` into a flat list[TokenTS].

    Observed patterns:
    - list[TokenTS]  (single stream)
    - list[list[TokenTS]] (batch-like / multi-stream)
    """
    if time_stamps is None:
        return []

    if isinstance(time_stamps, list):
        if not time_stamps:
            return []
        if isinstance(time_stamps[0], list):
            # take the first stream by default (most common in docs: r.time_stamps[0])
            inner = time_stamps[0]
            return [_token_from_any(t) for t in inner]
        return [_token_from_any(t) for t in time_stamps]

    # fallback: treat as iterable
    if isinstance(time_stamps, Iterable):
        return [_token_from_any(t) for t in list(time_stamps)]

    raise TypeError(f"Unsupported time_stamps container type: {type(time_stamps)}")


def tokens_to_segments(
    tokens: list[TokenTS],
    *,
    max_chars_per_line: int = 40,
    gap_break_s: float = 0.8,
    break_on_punct: bool = True,
) -> list[Segment]:
    """
    Convert token-level timestamps to sentence/line segments using simple heuristics:
    - break on punctuation tokens
    - break on long silence gap between adjacent tokens
    - break if text length exceeds max_chars_per_line
    """
    if not tokens:
        return []

    segs: list[Segment] = []
    buf_txt: list[str] = []
    seg_start = tokens[0].start_time
    seg_end = tokens[0].end_time

    def flush():
        nonlocal buf_txt, seg_start, seg_end
        text = "".join(buf_txt).strip()
        if text:
            segs.append(Segment(start_s=seg_start, end_s=max(seg_start, seg_end), text=text))
        buf_txt = []

    for i, tok in enumerate(tokens):
        if not buf_txt:
            seg_start = tok.start_time
        seg_end = tok.end_time
        buf_txt.append(tok.text)

        # heuristics
        text_len = sum(len(x) for x in buf_txt)
        is_punct = (tok.text.strip() in PUNCTUATION) if break_on_punct else False

        next_gap = 0.0
        if i + 1 < len(tokens):
            next_gap = max(0.0, tokens[i + 1].start_time - tok.end_time)

        if is_punct or next_gap >= gap_break_s or text_len >= max_chars_per_line:
            flush()

    flush()

    # if end times are missing / odd, make sure monotonic
    for idx in range(len(segs) - 1):
        if segs[idx].end_s <= segs[idx].start_s:
            segs[idx] = Segment(segs[idx].start_s, segs[idx + 1].start_s, segs[idx].text)

    return segs

