from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch


@dataclass(frozen=True)
class VadConfig:
    enabled: bool = False
    backend: Literal["silero"] = "silero"

    # how to obtain silero-vad assets
    # - auto: try local_repo_dir (if provided) else hub; if hub fails, raise actionable error
    # - local: only load from local_repo_dir
    # - hub: only load from torch.hub (auto download on first run if network is available)
    source: Literal["auto", "local", "hub"] = "auto"
    local_repo_dir: str | None = None

    # audio decode / resample
    target_sr: int = 16000

    # silero core params (match silero get_speech_timestamps style)
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    max_speech_duration_s: float = 60.0
    min_silence_duration_ms: int = 100
    speech_pad_ms: int = 30
    window_size_samples: int = 512

    # post process
    merge_gap_ms: int = 120
    min_segment_ms: int = 300

    # advanced: extra kwargs pass-through to get_speech_timestamps (even if not needed)
    vad_kwargs: dict[str, Any] | None = None


def parse_json_dict(name: str, s: str) -> dict[str, Any]:
    s = (s or "").strip()
    if not s:
        return {}
    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise ValueError(f"{name} 必须是 JSON 对象(dict)，但拿到的是 {type(obj)}")
    return obj


def _ffmpeg_decode(path: Path, *, sr: int) -> np.ndarray:
    """
    Decode audio to mono float32 waveform at `sr` using ffmpeg.
    Returns np.ndarray shape [n_samples], float32 in [-1,1] (approx).
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("未找到 ffmpeg，无法解码该音频用于 VAD。请安装 ffmpeg 或改用 wav/flac。")

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-f",
        "f32le",
        "pipe:1",
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if p.returncode != 0:
        err = p.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg 解码失败：{err.strip()}")
    # np.frombuffer() produces a read-only view; make it writable to avoid torch warnings/UB.
    audio = np.frombuffer(p.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise RuntimeError("ffmpeg 解码得到空音频。")
    return audio


def load_audio_mono(path: str | Path, *, sr: int) -> torch.Tensor:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"音频不存在: {p}")

    # try torchaudio first (if installed)
    try:
        import torchaudio  # type: ignore

        wav, src_sr = torchaudio.load(str(p))
        # wav: [channels, time]
        if wav.ndim == 2 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        elif wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if int(src_sr) != int(sr):
            wav = torchaudio.functional.resample(wav, int(src_sr), int(sr))
        return wav.squeeze(0).contiguous()
    except Exception:
        pass

    # fallback to ffmpeg
    audio = _ffmpeg_decode(p, sr=sr)
    return torch.from_numpy(audio).contiguous()


def _merge_and_filter_segments(
    segs: list[tuple[float, float]],
    *,
    merge_gap_s: float,
    min_segment_s: float,
) -> list[tuple[float, float]]:
    if not segs:
        return []
    segs = sorted(segs)
    merged: list[tuple[float, float]] = []
    cur_s, cur_e = segs[0]
    for s, e in segs[1:]:
        if s <= cur_e + merge_gap_s:
            cur_e = max(cur_e, e)
        else:
            if cur_e - cur_s >= min_segment_s:
                merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    if cur_e - cur_s >= min_segment_s:
        merged.append((cur_s, cur_e))
    return merged


class _SileroVADCache:
    def __init__(self) -> None:
        self._model = None
        self._utils = None

    def load(self, *, source: str, local_repo_dir: str | None):
        if self._model is not None and self._utils is not None:
            return self._model, self._utils

        repo_or_dir: str | None = None
        local = (local_repo_dir or os.getenv("VAD_REPO_DIR") or "").strip()
        if local:
            p = Path(local).expanduser()
            if p.exists() and p.is_dir():
                repo_or_dir = str(p)
            else:
                if source == "local":
                    raise RuntimeError(f"VAD source=local 但本地目录不存在：{p}")

        def load_from(repo: str):
            # Torch Hub is tightening security: future versions may require explicit trust
            # for GitHub repos and may prompt interactively. We force non-interactive trust
            # when supported, and gracefully fallback on older torch versions.
            kwargs = dict(
                repo_or_dir=repo,
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )
            try:
                return torch.hub.load(**kwargs, trust_repo=True)
            except TypeError:
                return torch.hub.load(**kwargs)

        if source == "local":
            if not repo_or_dir:
                raise RuntimeError("VAD source=local 但未提供 VAD_REPO_DIR/local_repo_dir")
            model, utils = load_from(repo_or_dir)
        elif source == "hub":
            model, utils = load_from("snakers4/silero-vad")
        else:
            # auto: local first, then hub
            try:
                if repo_or_dir:
                    model, utils = load_from(repo_or_dir)
                else:
                    raise FileNotFoundError("no local repo")
            except Exception:
                try:
                    model, utils = load_from("snakers4/silero-vad")
                except Exception as e:
                    if "No module named 'torchaudio'" in str(e):
                        raise RuntimeError(
                            "Silero VAD 加载失败：缺少依赖 `torchaudio`。\n"
                            "解决办法：在运行环境安装 torchaudio（推荐与 torch 版本/CPU-CUDA 匹配）。\n"
                            "如果你是在 Docker 里跑：重新 build 镜像后再启动。\n"
                            f"原始错误：{e}"
                        )
                    raise RuntimeError(
                        "Silero VAD 获取失败：当前环境可能离线，且未提供本地 VAD。\n"
                        "解决办法（二选一）：\n"
                        "1) 允许容器/机器首次联网运行一次（会自动下载并缓存）；\n"
                        "2) 准备本地 silero-vad 仓库目录并设置环境变量 VAD_REPO_DIR 挂载进去。\n"
                        "另外：Silero VAD 可能还需要安装 `torchaudio`。\n"
                        f"原始错误：{e}"
                    )

        self._model, self._utils = model, utils
        return model, utils


SILERO_CACHE = _SileroVADCache()


def detect_speech_segments(
    audio_path: str | Path,
    *,
    cfg: VadConfig,
) -> list[tuple[float, float]]:
    """
    Returns list of (start_s, end_s) on ORIGINAL audio timeline.
    """
    if not cfg.enabled:
        return []
    if cfg.backend != "silero":
        raise ValueError(f"未知 VAD backend: {cfg.backend}")

    model, utils = SILERO_CACHE.load(source=cfg.source, local_repo_dir=cfg.local_repo_dir)
    get_speech_timestamps = utils[0]

    wav = load_audio_mono(audio_path, sr=cfg.target_sr)

    kwargs: dict[str, Any] = dict(
        threshold=float(cfg.threshold),
        sampling_rate=int(cfg.target_sr),
        min_speech_duration_ms=int(cfg.min_speech_duration_ms),
        max_speech_duration_s=float(cfg.max_speech_duration_s),
        min_silence_duration_ms=int(cfg.min_silence_duration_ms),
        speech_pad_ms=int(cfg.speech_pad_ms),
        window_size_samples=int(cfg.window_size_samples),
    )
    if cfg.vad_kwargs:
        kwargs.update(cfg.vad_kwargs)

    ts = get_speech_timestamps(wav, model, **kwargs)
    segs: list[tuple[float, float]] = []
    for item in ts:
        # silero returns samples indices: {"start": int, "end": int}
        start = float(item["start"]) / float(cfg.target_sr)
        end = float(item["end"]) / float(cfg.target_sr)
        if end > start:
            segs.append((start, end))

    segs = _merge_and_filter_segments(
        segs,
        merge_gap_s=float(cfg.merge_gap_ms) / 1000.0,
        min_segment_s=float(cfg.min_segment_ms) / 1000.0,
    )
    return segs


def cut_waveform_segments(
    audio_path: str | Path,
    segments_s: list[tuple[float, float]],
    *,
    sr: int,
) -> list[tuple[np.ndarray, int, float]]:
    """
    Returns list of (wave_np, sr, offset_s)
    """
    wav = load_audio_mono(audio_path, sr=sr)
    wav_np = wav.detach().cpu().numpy().astype(np.float32, copy=False)
    out: list[tuple[np.ndarray, int, float]] = []
    n = wav_np.shape[0]
    for s, e in segments_s:
        i0 = max(0, min(n, int(round(s * sr))))
        i1 = max(0, min(n, int(round(e * sr))))
        if i1 <= i0:
            continue
        out.append((wav_np[i0:i1], sr, float(i0) / float(sr)))
    return out

