from __future__ import annotations

from pathlib import Path


AUDIO_EXTS_DEFAULT = [".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wma"]


def list_audio_files(
    input_dir: str,
    *,
    recursive: bool = True,
    exts: list[str] | None = None,
) -> list[Path]:
    p = Path(input_dir).expanduser()
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"输入目录不存在或不是文件夹: {p}")

    exts_n = [e.lower() if e.startswith(".") else f".{e.lower()}" for e in (exts or AUDIO_EXTS_DEFAULT)]
    pattern = "**/*" if recursive else "*"
    files = [x for x in p.glob(pattern) if x.is_file() and x.suffix.lower() in exts_n]
    return sorted(files)


def resolve_output_dir(
    *,
    input_audio_path: Path,
    mode: str,
    custom_dir: str | None,
    default_output_dir: Path,
) -> Path:
    if mode == "output":
        return default_output_dir
    if mode == "same":
        return input_audio_path.parent
    if mode == "custom":
        if not custom_dir:
            raise ValueError("输出路径模式为自定义，但未提供自定义目录")
        return Path(custom_dir).expanduser()
    raise ValueError(f"未知输出路径模式: {mode}")

