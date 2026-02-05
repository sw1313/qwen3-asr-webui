from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WebUIConfigIO:
    """
    Simple JSON config persistence for WebUI.

    Priority:
    - WEBUI_CONFIG env var (or WEBUI_CONFIG_PATH) if set
    - default_path (usually /data/webui_config.json in docker)
    """

    default_path: str = "/data/webui_config.json"

    def path(self) -> Path:
        p = (os.getenv("WEBUI_CONFIG") or os.getenv("WEBUI_CONFIG_PATH") or self.default_path).strip()
        return Path(p).expanduser()


def load_config_dict(io: WebUIConfigIO) -> dict[str, Any]:
    p = io.path()
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"配置文件解析失败：{p}（{e}）")
    if not isinstance(obj, dict):
        raise RuntimeError(f"配置文件必须是 JSON 对象(dict)：{p}")
    return obj


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def save_config_dict(io: WebUIConfigIO, data: dict[str, Any]) -> Path:
    p = io.path()
    # ensure JSON serializable + stable order for diffs
    text = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
    _atomic_write_text(p, text + "\n")
    return p

