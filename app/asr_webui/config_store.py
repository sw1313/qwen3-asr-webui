from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def default_data_dir() -> Path:
    """
    Resolve data dir for persistence.
    - Prefer env DATA_DIR if set
    - Else prefer /data if present (common docker volume mount)
    - Else default to repo_root/data (repo_root = parent of `app/`)
    """
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    env_dir = (os.getenv("DATA_DIR") or "").strip()
    if env_dir:
        return Path(env_dir)
    docker_data = Path("/data")
    if docker_data.exists() and docker_data.is_dir():
        return docker_data
    return repo_root / "data"


def default_webui_config_path() -> Path:
    return default_data_dir() / "webui_config.json"


def load_json(path: Path) -> dict[str, Any]:
    try:
        if not path.exists():
            return {}
        s = path.read_text(encoding="utf-8")
        obj = json.loads(s) if s.strip() else {}
        return obj if isinstance(obj, dict) else {}
    except Exception:
        # Be resilient: bad JSON should not prevent app from starting.
        return {}


def save_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def merge_update(path: Path, updates: dict[str, Any]) -> None:
    """
    Merge updates into existing JSON dict and persist.
    Keeps unknown keys to allow forward compatibility.
    """
    cur = load_json(path)
    if not isinstance(cur, dict):
        cur = {}
    cur.setdefault("__schema__", "asr_webui.webui_config")
    cur.setdefault("__schema_version__", 1)
    cur["__updated_at__"] = time.strftime("%Y-%m-%d %H:%M:%S")
    cur.update(updates)
    save_json_atomic(path, cur)

