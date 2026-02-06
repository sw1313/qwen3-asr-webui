from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def _data_dir() -> Path:
    # Align with config_store: prefer DATA_DIR, else /data if mounted, else ./data
    env_dir = (os.getenv("DATA_DIR") or "").strip()
    if env_dir:
        return Path(env_dir)
    docker_data = Path("/data")
    if docker_data.exists() and docker_data.is_dir():
        return docker_data
    return Path.cwd() / "data"


def state_path() -> Path:
    return _data_dir() / "job_state.json"


def log_path() -> Path:
    return _data_dir() / "job.log"


def _save_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_state() -> dict[str, Any]:
    p = state_path()
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def update_state(updates: dict[str, Any]) -> None:
    cur = read_state()
    if not isinstance(cur, dict):
        cur = {}
    cur.setdefault("__schema__", "asr_webui.job_state")
    cur.setdefault("__schema_version__", 1)
    cur["__updated_at__"] = time.strftime("%Y-%m-%d %H:%M:%S")
    cur.update(updates)
    _save_json_atomic(state_path(), cur)


def reset_job(*, job_id: str) -> None:
    # Truncate log and reset state.
    lp = log_path()
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text("", encoding="utf-8")
    update_state(
        dict(
            job_id=job_id,
            running=True,
            cancel_requested=False,
            done=False,
            progress_pct=0,
            current_file=None,
            current_idx=0,
            total=0,
            output_files=[],
            last_error=None,
        )
    )


def clear_job() -> None:
    """
    Clear persisted job state and truncate job.log.
    Used on cancel and on container start (per user preference).
    """
    try:
        lp = log_path()
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text("", encoding="utf-8")
    except Exception:
        pass
    try:
        update_state(
            dict(
                running=False,
                cancel_requested=False,
                done=False,
                progress_pct=0,
                current_file=None,
                current_idx=0,
                total=0,
                output_files=[],
                last_error=None,
                last_file_done_at=None,
                last_file_done_name=None,
                last_file_done_idx=None,
            )
        )
    except Exception:
        pass


def append_log(line: str) -> None:
    lp = log_path()
    lp.parent.mkdir(parents=True, exist_ok=True)
    with lp.open("a", encoding="utf-8") as f:
        f.write(line)


def read_log_tail(*, max_bytes: int = 1024 * 256) -> str:
    """
    Read last ~max_bytes of log file (UTF-8) to avoid huge transfers.
    """
    lp = log_path()
    if not lp.exists():
        return ""
    try:
        size = lp.stat().st_size
        if size <= max_bytes:
            return lp.read_text(encoding="utf-8", errors="ignore")
        with lp.open("rb") as f:
            f.seek(max(0, size - max_bytes))
            data = f.read()
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""

