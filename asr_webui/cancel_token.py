from __future__ import annotations

import os
import signal
import threading
import time
from typing import Any


class CancelToken:
    """
    Simple cooperative cancellation token.
    Stored in gr.State so cancel button and running job can share the same object.
    """

    def __init__(self) -> None:
        self._ev = threading.Event()
        self._running = threading.Event()
        self._worker_proc: Any | None = None
        self._mp_cancel: Any | None = None

    def reset(self) -> None:
        self._ev.clear()
        # do not touch running flag here (managed by runner)

    def cancel(self) -> None:
        self._ev.set()
        try:
            if self._mp_cancel is not None:
                self._mp_cancel.set()
        except Exception:
            pass

    def is_cancelled(self) -> bool:
        return self._ev.is_set()

    def set_running(self, running: bool) -> None:
        if running:
            self._running.set()
        else:
            self._running.clear()

    def is_running(self) -> bool:
        return self._running.is_set()

    def attach_worker(self, *, proc: Any | None, mp_cancel: Any | None) -> None:
        self._worker_proc = proc
        self._mp_cancel = mp_cancel

    def terminate_worker(self) -> None:
        """
        Best-effort terminate worker process (for vLLM). This is the only reliable way
        to fully release VRAM on cancel, because CUDA context may remain in-process.
        """
        p = self._worker_proc
        if p is None:
            return
        pid = getattr(p, "pid", None)

        # First, try cooperative cancel signal (if any), then give it a moment.
        try:
            if self._mp_cancel is not None:
                self._mp_cancel.set()
        except Exception:
            pass

        # On POSIX, if the worker called setsid(), it becomes a session leader and its PGID==PID.
        # Then we can kill the whole group safely to ensure EngineCore is also terminated.
        if pid and os.name == "posix":
            try:
                pgid = os.getpgid(int(pid))
            except Exception:
                pgid = None
            if pgid is not None and int(pgid) == int(pid):
                try:
                    os.killpg(int(pgid), signal.SIGTERM)
                except Exception:
                    pass

                t0 = time.time()
                while True:
                    try:
                        alive = bool(getattr(p, "is_alive", lambda: False)())
                    except Exception:
                        alive = False
                    if not alive:
                        break
                    if time.time() - t0 > 2.0:
                        break
                    time.sleep(0.05)

                try:
                    if bool(getattr(p, "is_alive", lambda: False)()):
                        os.killpg(int(pgid), signal.SIGKILL)
                except Exception:
                    pass

        # Fallback: terminate the worker itself.
        try:
            if hasattr(p, "is_alive") and p.is_alive():
                p.terminate()
        except Exception:
            pass
        try:
            if hasattr(p, "join"):
                p.join(timeout=2.0)
        except Exception:
            pass
        self._worker_proc = None
        self._mp_cancel = None

