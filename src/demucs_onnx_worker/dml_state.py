"""Remember which DirectML mode worked on this machine.

The GUI starts one worker process per song, so without this every song would
pay for a failed GPU attempt before falling back. The ladder is::

    fused  ->  unfused (ep.dml.disable_graph_fusion)  ->  cpu

A remembered step is keyed on ONNX Runtime version, adapter name, and model
file, and expires after ``RETRY_AFTER_S`` so a driver update gets retried.
``DEMUCS_ONNX_DML`` overrides it: ``fused``, ``unfused``, ``cpu``, or
``retry`` (ignore what was remembered).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

MODES: tuple[str, ...] = ("fused", "unfused", "cpu")
RETRY_AFTER_S = 7 * 24 * 3600
ENV = "DEMUCS_ONNX_DML"


def state_path() -> Path:
    override = os.environ.get("DEMUCS_ONNX_STATE_DIR", "").strip()
    if override:
        base = Path(override)
    elif sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        base = Path(os.environ["LOCALAPPDATA"]) / "demucs-onnx-worker"
    else:
        base = Path.home() / ".cache" / "demucs-onnx-worker"
    return base / "dml_state.json"


def state_key(ort_version: str, gpu: str | None, model_path: Path) -> str:
    try:
        size = model_path.stat().st_size
    except OSError:
        size = 0
    return f"{ort_version}|{gpu or '?'}|{model_path.name}|{size}"


def _load(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def next_mode(mode: str) -> str:
    return MODES[min(MODES.index(mode) + 1, len(MODES) - 1)]


def start_mode(key: str, *, now: float | None = None, path: Path | None = None) -> tuple[str, str]:
    """Return ``(mode, why)``. ``why`` is ``env``, ``remembered``, or ``default``."""
    forced = os.environ.get(ENV, "").strip().lower()
    if forced in MODES:
        return forced, "env"
    if forced == "retry":
        return "fused", "env"
    entry = _load(path or state_path()).get(key)
    if isinstance(entry, dict) and entry.get("mode") in MODES:
        age = (now if now is not None else time.time()) - float(entry.get("at", 0))
        if 0 <= age < RETRY_AFTER_S:
            return str(entry["mode"]), "remembered"
    return "fused", "default"


def remember(key: str, mode: str, reason: str, *, path: Path | None = None) -> Path | None:
    """Record the mode to start from next time. Never raises."""
    target = path or state_path()
    try:
        data = _load(target)
        data[key] = {"mode": mode, "at": time.time(), "reason": reason[:2000]}
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, target)
    except OSError:
        return None
    return target
