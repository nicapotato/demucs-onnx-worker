"""Find the bundled htdemucs_6s ONNX file. Never downloads."""

from __future__ import annotations

import os
import sys
from pathlib import Path

MODEL_NAME = "htdemucs_6s"
MODEL_FILES = ("htdemucs_6s_fp16weights.onnx", "htdemucs_6s.onnx")
CACHE_ENVS = (
    "DEMUCS_ONNX_CACHE",
    "DEMUCS_MLX_CACHE",
    "DEMUCS_CUDA_CACHE",
    "DEMUCS_TORCH_CACHE",
)


def executable_dir(argv0: str | None) -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    if argv0:
        candidate = Path(argv0)
        if candidate.is_file():
            return candidate.resolve().parent
    return Path(sys.executable).resolve().parent


def search_dirs(exe_dir: Path) -> list[Path]:
    dirs: list[Path] = []
    for key in CACHE_ENVS:
        value = os.environ.get(key, "").strip()
        if value:
            dirs.append(Path(value))
    dirs.extend(
        [
            exe_dir,
            exe_dir / "models",
            exe_dir.parent / "models",
            exe_dir.parent.parent / "models",
            exe_dir.parent.parent / "Resources" / "models",
            Path.cwd() / "models",
            Path.cwd(),
        ]
    )
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        root = Path(meipass)
        dirs.extend([root, root / "models"])
    return dirs


def find_model(exe_dir: Path, weights: str | None = None) -> Path:
    """Return the ONNX file to load.

    ``weights`` and ``DEMUCS_ONNX_MODEL`` win. Otherwise the first existing
    ``htdemucs_6s_fp16weights.onnx`` (then the fp32 file) under the cache
    directories the GUI already sets, then next to the executable.
    """
    explicit = (weights or os.environ.get("DEMUCS_ONNX_MODEL") or "").strip()
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {path}")
        return path

    seen: set[Path] = set()
    for directory in search_dirs(exe_dir):
        try:
            resolved = directory.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        for name in MODEL_FILES:
            candidate = directory / name
            if candidate.is_file():
                return candidate

    looked = ", ".join(str(d) for d in search_dirs(exe_dir)[:6])
    raise FileNotFoundError(
        "htdemucs_6s ONNX weights not found. "
        f"Place {MODEL_FILES[0]} in the models directory ({looked})."
    )
