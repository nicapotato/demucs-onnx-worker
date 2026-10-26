#!/usr/bin/env python3
"""Zip the PyInstaller onedir so the archive root is demucs_onnx_worker/."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="dist/worker/demucs_onnx_worker")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    exe_name = "demucs_onnx_worker.exe" if os.name == "nt" else "demucs_onnx_worker"
    exe = src / exe_name
    if not exe.is_file():
        print(f"missing {exe}", file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    base = out.with_suffix("") if out.suffix == ".zip" else out
    archive = shutil.make_archive(str(base), "zip", root_dir=str(src.parent), base_dir=src.name)
    archive = Path(archive)
    if archive.resolve() != out.resolve():
        archive.replace(out)
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
