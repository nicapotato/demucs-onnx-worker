import os
import subprocess
import sys
import zipfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "package_release_zip.py"


def test_zip_root_is_the_onedir(tmp_path: Path) -> None:
    src = tmp_path / "demucs_onnx_worker"
    src.mkdir()
    exe_name = "demucs_onnx_worker.exe" if os.name == "nt" else "demucs_onnx_worker"
    (src / exe_name).write_bytes(b"bin")
    (src / "lib").mkdir()
    (src / "lib" / "note.txt").write_text("ok")
    out = tmp_path / "demucs-onnx-worker-linux-x64-v0.1.0.zip"

    subprocess.check_call([sys.executable, str(SCRIPT), "--src", str(src), "--out", str(out)])

    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
    assert f"demucs_onnx_worker/{exe_name}" in names
    assert "demucs_onnx_worker/lib/note.txt" in names
