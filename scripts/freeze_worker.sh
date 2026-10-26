#!/usr/bin/env bash
# Freeze an onedir worker. The ORT wheel already in the venv decides the GPU:
#   macOS arm64  onnxruntime        -> CoreML, then CPU
#   macOS x64    onnxruntime        -> CPU
#   Linux        onnxruntime        -> CPU
#   Linux NVIDIA DEMUCS_ONNX_GPU=1  -> CUDA wheel, then CPU
# Windows uses scripts/freeze_worker.ps1 (DirectML).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${DEMUCS_ONNX_PYTHON:-$ROOT/.venv/bin/python}"
DIST="${DEMUCS_ONNX_WORKER_DIST:-$ROOT/dist/worker}"
WORK="$ROOT/dist/pyi"

if [[ ! -x "$PY" ]]; then
  echo "missing $PY — run: make sync" >&2
  exit 1
fi

py_install() {
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$PY" "$@"
  else
    "$PY" -m pip install "$@"
  fi
}

py_uninstall() {
  if command -v uv >/dev/null 2>&1; then
    uv pip uninstall --python "$PY" "$@" || true
  else
    "$PY" -m pip uninstall -y "$@" >/dev/null 2>&1 || true
  fi
}

if [[ "${DEMUCS_ONNX_GPU:-}" == "1" ]]; then
  echo "installing onnxruntime-gpu into this venv (replaces the CPU wheel)"
  py_uninstall onnxruntime onnxruntime-gpu onnxruntime-directml
  py_install "onnxruntime-gpu>=1.17"
fi

"$PY" -c "import PyInstaller" 2>/dev/null || py_install "pyinstaller>=6"
"$PY" -c "import onnxruntime, demucs_onnx, lameenc, soundfile, soxr; print('onnxruntime', onnxruntime.__version__, onnxruntime.get_available_providers())"

rm -rf "$DIST/demucs_onnx_worker" "$WORK"
mkdir -p "$WORK"

"$PY" -m PyInstaller \
  --noconfirm \
  --clean \
  --onedir \
  --name demucs_onnx_worker \
  --paths "$ROOT/src" \
  --collect-submodules demucs_onnx_worker \
  --collect-all onnxruntime \
  --collect-all soundfile \
  --collect-all soxr \
  --collect-all lameenc \
  --hidden-import demucs_onnx \
  --hidden-import demucs_onnx.inference \
  --hidden-import demucs_onnx._audio \
  --hidden-import demucs_onnx._hub \
  --hidden-import demucs_onnx.providers \
  --hidden-import numpy \
  --exclude-module huggingface_hub \
  --exclude-module torch \
  --exclude-module torchaudio \
  --exclude-module tensorflow \
  --exclude-module demucs_onnx.export \
  --distpath "$DIST" \
  --workpath "$WORK/build" \
  --specpath "$WORK" \
  "$ROOT/src/demucs_onnx_worker/__main__.py"

BIN="$DIST/demucs_onnx_worker/demucs_onnx_worker"
test -x "$BIN"
echo "worker: $BIN"
"$BIN" --version
"$BIN" --list-models
