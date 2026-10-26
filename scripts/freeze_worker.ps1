# Freeze the Windows worker with DirectML (NVIDIA, AMD, Intel, then CPU).
# The default `onnxruntime` wheel is CPU-only. This script replaces it in the
# venv with onnxruntime-directml before PyInstaller collects the DLLs.
$ErrorActionPreference = "Stop"
# A uv venv has no pip. Native-command failures must not abort before we check them.
$PSNativeCommandUseErrorActionPreference = $false

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if ($env:DEMUCS_ONNX_PYTHON) {
    $Python = $env:DEMUCS_ONNX_PYTHON
} else {
    $Python = Join-Path $Root ".venv\Scripts\python.exe"
}
if ($env:DEMUCS_ONNX_WORKER_DIST) {
    $Dist = $env:DEMUCS_ONNX_WORKER_DIST
} else {
    $Dist = Join-Path $Root "dist\worker"
}
$Work = Join-Path $Root "dist\pyi"

if (-not (Test-Path $Python)) {
    Write-Error "Missing $Python — run: uv sync --extra dev --extra freeze"
}

Write-Host "switching this venv to onnxruntime-directml (CPU wheel cannot see DirectML)"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv is required (the project venv does not include pip)"
}
& uv pip uninstall --python $Python onnxruntime
& uv pip install --python $Python "onnxruntime-directml==1.24.4"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    & uv pip install --python $Python "pyinstaller>=6"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

& $Python -c "import onnxruntime, demucs_onnx, lameenc, soundfile, soxr; print('onnxruntime', onnxruntime.__version__, onnxruntime.get_available_providers())"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$Onedir = Join-Path $Dist "demucs_onnx_worker"
if (Test-Path $Onedir) { Remove-Item -Recurse -Force $Onedir }
if (Test-Path $Work) { Remove-Item -Recurse -Force $Work }
New-Item -ItemType Directory -Force -Path $Work | Out-Null

$Entry = Join-Path $Root "src\demucs_onnx_worker\__main__.py"
& $Python -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --name demucs_onnx_worker `
    --paths (Join-Path $Root "src") `
    --collect-submodules demucs_onnx_worker `
    --collect-all onnxruntime `
    --collect-all soundfile `
    --collect-all soxr `
    --collect-all lameenc `
    --hidden-import demucs_onnx `
    --hidden-import demucs_onnx.inference `
    --hidden-import demucs_onnx._audio `
    --hidden-import demucs_onnx._hub `
    --hidden-import demucs_onnx.providers `
    --hidden-import numpy `
    --exclude-module huggingface_hub `
    --exclude-module torch `
    --exclude-module torchaudio `
    --exclude-module tensorflow `
    --exclude-module demucs_onnx.export `
    --distpath $Dist `
    --workpath (Join-Path $Work "build") `
    --specpath $Work `
    $Entry
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$Bin = Join-Path $Onedir "demucs_onnx_worker.exe"
if (-not (Test-Path $Bin)) { Write-Error "missing $Bin" }
Write-Host "worker: $Bin"
& $Bin --version
& $Bin --list-models
