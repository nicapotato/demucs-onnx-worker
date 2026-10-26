# demucs-onnx-worker

Desktop worker for [demucs-ui-app](https://github.com/nicapotato/demucs-ui-app). It runs the published `htdemucs_6s` ONNX graph from [demucs-onnx](https://github.com/StemSplit/demucs-onnx) 0.3.4 and speaks the same CLI as the MLX and CUDA workers:

```text
demucs_onnx_worker track.wav -n htdemucs_6s -o DIR --track-name NAME \
    -b 8 --progress-json --prefetch-tracks 0 --mp3
```

Stems land in `DIR/NAME/{drums,bass,other,vocals,guitar,piano}.mp3` (or `.wav`). Progress is one JSON object per stdout line. `-b`, `--shifts`, `--segment`, and `--no-split` are accepted so the GUI can pass them. The graph is one fixed 7.8 s segment, batch 1, overlap 0.25, one pass. Weights are never downloaded at separation time.

Version: [`project.conf`](project.conf).

## What one build covers

The execution provider is whichever ONNX Runtime wheel was frozen into that build. `auto` picks the best provider that wheel actually contains, and falls back to CPU if the GPU provider fails to load.

| Build | Wheel | GPU | CPU fallback |
|---|---|---|---|
| macOS arm64 | `onnxruntime` | CoreML | yes |
| macOS x64 | `onnxruntime` | none | yes |
| Windows x64 | `onnxruntime-directml` 1.24.4 | DirectML (NVIDIA, AMD, Intel) | yes |
| Linux x64 | `onnxruntime` | none | yes |
| Linux x64, `DEMUCS_ONNX_GPU=1` | `onnxruntime-gpu` | CUDA | yes |

Linux AMD and Intel GPUs stay on the CPU build. CoreML's first session compiles the graph before the first song.

The frozen runtime (Python, numpy, ONNX Runtime, audio libraries) is on the order of 100–160 MB for the CoreML, DirectML, and CPU builds. The CUDA Linux wheel alone is about 235 MB. The model file is separate: `htdemucs_6s_fp16weights.onnx` is about 136 MB and still computes in fp32.

## Dev

```bash
uv sync --extra dev --extra freeze
make test
make models          # curl the ONNX file into models/
uv run demucs-onnx-worker --list-models
uv run demucs-onnx-worker song.wav -o out --track-name song --progress-json --mp3
```

`make models` writes `models/htdemucs_6s_fp16weights.onnx`. The worker also looks in `DEMUCS_ONNX_CACHE`, `DEMUCS_MLX_CACHE`, `DEMUCS_CUDA_CACHE`, and `DEMUCS_TORCH_CACHE`, which is how demucs-ui-app points it at `Contents/Resources/models` or the Windows `models` folder.

## Freeze

```bash
make freeze          # macOS or Linux CPU. Linux NVIDIA: DEMUCS_ONNX_GPU=1 make freeze
```

Windows, from this directory:

```powershell
uv sync --extra dev --extra freeze
pwsh -File scripts/freeze_worker.ps1
```

The Windows script replaces the CPU `onnxruntime` wheel in this venv with `onnxruntime-directml`. The onedir is `dist/worker/demucs_onnx_worker/demucs_onnx_worker` (`.exe` on Windows).

## Release

Bump `VERSION` in `project.conf`, push this repo, then:

```bash
make release-watch
```

That runs the test gate, freezes four workers, and tags `v$VERSION`. The GitHub Release is the permanent pin. The same files are also kept as a workflow artifact for 90 days (`make release PUBLISH_GH_RELEASE=false` uploads the artifact and skips the tag).

| Asset | Runner | Provider |
|---|---|---|
| `demucs-onnx-worker-mac-arm64-v*.zip` | `macos-latest` | CoreML, then CPU |
| `demucs-onnx-worker-mac-x64-v*.zip` | `macos-15-intel` | CPU |
| `demucs-onnx-worker-windows-x64-v*.zip` | `windows-latest` | DirectML, then CPU |
| `demucs-onnx-worker-linux-x64-v*.zip` | `ubuntu-latest` | CPU |
| `htdemucs_6s_fp16weights.onnx` | fetched once | shared graph |
| `SHA256SUMS` | | |

Each zip's root is `demucs_onnx_worker/`. Linux NVIDIA (`DEMUCS_ONNX_GPU=1`) is not in the release. The Apple Silicon app download stays the mlx-c build; the arm64 zip here is the ONNX worker only.

## demucs-ui-app

The GUI flavor is `DMX_VARIANT=onnx` (`DemucsOnnxMac` / `DemucsOnnxWindows` / `DemucsOnnxLinux`). From the app repo, with this checkout as a sibling:

```bash
make -C ../demucs-onnx-worker models
make -C app run-onnx
make -C app process-onnx FILES="song.wav"
```

`DEMUCS_ONNX_PYTHON` selects a dev interpreter (`python -m demucs_onnx_worker`). `DEMUCS_ONNX_WORKER` selects a frozen binary. `--weights` or `DEMUCS_ONNX_MODEL` selects a file.

## License

MIT. Inference is [demucs-onnx](https://github.com/StemSplit/demucs-onnx) (MIT) running the Hybrid Transformer Demucs weights (Meta Research, MIT).
