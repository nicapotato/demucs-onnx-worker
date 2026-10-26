"""GUI-facing CLI. Same argv as demucs_mlx_worker / demucs_cuda_worker."""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

from demucs_onnx.providers import resolve_providers

from . import events
from .device import gpu_name
from .model_path import MODEL_NAME, executable_dir, find_model
from .separate import separate_file, sources

VERSION = "0.1.1"


def _batch_size(value: str) -> int | str:
    if value.lower() == "auto":
        return "auto"
    try:
        n = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--batch-size must be an integer or 'auto'") from exc
    if n <= 0:
        raise argparse.ArgumentTypeError("--batch-size must be > 0")
    return n


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="demucs_onnx_worker",
        description="htdemucs_6s stem separation via ONNX Runtime (demucs-ui-app worker)",
    )
    parser.add_argument("--version", action="version", version=f"demucs-onnx-worker {VERSION}")
    parser.add_argument("tracks", nargs="*", help="Audio files to separate")
    parser.add_argument("-n", "--name", default=MODEL_NAME, help="Model name (only htdemucs_6s)")
    parser.add_argument("-o", "--out", default="separated", help="Output directory")
    parser.add_argument("--track-name", default=None, help="Output subfolder (one track only)")
    parser.add_argument("--weights", default=None, help="Path to the .onnx file")
    parser.add_argument(
        "--providers",
        default="auto",
        help="auto, cpu, coreml, cuda, or dml. Default: auto for this build's ORT wheel",
    )
    parser.add_argument("--segment", type=float, default=None, help="Accepted and ignored")
    parser.add_argument("--overlap", type=float, default=0.25, help="Overlap ratio")
    parser.add_argument("--shifts", type=int, default=1, help="Accepted; this worker runs one pass")
    parser.add_argument("--seed", type=int, default=None, help="Accepted and ignored")
    parser.add_argument("-b", "--batch-size", type=_batch_size, default="auto")
    parser.add_argument("--no-split", action="store_true", help="Accepted and ignored")
    parser.add_argument("--attention", choices=["fp32", "fp16"], default=None)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--write-workers", type=int, default=1)
    parser.add_argument("--prefetch-tracks", type=int, default=0, help="Accepted and ignored")
    parser.add_argument("--mp3", action="store_true", help="Write MP3 stems instead of WAV")
    parser.add_argument("--mp3-bitrate", type=int, default=128, help="MP3 bitrate (kbps)")
    parser.add_argument("--progress-json", action="store_true", help="JSON events on stdout")
    parser.add_argument("--offline", action="store_true", help="Accepted; never downloads")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _validate(args: argparse.Namespace) -> None:
    if args.name not in (MODEL_NAME, "6s", "htdemucs6s"):
        raise SystemExit(f"only {MODEL_NAME} is bundled (got {args.name})")
    if args.shifts < 0:
        raise SystemExit("--shifts must be >= 0")
    if not (0.0 <= args.overlap < 1.0):
        raise SystemExit("--overlap must be in [0, 1)")
    if args.mp3_bitrate < 32 or args.mp3_bitrate > 320:
        raise SystemExit("--mp3-bitrate must be in [32, 320]")
    if args.track_name is not None and len(args.tracks) != 1:
        raise SystemExit("--track-name requires exactly one input track")
    if args.track_name is not None and ("/" in args.track_name or "\\" in args.track_name):
        raise SystemExit("--track-name must be a single path segment")


def _one_line(exc: BaseException) -> str:
    text = str(exc).replace("\n", " ").strip()
    return text or exc.__class__.__name__


def _run(args: argparse.Namespace) -> int:
    from .separate import CPU_ONLY, open_session

    events.emit("status", stage="loading_model", model=MODEL_NAME)
    model_path = find_model(executable_dir(sys.argv[0] if sys.argv else None), args.weights)
    providers = resolve_providers(args.providers)
    session, device = open_session(model_path, providers)

    def cpu_fallback():
        return open_session(model_path, CPU_ONLY)[0]

    ready = {"model": MODEL_NAME, "device": device, "sources": list(sources())}
    gpu = gpu_name(device)
    if gpu:
        ready["gpu"] = gpu
    events.emit("status", stage="model_ready", **ready)
    for track in args.tracks:
        path = Path(track)
        if not path.is_file():
            raise FileNotFoundError(f"input not found: {path}")
        name = args.track_name or path.stem
        session = separate_file(
            session,
            path,
            Path(args.out),
            name,
            mp3=args.mp3,
            bitrate_kbps=args.mp3_bitrate,
            overlap=args.overlap,
            cpu_fallback=cpu_fallback,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except (OSError, ValueError):
            pass

    args = build_parser().parse_args(argv)
    events.set_enabled(args.progress_json)
    logging.getLogger("demucs_onnx").setLevel(logging.INFO if args.verbose else logging.WARNING)

    if args.list_models:
        print(MODEL_NAME)
        return 0
    if not args.tracks:
        build_parser().print_help(sys.stderr)
        return 2
    _validate(args)

    try:
        return _run(args)
    except Exception as exc:
        events.emit("error", message=_one_line(exc))
        # stdout carries one-line JSON for the GUI; the full traceback goes to stderr.
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
