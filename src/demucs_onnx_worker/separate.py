"""One-pass overlap-add over the bundled htdemucs_6s ONNX graph.

The graph input is fixed at ``(1, 2, 343980)``. Random shifts are not
applied: demucs-onnx itself runs a single overlap-add, and ``--shifts``
on this worker is accepted only so the GUI's flag set still parses.
"""

from __future__ import annotations

import sys
import typing as tp
from pathlib import Path

import numpy as np
import onnxruntime as ort
from demucs_onnx._audio import load_audio, resample_to_native, write_audio
from demucs_onnx._hub import MODEL_REGISTRY
from demucs_onnx.inference import N_CHANNELS, N_SAMPLES, SAMPLE_RATE

from .chunking import chunk_count, chunk_span, transition_window
from .device import device_name
from .errors import CPU_HINT, GpuFailed, describe, gpu_failed_message
from .events import emit

SOURCES: tuple[str, ...] = ("drums", "bass", "other", "vocals", "guitar", "piano")


def sources() -> tuple[str, ...]:
    registered = tuple(MODEL_REGISTRY["htdemucs_6s"].sources)
    if registered != SOURCES:
        raise RuntimeError(
            f"demucs-onnx htdemucs_6s stem order changed: {registered} != {SOURCES}"
        )
    return SOURCES


CPU_ONLY = ["CPUExecutionProvider"]


DML_DISABLE_FUSION = "ep.dml.disable_graph_fusion"


def session_options(providers: list[str], *, dml_fusion: bool = True) -> ort.SessionOptions:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if "DmlExecutionProvider" in providers:
        # DirectML does not support memory-pattern planning or parallel execution.
        # Left on, fused DML nodes fail at run time (DmlFusedNode_* non-zero status).
        opts.enable_mem_pattern = False
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if not dml_fusion:
            # One DML dispatch per ONNX node instead of one fused graph. Slower, but
            # avoids DmlFusedNode failures and names the failing node if one remains.
            opts.add_session_config_entry(DML_DISABLE_FUSION, "1")
    return opts


def is_cpu_only(session: ort.InferenceSession) -> bool:
    return [p for p in session.get_providers() if p != "CPUExecutionProvider"] == []


def open_session(
    model_path: Path, providers: list[str], *, dml_fusion: bool = True, strict: bool = False
) -> tuple[ort.InferenceSession, str]:
    """Open the graph. If the GPU provider fails to load, retry on CPU.

    ``strict`` (``--device gpu``) raises :class:`GpuFailed` instead of retrying.
    """
    try:
        session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options(providers, dml_fusion=dml_fusion),
            providers=list(providers),
        )
    except Exception as exc:
        if providers == CPU_ONLY:
            raise
        if strict:
            raise GpuFailed(gpu_failed_message(providers, "load", exc)) from exc
        print(f"demucs_onnx_worker: {providers[0]} failed to load, using CPU: {describe(exc)}",
              file=sys.stderr)
        session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options(CPU_ONLY),
            providers=CPU_ONLY,
        )
    if strict and providers != CPU_ONLY and is_cpu_only(session):
        raise GpuFailed(
            f"{providers[0]} was not enabled for this session (only CPU loaded). {CPU_HINT}"
        )
    return session, device_name(list(session.get_providers()))


def open_mode(
    model_path: Path, providers: list[str], mode: str, cache_dir: Path, *, strict: bool = False
) -> tuple[tp.Any, str, str]:
    """Open the session for a DirectML ladder mode (see dml_state).

    Returns ``(session, device, mode)``; ``mode`` changes when a split cannot be
    built (no ``onnx`` package, unexpected graph) or the GPU fails to load.
    With ``strict`` a GPU that fails to load raises :class:`GpuFailed`.
    """
    if mode == "cpu" or providers == CPU_ONLY:
        return (*open_session(model_path, CPU_ONLY), "cpu")
    fusion = mode in ("split", "fused")
    if mode.startswith("split"):
        try:
            from .split_model import SplitSession, ensure_split

            head_path, tail_path = ensure_split(model_path, cache_dir)
        except Exception as exc:
            print(f"demucs_onnx_worker: cannot split the graph ({describe(exc)}); "
                  "running it whole on the GPU", file=sys.stderr)
            mode = "fused" if fusion else "unfused"
        else:
            head, device = open_session(head_path, providers, dml_fusion=fusion, strict=strict)
            if is_cpu_only(head):
                return (*open_session(model_path, CPU_ONLY), "cpu")
            tail, _ = open_session(tail_path, CPU_ONLY)
            return SplitSession(head, tail), device, mode
    session, device = open_session(model_path, providers, dml_fusion=fusion, strict=strict)
    return session, device, ("cpu" if is_cpu_only(session) else mode)


def separate_array(
    session: ort.InferenceSession,
    mix: np.ndarray,
    *,
    overlap: float = 0.25,
    on_chunk=None,
) -> dict[str, np.ndarray]:
    """Run overlap-add. ``mix`` is float32 ``(2, samples)`` at 44.1 kHz."""
    if mix.ndim != 2 or mix.shape[0] != N_CHANNELS:
        raise ValueError(f"expected (channels=2, samples), got {mix.shape}")

    stem_names = sources()
    total_len = int(mix.shape[1])
    n_chunks = chunk_count(total_len, N_SAMPLES, overlap)
    window = transition_window(N_SAMPLES, overlap)
    out = {name: np.zeros((N_CHANNELS, total_len), dtype=np.float32) for name in stem_names}
    weight = np.zeros(total_len, dtype=np.float32)

    for index in range(n_chunks):
        start, end = chunk_span(index, total_len, N_SAMPLES, overlap)
        chunk = mix[:, start:end]
        if chunk.shape[1] < N_SAMPLES:
            chunk = np.pad(chunk, ((0, 0), (0, N_SAMPLES - chunk.shape[1])), mode="constant")
        tensor = chunk[np.newaxis, ...].astype(np.float32, copy=False)
        chunk_len = end - start
        fade = window[:chunk_len]
        stems = session.run(["stems"], {"mix": tensor})[0][0]
        for row, name in enumerate(stem_names):
            out[name][:, start:end] += stems[row, :, :chunk_len] * fade
        weight[start:end] += fade
        if on_chunk is not None:
            on_chunk(index + 1, n_chunks)

    weight = np.maximum(weight, 1e-8)
    for name in out:
        out[name] /= weight
    return out


def separate_file(
    session: ort.InferenceSession,
    audio_path: Path,
    out_dir: Path,
    track_name: str,
    *,
    mp3: bool,
    bitrate_kbps: int,
    overlap: float = 0.25,
    on_failure: tp.Callable[[ort.InferenceSession, Exception], ort.InferenceSession | None]
    | None = None,
) -> ort.InferenceSession:
    """Separate one track. Returns the session to reuse for the next track.

    When ``run`` raises, ``on_failure(session, exc)`` may return a replacement
    session (DML without fusion, then CPU) and the track restarts on it.
    Returning None re-raises.
    """
    stem_names = sources()
    dest = out_dir / track_name
    dest.mkdir(parents=True, exist_ok=True)
    emit("loading", track=str(audio_path), out=str(dest))

    mix, native_sr = load_audio(audio_path, target_sr=SAMPLE_RATE)
    n_chunks = chunk_count(int(mix.shape[1]), N_SAMPLES, overlap)
    emit("separating", track=str(audio_path), done=0, total=n_chunks, pct=0.0)

    def on_chunk(done: int, total: int) -> None:
        pct = round(100.0 * done / total, 2) if total else 0.0
        emit("separating", track=str(audio_path), done=done, total=total, pct=pct)

    while True:
        try:
            stems = separate_array(session, mix, overlap=overlap, on_chunk=on_chunk)
            break
        except Exception as exc:
            replacement = on_failure(session, exc) if on_failure is not None else None
            if replacement is None:
                raise
            session = replacement
            emit("separating", track=str(audio_path), done=0, total=n_chunks, pct=0.0)
    if native_sr != SAMPLE_RATE:
        stems = {
            name: resample_to_native(audio, SAMPLE_RATE, native_sr)
            for name, audio in stems.items()
        }

    ext = "mp3" if mp3 else "wav"
    for name in stem_names:
        path = dest / f"{name}.{ext}"
        write_audio(path, stems[name], native_sr, bitrate_kbps=bitrate_kbps)
        emit("writing", stem=name, path=str(path))

    emit(
        "done",
        track=str(audio_path),
        out=str(dest),
        sources=list(stem_names),
        format=ext,
    )
    return session
