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


def session_options(providers: list[str]) -> ort.SessionOptions:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if "DmlExecutionProvider" in providers:
        # DirectML does not support memory-pattern planning or parallel execution.
        # Left on, fused DML nodes fail at run time (DmlFusedNode_* non-zero status).
        opts.enable_mem_pattern = False
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return opts


def is_cpu_only(session: ort.InferenceSession) -> bool:
    return [p for p in session.get_providers() if p != "CPUExecutionProvider"] == []


def open_session(model_path: Path, providers: list[str]) -> tuple[ort.InferenceSession, str]:
    """Open the graph. If the GPU provider fails to load, retry on CPU."""
    try:
        session = ort.InferenceSession(
            str(model_path), sess_options=session_options(providers), providers=list(providers)
        )
    except Exception as exc:
        if providers == CPU_ONLY:
            raise
        print(f"demucs_onnx_worker: {providers[0]} failed to load, using CPU: {exc}",
              file=sys.stderr)
        session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options(CPU_ONLY),
            providers=CPU_ONLY,
        )
    return session, device_name(list(session.get_providers()))


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
    cpu_fallback: tp.Callable[[], ort.InferenceSession] | None = None,
) -> ort.InferenceSession:
    """Separate one track. Returns the session to reuse for the next track.

    When the GPU session raises during ``run`` and ``cpu_fallback`` is given,
    the track restarts on a CPU session, which is returned.
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

    try:
        stems = separate_array(session, mix, overlap=overlap, on_chunk=on_chunk)
    except Exception as exc:
        if cpu_fallback is None or is_cpu_only(session):
            raise
        gpu = device_name(list(session.get_providers()))
        reason = str(exc).replace("\n", " ")
        print(f"demucs_onnx_worker: {gpu} inference failed, retrying on CPU: {reason}",
              file=sys.stderr)
        session = cpu_fallback()
        emit("status", stage="fallback_cpu", device="cpu", reason=reason[:300])
        emit("separating", track=str(audio_path), done=0, total=n_chunks, pct=0.0)
        stems = separate_array(session, mix, overlap=overlap, on_chunk=on_chunk)
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
