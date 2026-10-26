"""Chunk edges for the fixed 7.8 s ONNX graph.

Same overlap-add as ``demucs_onnx.inference``: quarter-segment stride and a
triangular window. Kept here so the worker can emit progress per chunk
without calling the library's tqdm loop.
"""

from __future__ import annotations

import numpy as np


def transition_samples(segment: int, overlap: float) -> int:
    if overlap == 0.25:
        return segment // 4
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    return int(segment * overlap)


def transition_window(segment: int, overlap: float = 0.25) -> np.ndarray:
    transition = transition_samples(segment, overlap)
    window = np.ones(segment, dtype=np.float32)
    if transition <= 0:
        return window
    fade = np.linspace(0, 1, transition, dtype=np.float32)
    window[:transition] = fade
    window[-transition:] = fade[::-1]
    return window


def chunk_count(total_len: int, segment: int, overlap: float = 0.25) -> int:
    stride = segment - transition_samples(segment, overlap)
    if stride <= 0:
        raise ValueError("overlap leaves no stride")
    if total_len <= 0:
        return 1
    return max(1, (total_len + stride - 1) // stride)


def chunk_span(index: int, total_len: int, segment: int, overlap: float = 0.25) -> tuple[int, int]:
    stride = segment - transition_samples(segment, overlap)
    start = index * stride
    end = min(start + segment, total_len)
    return start, end
