"""Run htdemucs_6s as two graphs: encoder/transformer/decoder on the GPU, iSTFT on CPU.

The exported iSTFT (``/real_istft/*``) is ~16k unrolled nodes: two ConvTranspose
ops with a 4096-tap kernel over 2049 channels and hundreds of ScatterND
overlap-adds. DirectML fails there with E_OUTOFMEMORY (8007000E) on 10 GB
cards; on CPU it is a small fraction of a chunk's time.

The cut is computed from the graph: the *tail* is every node downstream of the
tensors that feed ``/real_istft/``, and the *head* produces exactly the runtime
tensors the tail reads. Split files are cached next to the DML state file.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import onnxruntime as ort

ISTFT_PREFIX = "/real_istft/"
INPUT = "mix"
OUTPUT = "stems"


HEAVY_OPS = frozenset({"Conv", "MatMul", "Gemm", "Softmax", "LayerNormalization", "Attention"})


def _graph_cut(model) -> list[str]:
    """Float tensors the head hands to the tail, in a stable order.

    ``model`` must be shape-inferred. Integer index tensors (Range/Shape chains
    off ``mix``) are left out: the tail also takes ``mix`` and recomputes them.
    """
    import onnx

    floats = {onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16, onnx.TensorProto.DOUBLE}
    graph = model.graph
    elem = {
        vi.name: vi.type.tensor_type.elem_type
        for vi in list(graph.value_info) + list(graph.input) + list(graph.output)
    }
    producer = {out: node for node in graph.node for out in node.output}

    runtime: set[str] = {INPUT}
    for node in graph.node:  # ONNX nodes are topologically sorted
        if any(i in runtime for i in node.input):
            runtime.update(node.output)

    seeds = {
        i
        for node in graph.node
        if node.name.startswith(ISTFT_PREFIX)
        for i in node.input
        if i in runtime and not producer.get(i, node).name.startswith(ISTFT_PREFIX)
    }
    if not seeds:
        raise ValueError(f"no {ISTFT_PREFIX} nodes: not an htdemucs export")

    tail_tensors = set(seeds)
    tail_nodes = []
    for node in graph.node:
        if any(i in tail_tensors for i in node.input):
            tail_tensors.update(node.output)
            tail_nodes.append(node)
    if OUTPUT not in tail_tensors:
        raise ValueError(f"{OUTPUT!r} does not depend on the iSTFT")

    tail_outputs = {o for n in tail_nodes for o in n.output}
    cut = set(seeds)
    for node in tail_nodes:
        for i in node.input:
            if i and i in runtime and i not in tail_outputs:
                cut.add(i)
    return sorted(t for t in cut if elem.get(t) in floats)


def _check_tail(tail) -> None:
    heavy = [n.name for n in tail.graph.node
             if n.op_type in HEAVY_OPS and not n.name.startswith(ISTFT_PREFIX)]
    if heavy:
        raise ValueError(f"tail would recompute {len(heavy)} heavy nodes (first {heavy[0]})")


def split_paths(model_path: Path, cache_dir: Path) -> tuple[Path, Path]:
    st = model_path.stat()
    tag = hashlib.sha1(f"{model_path.name}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()
    base = cache_dir / "split" / tag[:16]
    return base / "head.onnx", base / "tail.onnx"


def ensure_split(model_path: Path, cache_dir: Path) -> tuple[Path, Path]:
    """Write (once) and return ``(head.onnx, tail.onnx)``. Needs the ``onnx`` package."""
    head, tail = split_paths(model_path, cache_dir)
    if head.is_file() and tail.is_file():
        return head, tail
    import onnx
    from onnx import shape_inference
    from onnx.utils import Extractor

    model = shape_inference.infer_shapes(onnx.load(str(model_path)))
    cut = _graph_cut(model)
    extractor = Extractor(model)
    head_model = extractor.extract_model([INPUT], cut)
    tail_model = extractor.extract_model([INPUT, *cut], [OUTPUT])
    _check_tail(tail_model)
    head.parent.mkdir(parents=True, exist_ok=True)
    for path, part in ((head, head_model), (tail, tail_model)):
        tmp = path.with_suffix(".tmp")
        onnx.save(part, str(tmp))
        tmp.replace(path)
    return head, tail


class SplitSession:
    """Duck-types the bits of ``InferenceSession`` the worker uses."""

    def __init__(self, head: ort.InferenceSession, tail: ort.InferenceSession) -> None:
        self.head = head
        self.tail = tail
        self._cut = [o.name for o in head.get_outputs()]

    def get_providers(self) -> list[str]:
        return self.head.get_providers()

    def run(self, output_names, feeds: dict[str, np.ndarray]):
        values = self.head.run(self._cut, feeds)
        tail_feeds = dict(zip(self._cut, values))
        tail_feeds[INPUT] = feeds[INPUT]
        return self.tail.run(output_names, tail_feeds)
