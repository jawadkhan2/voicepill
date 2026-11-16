"""Prepare the Parakeet TDT 0.6B v2 ONNX assets VoicePill downloads.

Source: istupakov/parakeet-tdt-0.6b-v2-onnx on Hugging Face (NVIDIA's
CC-BY-4.0 checkpoint exported for onnx-asr). Two changes are made so the whole
model runs on ONNX Runtime's CUDA provider, fast:

1. nemo128.onnx (the NeMo log-mel featurizer) computes its STFT with the ONNX
   `STFT` op in float64. ONNX Runtime has no CUDA kernel for it, so it would
   silently run on the CPU. It is replaced by a float32 Conv1d against a
   windowed DFT basis (|X|^2 summed over re/im) — numerically equal to ~3e-5.
2. The encoder (2.4 GB fp32, external data) is converted to fp16 with fp32
   inputs/outputs: half the VRAM, faster on tensor cores, a single file under
   GitHub's 2 GB asset limit, and identical transcripts on our test clips.

Usage (Python 3.12 with: onnx onnxruntime onnxconverter-common numpy):
    python -I scripts/export_parakeet.py <istupakov-model-dir> <out-dir>

Outputs parakeet-tdt-0.6b-v2-{preprocessor.onnx,encoder.onnx,
decoder_joint.onnx,vocab.txt}, the names models.rs downloads from the
`parakeet-v1` release.
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnxruntime.transformers.float16 import convert_float_to_float16

PREFIX = "parakeet-tdt-0.6b-v2"


def stft_to_conv(src: Path, dst: Path) -> None:
    m = onnx.load(src)
    g = m.graph
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}

    stft = next(n for n in g.node if n.op_type == "STFT")
    cast64 = next(n for n in g.node if n.output[0] == stft.input[0])
    rss = next(n for n in g.node if n.op_type == "ReduceSumSquare" and n.input[0] == stft.output[0])
    cast32 = next(n for n in g.node if n.input and n.input[0] == rss.output[0])

    window = inits[stft.input[2]].astype(np.float64)
    hop = int(inits[stft.input[1]])
    n_fft = window.shape[0]
    bins = n_fft // 2 + 1
    k = np.arange(bins)[:, None]
    t = np.arange(n_fft)[None, :]
    ang = 2 * np.pi * k * t / n_fft
    basis = np.concatenate([np.cos(ang) * window, -np.sin(ang) * window]).astype(np.float32)
    g.initializer.extend(
        [
            numpy_helper.from_array(basis[:, None, :], "stft_basis"),
            numpy_helper.from_array(np.array([0, 2, bins, -1], dtype=np.int64), "stft_shape4"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), "stft_ax1"),
        ]
    )
    new = [
        helper.make_node("Unsqueeze", [cast64.input[0], "stft_ax1"], ["stft_sig3"]),
        helper.make_node("Conv", ["stft_sig3", "stft_basis"], ["stft_conv"], strides=[hop], kernel_shape=[n_fft]),
        helper.make_node("Mul", ["stft_conv", "stft_conv"], ["stft_sq"]),
        helper.make_node("Reshape", ["stft_sq", "stft_shape4"], ["stft_sq4"]),
        helper.make_node("ReduceSum", ["stft_sq4", "stft_ax1"], ["stft_pow"], keepdims=0),
        helper.make_node("Transpose", ["stft_pow"], [cast32.output[0]], perm=[0, 2, 1]),
    ]
    idx = list(g.node).index(cast64)
    for n in (cast64, stft, rss, cast32):
        g.node.remove(n)
    for i, n in enumerate(new):
        g.node.insert(idx + i, n)
    for name in (stft.input[1], stft.input[2]):
        if not any(name in n.input for n in g.node):
            g.initializer.remove(next(i for i in g.initializer if i.name == name))
    onnx.checker.check_model(m)
    onnx.save(m, dst)


def encoder_fp16(src: Path, dst: Path) -> None:
    m = onnx.load(src)  # pulls encoder-model.onnx.data
    m16 = convert_float_to_float16(m, keep_io_types=True, disable_shape_infer=True)
    onnx.save(m16, dst)


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    stft_to_conv(src / "nemo128.onnx", out / f"{PREFIX}-preprocessor.onnx")
    encoder_fp16(src / "encoder-model.onnx", out / f"{PREFIX}-encoder.onnx")
    shutil.copy(src / "decoder_joint-model.onnx", out / f"{PREFIX}-decoder_joint.onnx")
    shutil.copy(src / "vocab.txt", out / f"{PREFIX}-vocab.txt")
    print("wrote", *sorted(p.name for p in out.iterdir()), sep="\n  ")


if __name__ == "__main__":
    main()
