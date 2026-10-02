"""Static INT8 post-training quantization (PTQ) with ONNX Runtime."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

METHODS = {
    "minmax": CalibrationMethod.MinMax,        # range = observed min/max (outliers stretch it)
    "entropy": CalibrationMethod.Entropy,      # pick range minimizing KL divergence
    "percentile": CalibrationMethod.Percentile,  # clip the extreme tail (99.99 %)
}


class ArrayReader(CalibrationDataReader):
    """Feeds calibration images (float32 NCHW) to the calibrator in small batches."""

    def __init__(self, x: np.ndarray, input_name: str = "input", batch: int = 8):
        self._batches = [{input_name: x[i : i + batch]} for i in range(0, len(x), batch)]
        self._it = iter(self._batches)

    def get_next(self):
        return next(self._it, None)

    def rewind(self):
        self._it = iter(self._batches)


def preprocess_for_quant(fp32_path: Path, out_path: Path) -> Path:
    """ORT's recommended pre-step: shape inference + graph clean-up (e.g. constant folding)."""
    from onnxruntime.quantization.shape_inference import quant_pre_process

    try:
        quant_pre_process(str(fp32_path), str(out_path))
    except Exception as e:  # symbolic shape inference can fail on some graphs
        print(f"  quant_pre_process failed ({e!r}); retrying without symbolic shape inference")
        try:
            quant_pre_process(str(fp32_path), str(out_path), skip_symbolic_shape=True)
        except Exception as e2:
            print(f"  still failing ({e2!r}); quantizing the original graph instead")
            return fp32_path
    return out_path


def find_first_conv_and_last_fc(model_path: Path):
    """Names of the first Conv node and the last Gemm/MatMul node (classic 'sensitive' layers)."""
    m = onnx.load(str(model_path), load_external_data=False)
    first_conv = next((n.name for n in m.graph.node if n.op_type == "Conv"), None)
    last_fc = None
    for n in m.graph.node:
        if n.op_type in ("Gemm", "MatMul"):
            last_fc = n.name
    return [n for n in (first_conv, last_fc) if n]


def quantize_int8(
    fp32_path,
    out_path,
    calib_x: np.ndarray,
    per_channel: bool = True,
    method: str = "minmax",
    fmt: str = "QDQ",
    keep_first_last_fp32: bool = False,
    reduce_range: bool = False,
):
    """Quantize an FP32 ONNX model to INT8 (weights: int8, activations: uint8)."""
    fp32_path, out_path = Path(fp32_path), Path(out_path)
    prep = preprocess_for_quant(fp32_path, out_path.with_name(fp32_path.stem + "_prep.onnx"))
    exclude = find_first_conv_and_last_fc(prep) if keep_first_last_fp32 else []
    if exclude:
        print(f"  keeping in FP32: {exclude}")
    input_name = onnx.load(str(prep), load_external_data=False).graph.input[0].name
    extra = {"CalibPercentile": 99.99} if method == "percentile" else {}
    quantize_static(
        model_input=str(prep),
        model_output=str(out_path),
        calibration_data_reader=ArrayReader(calib_x, input_name),
        quant_format=QuantFormat.QDQ if fmt == "QDQ" else QuantFormat.QOperator,
        per_channel=per_channel,
        reduce_range=reduce_range,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=METHODS[method],
        nodes_to_exclude=exclude or None,
        extra_options=extra or None,
    )
    return out_path
