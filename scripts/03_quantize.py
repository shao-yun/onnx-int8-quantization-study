"""Quantize ONE model with ONE configuration (good for learning; 05_ablation.py runs a whole grid).

Example:
  python scripts/03_quantize.py --model resnet50 --per-channel --method minmax --calib-size 128
Output: work/models/<model>_int8_<tag>.onnx
"""
import argparse

from common import MODELS, Paths, load_calibration
from quantlib import METHODS, quantize_int8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50", choices=sorted(MODELS))
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--per-channel", action="store_true", help="one scale per output channel for weights")
    ap.add_argument("--method", default="minmax", choices=sorted(METHODS))
    ap.add_argument("--format", default="QDQ", choices=["QDQ", "QOperator"])
    ap.add_argument("--calib-size", type=int, default=128, help="number of TRAIN images used for calibration")
    ap.add_argument("--keep-first-last-fp32", action="store_true")
    ap.add_argument("--reduce-range", action="store_true", help="7-bit activations; try if accuracy collapses on older CPUs")
    a = ap.parse_args()
    paths = Paths(a.work_dir).make()

    fp32 = paths.models / f"{a.model}_fp32.onnx"
    if not fp32.exists():
        raise SystemExit(f"{fp32} not found. Run scripts/02_export_onnx.py first.")
    tag = ("pc" if a.per_channel else "pt") + f"-{a.method}"
    if a.format != "QDQ":
        tag += "-qop"
    if a.keep_first_last_fp32:
        tag += "-keep-first-last-fp32"
    out = paths.models / f"{a.model}_int8_{tag}.onnx"

    print(f"Loading {a.calib_size} calibration images from the TRAIN split ...")
    calib = load_calibration(paths, a.calib_size)
    print(f"Quantizing -> {out.name}")
    quantize_int8(fp32, out, calib, a.per_channel, a.method, a.format, a.keep_first_last_fp32, a.reduce_range)
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
