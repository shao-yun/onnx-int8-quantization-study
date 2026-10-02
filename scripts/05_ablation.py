"""Run the whole quantization grid, then compare everything in one table.

Example:
  python scripts/05_ablation.py --model resnet50
Configurations (all static PTQ, QDQ format, INT8):
  pt-minmax          per-tensor weights, min/max calibration   (the naive baseline)
  pc-minmax          per-channel weights, min/max
  pc-percentile      per-channel weights, percentile (99.99) calibration
  pc-entropy         per-channel weights, entropy (KL) calibration
  pc-minmax-keep-first-last-fp32   like pc-minmax but first Conv and last FC stay in FP32 (mixed precision)
"""
import argparse

from common import MODELS, Paths, evaluate_models, load_calibration, write_results
from quantlib import quantize_int8

CONFIGS = [
    dict(tag="pt-minmax", per_channel=False, method="minmax", keep=False),
    dict(tag="pc-minmax", per_channel=True, method="minmax", keep=False),
    dict(tag="pc-percentile", per_channel=True, method="percentile", keep=False),
    dict(tag="pc-entropy", per_channel=True, method="entropy", keep=False),
    dict(tag="pc-minmax-keep-first-last-fp32", per_channel=True, method="minmax", keep=True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50", choices=sorted(MODELS))
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--calib-size", type=int, default=128)
    ap.add_argument("--val-limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--bench-threads", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="re-quantize even if the .onnx already exists")
    a = ap.parse_args()
    paths = Paths(a.work_dir).make()

    fp32 = paths.models / f"{a.model}_fp32.onnx"
    if not fp32.exists():
        raise SystemExit(f"{fp32} not found. Run scripts/02_export_onnx.py first.")

    calib = load_calibration(paths, a.calib_size)
    made = [fp32]
    for c in CONFIGS:
        out = paths.models / f"{a.model}_int8_{c['tag']}.onnx"
        if out.exists() and not a.force:
            print(f"[skip] {out.name} exists (use --force to redo)")
        else:
            print(f"[quantize] {c['tag']}")
            quantize_int8(fp32, out, calib, c["per_channel"], c["method"], "QDQ", c["keep"])
        made.append(out)

    rows = evaluate_models(made, paths, a.val_limit, a.threads, a.bench_threads)
    write_results(rows, paths, f"{a.model}_ablation")


if __name__ == "__main__":
    main()
