"""Compare models: size, top-1 accuracy, agreement with FP32, batch-1 latency.

Example:
  python scripts/04_evaluate.py --model resnet50
Reads every work/models/<model>_*.onnx (except *_prep.onnx) and writes work/results/<model>_results.{md,json}.
"""
import argparse

from common import MODELS, Paths, evaluate_models, write_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50", choices=sorted(MODELS))
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--val-limit", type=int, default=None, help="use a random subset of the validation set")
    ap.add_argument("--threads", type=int, default=0, help="threads for the accuracy pass (0 = ORT default)")
    ap.add_argument("--bench-threads", type=int, default=1, help="threads for the latency benchmark")
    a = ap.parse_args()
    paths = Paths(a.work_dir).make()

    files = [p for p in sorted(paths.models.glob(f"{a.model}_*.onnx")) if not p.stem.endswith("_prep")]
    if not files:
        raise SystemExit("No models found. Run 02_export_onnx.py and 03_quantize.py first.")
    rows = evaluate_models(files, paths, a.val_limit, a.threads, a.bench_threads)
    write_results(rows, paths, f"{a.model}_results")


if __name__ == "__main__":
    main()
