"""Check that your Python environment works, in a few minutes, WITHOUT any download.

It builds a tiny fake dataset (random noise images) in a temp folder, exports a
randomly-initialised ResNet-18 and runs quantize -> evaluate -> ablation on it.
The accuracy numbers it prints are MEANINGLESS. It only proves the pipeline runs.

Run:  python scripts/smoke_test.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from common import IMAGENETTE

HERE = Path(__file__).resolve().parent


def make_fake_imagenette(root: Path, n_train=12, n_val=8):
    rng = np.random.default_rng(0)
    for split, n in (("train", n_train), ("val", n_val)):
        for wnid in IMAGENETTE:
            d = root / "imagenette2-320" / split / wnid
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                arr = rng.integers(0, 256, (320, 400, 3), dtype=np.uint8)
                Image.fromarray(arr).save(d / f"{wnid}_{i}.JPEG")


def run(*args):
    cmd = [sys.executable, str(HERE / args[0]), *args[1:]]
    print("\n$", " ".join(Path(c).name if i == 1 else c for i, c in enumerate(cmd[1:], 1)))
    subprocess.run(cmd, check=True, cwd=str(HERE))


def main():
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "work"
        make_fake_imagenette(work / "data")
        w = ["--work-dir", str(work)]
        run("02_export_onnx.py", "--model", "resnet18", "--random-weights", *w)
        run("03_quantize.py", "--model", "resnet18", "--per-channel", "--calib-size", "16", *w)
        run("05_ablation.py", "--model", "resnet18", "--calib-size", "16", "--val-limit", "40", *w)
    print("\nSMOKE TEST PASSED: the pipeline runs on this machine. Ignore the numbers above.")


if __name__ == "__main__":
    main()
