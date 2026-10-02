"""Shared helpers: paths, Imagenette data, preprocessing, evaluation, benchmarking, reporting.

Design notes
------------
* Everything after the ONNX export needs only numpy + pillow + onnx + onnxruntime.
* Preprocessing is done with PIL + numpy and mirrors torchvision's
  Resize(256) -> CenterCrop(224) -> Normalize(ImageNet mean/std), so the accuracy of the
  FP32 ONNX model matches what PyTorch would give.
* Accuracy is measured on Imagenette (10 ImageNet classes). The model still predicts over
  all 1000 ImageNet classes; a prediction is correct only if it equals the ImageNet index
  of the true class. That is stricter than a 10-way problem.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

# Windows consoles (e.g. cp950) can choke on symbols such as the Greek delta in tables.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WORK_DIR = SCRIPT_DIR.parent / "work"

# Imagenette class folder (WordNet id) -> (index in the 1000-class ImageNet label space, name)
IMAGENETTE = {
    "n01440764": (0, "tench"),
    "n02102040": (217, "English springer"),
    "n02979186": (482, "cassette player"),
    "n03000684": (491, "chain saw"),
    "n03028079": (497, "church"),
    "n03394916": (566, "French horn"),
    "n03417042": (569, "garbage truck"),
    "n03425413": (571, "gas pump"),
    "n03445777": (574, "golf ball"),
    "n03888257": (701, "parachute"),
}

# model name -> torchvision weights enum name (V1 weights use the classic 256/224 preprocessing)
MODELS = {
    "resnet18": "ResNet18_Weights.IMAGENET1K_V1",
    "resnet50": "ResNet50_Weights.IMAGENET1K_V1",
    "mobilenet_v2": "MobileNet_V2_Weights.IMAGENET1K_V1",
}

_MEAN255 = (np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255).reshape(1, 3, 1, 1)
_STD255 = (np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255).reshape(1, 3, 1, 1)


class Paths:
    """All files live under one work directory (default: <project>/work)."""

    def __init__(self, work_dir=None):
        self.root = Path(work_dir or DEFAULT_WORK_DIR).resolve()
        self.data = self.root / "data"
        self.models = self.root / "models"
        self.cache = self.root / "cache"
        self.results = self.root / "results"
        self.imagenette = self.data / "imagenette2-320"

    def make(self):
        for p in (self.data, self.models, self.cache, self.results):
            p.mkdir(parents=True, exist_ok=True)
        return self


# --------------------------------------------------------------------------- data


def list_split(paths: Paths, split: str):
    """Return [(image_path, imagenet_index), ...] for 'train' or 'val'."""
    root = paths.imagenette / split
    if not root.is_dir():
        raise FileNotFoundError(
            f"{root} not found. Run scripts/01_get_data.py first (or use --work-dir)."
        )
    items = []
    for wnid, (idx, _name) in sorted(IMAGENETTE.items()):
        folder = root / wnid
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in {".jpeg", ".jpg", ".png"}:
                items.append((p, idx))
    if not items:
        raise FileNotFoundError(f"No images found under {root}")
    return items


def load_image_u8(path) -> np.ndarray:
    """Resize shorter side to 256 (bilinear), center-crop 224. Returns HxWx3 uint8."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w <= h:
        nw, nh = 256, int(256 * h / w)
    else:
        nw, nh = int(256 * w / h), 256
    img = img.resize((nw, nh), Image.BILINEAR)
    left = int(round((nw - 224) / 2.0))
    top = int(round((nh - 224) / 2.0))
    img = img.crop((left, top, left + 224, top + 224))
    return np.asarray(img, dtype=np.uint8)


def to_model_input(x_u8_nhwc: np.ndarray) -> np.ndarray:
    """uint8 NHWC -> float32 NCHW, ImageNet-normalized."""
    x = np.ascontiguousarray(np.asarray(x_u8_nhwc).transpose(0, 3, 1, 2)).astype(np.float32)
    return (x - _MEAN255) / _STD255


def load_val(paths: Paths, limit=None):
    """Validation images as uint8 NHWC + labels. Cached on disk after the first call."""
    paths.make()
    cx, cy = paths.cache / "val_u8.npy", paths.cache / "val_y.npy"
    if not (cx.exists() and cy.exists()):
        items = list_split(paths, "val")
        print(f"Preprocessing {len(items)} validation images (one-time, cached) ...")
        x = np.stack([load_image_u8(p) for p, _ in items])
        y = np.array([lbl for _, lbl in items], dtype=np.int64)
        np.save(cx, x)
        np.save(cy, y)
    x = np.load(cx, mmap_mode="r")
    y = np.load(cy)
    if limit and limit < len(y):
        idx = np.sort(np.random.default_rng(0).permutation(len(y))[:limit])
        return np.asarray(x[idx]), y[idx]
    return np.asarray(x), y


def load_calibration(paths: Paths, n: int, seed: int = 0) -> np.ndarray:
    """n random TRAIN images (never validation images!) as float32 NCHW."""
    items = list_split(paths, "train")
    rng = np.random.default_rng(seed)
    pick = rng.permutation(len(items))[: min(n, len(items))]
    x = np.stack([load_image_u8(items[i][0]) for i in pick])
    return to_model_input(x)


# ----------------------------------------------------------------- inference / metrics


def make_session(model_path, threads: int = 0):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads  # 0 = let ONNX Runtime decide
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])


def predict(sess, x_u8: np.ndarray, batch: int = 32) -> np.ndarray:
    name = sess.get_inputs()[0].name
    preds = []
    for i in range(0, len(x_u8), batch):
        xb = to_model_input(x_u8[i : i + batch])
        logits = sess.run(None, {name: xb})[0]
        preds.append(np.argmax(logits, axis=1))
    return np.concatenate(preds)


def latency_ms(sess, warmup: int = 20, iters: int = 100):
    """Batch-1 latency on a fixed random input (CPU latency does not depend on pixel values)."""
    name = sess.get_inputs()[0].name
    x = np.random.default_rng(0).standard_normal((1, 3, 224, 224)).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {name: x})
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        times.append((time.perf_counter() - t0) * 1000.0)
    times = np.array(times)
    return float(np.median(times)), float(np.percentile(times, 95))


def file_size_mb(path) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def count_ops(model_path) -> Counter:
    import onnx

    m = onnx.load(str(model_path), load_external_data=False)
    return Counter(n.op_type for n in m.graph.node)


def env_info() -> dict:
    import onnxruntime as ort

    return {
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
    }


# ---------------------------------------------------------------------------- reporting


def evaluate_models(model_paths, paths: Paths, val_limit=None, threads: int = 0, bench_threads: int = 1):
    """Accuracy, agreement with FP32, size and latency for each model. Returns list of dict rows."""
    model_paths = [Path(p) for p in model_paths]
    x_val, y_val = load_val(paths, val_limit)
    print(f"Evaluating on {len(y_val)} validation images.")
    rows, fp32_preds = [], None
    # Evaluate FP32 first so the other rows can be compared against it.
    model_paths.sort(key=lambda p: 0 if "_fp32" in p.stem else 1)  # stable: FP32 first, rest keep given order
    for mp in model_paths:
        print(f"  -> {mp.name}")
        preds = predict(make_session(mp, threads), x_val)
        acc = float((preds == y_val).mean() * 100)
        if fp32_preds is None and "_fp32" in mp.stem:
            fp32_preds = preds
        agree = float((preds == fp32_preds).mean() * 100) if fp32_preds is not None else None
        med, p95 = latency_ms(make_session(mp, bench_threads))
        ops = count_ops(mp)
        rows.append(
            {
                "model": mp.stem,
                "size_mb": round(file_size_mb(mp), 2),
                "top1_pct": round(acc, 2),
                "agree_with_fp32_pct": None if agree is None else round(agree, 2),
                "latency_ms_median": round(med, 2),
                "latency_ms_p95": round(p95, 2),
                "qdq_nodes": int(ops.get("QuantizeLinear", 0) + ops.get("DequantizeLinear", 0)),
                "n_images": int(len(y_val)),
                "bench_threads": bench_threads,
            }
        )
    base = next((r for r in rows if "_fp32" in r["model"]), None)
    for r in rows:
        r["delta_top1_pp"] = None if base is None else round(r["top1_pct"] - base["top1_pct"], 2)
        r["speedup_x"] = None if base is None else round(base["latency_ms_median"] / r["latency_ms_median"], 2)
    return rows


def write_results(rows, paths: Paths, name: str = "results"):
    paths.make()
    payload = {"env": env_info(), "rows": rows}
    (paths.results / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    bt = rows[0]["bench_threads"] if rows else "?"
    lines = [
        f"<!-- {payload['env']['processor']} | {payload['env']['logical_cpus']} logical CPUs | "
        f"onnxruntime {payload['env']['onnxruntime']} | {payload['env']['platform']} -->",
        "",
        f"| model | size (MB) | top-1 (%) | Δ vs FP32 (pp) | agrees w/ FP32 (%) | latency ms, median ({bt} thread) | p95 ms | speedup | Q/DQ nodes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        f = lambda v, d="-": d if v is None else v  # noqa: E731
        lines.append(
            f"| {r['model']} | {r['size_mb']} | {r['top1_pct']} | {f(r['delta_top1_pp'])} | "
            f"{f(r['agree_with_fp32_pct'])} | {r['latency_ms_median']} | {r['latency_ms_p95']} | "
            f"{f(r['speedup_x'])}x | {r['qdq_nodes']} |"
        )
    lines += ["", f"Images evaluated: {rows[0]['n_images'] if rows else 0}"]
    (paths.results / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved: {paths.results / (name + '.md')}  and  {paths.results / (name + '.json')}")
