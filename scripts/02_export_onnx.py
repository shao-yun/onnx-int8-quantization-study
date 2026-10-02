"""Export a pretrained torchvision classifier to ONNX (FP32) and verify the export.

Run:  python scripts/02_export_onnx.py --model resnet50
Output: work/models/<model>_fp32.onnx
"""
import argparse
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

from common import IMAGENETTE, MODELS, Paths, make_session


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50", choices=sorted(MODELS))
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--random-weights", action="store_true",
                    help="SMOKE TEST ONLY: skip the pretrained-weight download. Results are meaningless.")
    a = ap.parse_args()
    paths = Paths(a.work_dir).make()

    import torch
    import torchvision

    weights = torchvision.models.get_weight(MODELS[a.model])
    if a.random_weights:
        print("!! --random-weights: smoke test only, do not report any numbers from this model !!")
        model = getattr(torchvision.models, a.model)(weights=None)
    else:
        model = getattr(torchvision.models, a.model)(weights=weights)  # downloads once, cached by torch
        cats = weights.meta["categories"]
        for wnid, (idx, name) in IMAGENETTE.items():
            assert cats[idx] == name, f"label map mismatch at {idx}: {cats[idx]!r} != {name!r}"
        print("Label map check passed: the 10 Imagenette classes match the model's ImageNet indices.")
    model.eval()

    out = paths.models / f"{a.model}_fp32.onnx"
    dummy = torch.randn(1, 3, 224, 224)
    kwargs = dict(
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    try:  # newer PyTorch: use the classic TorchScript-based exporter (no extra dependency)
        torch.onnx.export(model, dummy, str(out), dynamo=False, **kwargs)
    except TypeError:  # older PyTorch has no `dynamo` argument
        torch.onnx.export(model, dummy, str(out), **kwargs)
    print(f"Exported: {out}")

    import onnx

    onnx.checker.check_model(onnx.load(str(out)))
    # Correctness check: PyTorch vs ONNX Runtime on the same random batch.
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        ref = model(x).numpy()
    got = make_session(out).run(None, {"input": x.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    print(f"max |PyTorch - ONNX Runtime| logit difference: {diff:.2e}")
    assert diff < 1e-3, "Export mismatch - do not continue until this is fixed."
    print("Export verified.")


if __name__ == "__main__":
    main()
