# ONNX INT8 Post-Training Quantization Study

How much do model size, CPU latency and accuracy change when a pretrained image classifier is
converted from FP32 to INT8 with post-training quantization (PTQ) in ONNX Runtime — and which
quantization choices matter (per-tensor vs per-channel weights, calibration method, keeping
sensitive layers in FP32)?

> **Status:** complete. Experiments run on ResNet-50 and MobileNetV2; results and findings below.

## What is measured

| Metric | How |
|---|---|
| Model size | ONNX file size (MB) |
| Accuracy | Top-1 on the Imagenette validation split (about 3.9k images) (10 ImageNet classes). The model predicts over all 1,000 ImageNet classes; a prediction counts only if it equals the true class's ImageNet index. |
| Agreement with FP32 | % of images where the INT8 model gives the same top-1 class as the FP32 model |
| Latency | Batch-1 median and p95 over 100 runs after 20 warm-up runs, ONNX Runtime CPU execution provider, 1 thread (fixed for low noise and easy comparison) |

Models: ResNet-18 / ResNet-50 / MobileNetV2 (torchvision `IMAGENET1K_V1` weights).
Calibration data: 128 random **training** images (never validation images).

Quantization configurations (static PTQ, QDQ format, INT8 weights + UINT8 activations):

| Tag | Weights | Activation range calibration | Notes |
|---|---|---|---|
| `pt-minmax` | per-tensor | min/max | naive baseline |
| `pc-minmax` | per-channel | min/max | |
| `pc-percentile` | per-channel | percentile (99.99) | clips outliers |
| `pc-entropy` | per-channel | entropy (KL divergence) | |
| `pc-minmax-keep-first-last-fp32` | per-channel | min/max | first Conv and last FC kept in FP32 |

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python scripts/smoke_test.py                  # 2-minute environment check, no downloads, numbers meaningless
python scripts/00_toy_quantization.py         # numpy-only intuition demo
python scripts/01_get_data.py                 # Imagenette (~325 MB)
python scripts/02_export_onnx.py --model resnet50
python scripts/05_ablation.py --model resnet50   # quantizes 5 configs, evaluates everything
# single configuration, for experimenting:
python scripts/03_quantize.py --model resnet50 --per-channel --method percentile
python scripts/04_evaluate.py --model resnet50
```

Outputs land in `work/` (models, cached data, `results/*.md|json`).

## Results

Machine: AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD, 16 logical CPUs, no AVX-512/VNNI.
ONNX Runtime 1.23.2, Windows-10-10.0.26200-SP0. Imagenette validation split, 3925 images.

All configurations below use `--reduce-range`. An earlier run without it showed much larger
accuracy drops on this CPU (up to -10pp) — see `work/results/resnet50_ablation.md` and
`work/results/mobilenet_v2_ablation.md` for those original numbers, and explain the difference
in Findings below.

### ResNet-50

| model | size (MB) | top-1 (%) | Δ vs FP32 (pp) | agrees w/ FP32 (%) | latency ms, median (1 thread) | p95 ms | speedup | Q/DQ nodes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| resnet50_fp32 | 97.41 | 84.54 | 0.0 | 100.0 | 89.72 | 128.44 | 1.0x | 0 |
| resnet50_int8_pt-minmax | 24.53 | 84.87 | 0.33 | 93.15 | 49.2 | 54.92 | 1.82x | 256 |
| resnet50_int8_pc-minmax | 24.87 | 84.59 | 0.05 | 96.94 | 50.41 | 63.26 | 1.78x | 256 |
| resnet50_int8_pc-percentile | 24.87 | 84.71 | 0.17 | 96.87 | 49.11 | 54.45 | 1.83x | 256 |
| resnet50_int8_pc-entropy | 24.87 | 84.59 | 0.05 | 96.94 | 50.87 | 55.21 | 1.76x | 256 |
| resnet50_int8_pc-minmax-keep-first-last-fp32 | 30.74 | 84.15 | -0.39 | 97.86 | 54.61 | 61.55 | 1.64x | 244 |

### MobileNetV2

| model | size (MB) | top-1 (%) | Δ vs FP32 (pp) | agrees w/ FP32 (%) | latency ms, median (1 thread) | p95 ms | speedup | Q/DQ nodes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mobilenet_v2_fp32 | 13.34 | 78.8 | 0.0 | 100.0 | 7.99 | 9.38 | 1.0x | 0 |
| mobilenet_v2_int8_pt-minmax | 3.49 | 73.07 | -5.73 | 79.39 | 5.42 | 6.1 | 1.47x | 238 |
| mobilenet_v2_int8_pc-minmax | 3.72 | 78.57 | -0.23 | 90.22 | 5.57 | 6.08 | 1.43x | 238 |
| mobilenet_v2_int8_pc-percentile | 3.72 | 78.04 | -0.76 | 91.59 | 5.71 | 6.78 | 1.4x | 238 |
| mobilenet_v2_int8_pc-entropy | 3.72 | 78.57 | -0.23 | 90.22 | 5.5 | 5.75 | 1.45x | 238 |
| mobilenet_v2_int8_pc-minmax-keep-first-last-fp32 | 7.37 | 78.5 | -0.3 | 90.19 | 5.41 | 5.9 | 1.48x | 228 |

## Findings

### 1. `reduce_range` mattered much more than expected on this CPU

In the original full-range INT8 runs, accuracy degraded substantially: ResNet-50 lost about **5.5 percentage points**, while MobileNetV2 lost as much as **10 percentage points** in the affected configurations. After enabling `reduce_range`, the best per-channel configurations recovered to within a few tenths of a percentage point of FP32.

This behavior is consistent with the CPU used for the experiment. The machine does not support AVX-512/VNNI, so ONNX Runtime may use older x86 integer execution paths for `uint8 × int8` operations. With the full 8-bit range, intermediate integer operations can be more vulnerable to saturation. Reducing the quantization range lowers that risk.

Several checks make a preprocessing or export bug less likely:

- The FP32 ONNX model was verified against the original PyTorch model before quantization.
- The same preprocessing and evaluation pipeline was used before and after the change.
- `reduce_range` was the main quantization setting changed.
- Both ResNet-50 and MobileNetV2 improved substantially after enabling it.

The main lesson is that PTQ accuracy is not determined only by the model and calibration algorithm; the CPU's available integer instructions can also affect the result.

### 2. Per-channel quantization is especially important for MobileNetV2

With `reduce_range` enabled, ResNet-50 is relatively insensitive to per-tensor versus per-channel weight quantization:

- per-tensor MinMax: **84.87%**
- per-channel MinMax: **84.59%**

MobileNetV2 behaves very differently:

- per-tensor MinMax: **73.07%**
- per-channel MinMax: **78.57%**

The difference is about **5.5 percentage points**.

A likely reason is MobileNetV2's extensive use of **depthwise convolutions**. Different depthwise channels can have very different weight ranges. With per-tensor quantization, all channels share one scale, so a large-range channel can determine the scale for the entire tensor and reduce the effective resolution available to small-range channels.

Per-channel quantization assigns a separate scale to each output channel, which preserves those smaller weight distributions much better.

ResNet-50 relies primarily on standard convolutions and, in this experiment, its weight tensors were much less sensitive to sharing a single scale.

### 3. Keeping the first and last layers in FP32 was less useful after fixing the range issue

After enabling `reduce_range`, keeping the first convolution and final fully connected layer in FP32 did not improve ResNet-50 accuracy.

Compared with fully quantized per-channel MinMax:

| Configuration | Size | Top-1 | Median latency | FP32 agreement |
|---|---:|---:|---:|---:|
| `pc-minmax` | 24.87 MB | 84.59% | 50.41 ms | 96.94% |
| `pc-minmax-keep-first-last-fp32` | 30.74 MB | 84.15% | 54.61 ms | 97.86% |

The mixed-precision version is larger and slower, while its top-1 accuracy is slightly lower. Its main benefit is higher agreement with FP32.

The earlier full-range experiment made keeping sensitive layers in FP32 appear more useful. After `reduce_range` removed much of the CPU-related quantization error, that advantage largely disappeared. This suggests that part of the earlier benefit came from avoiding problematic INT8 execution in those layers rather than from an inherent requirement that the first and last layers remain FP32.

This is a useful reminder that mixed precision should be justified by measurements rather than applied automatically.

### 4. Top-1 accuracy alone can hide substantial prediction changes

On ResNet-50, `pt-minmax` has the highest measured top-1 accuracy:

- FP32: **84.54%**
- per-tensor MinMax: **84.87%**

However, its agreement with the FP32 model is only **93.15%**, compared with roughly **96.9–97.9%** for the per-channel configurations.

This means the per-tensor model changes substantially more predictions, even though its final accuracy happens to be slightly higher. Some FP32 mistakes may become correct after quantization, while some correct FP32 predictions become wrong. Those changes can partially cancel when only aggregate top-1 accuracy is measured.

Therefore, the `+0.33 pp` result should not be interpreted as evidence that quantization made the model better. On a validation set of only 3,925 images and with a single calibration seed, such a small difference can easily reflect sampling and quantization noise.

Agreement with FP32 is useful because it measures **model fidelity**: how much the quantized model's behavior changed, independently of whether those changes happened to improve or hurt accuracy on this particular dataset.

### 5. Deployment choice: per-channel MinMax for both models

For **ResNet-50**, I would deploy `pc-minmax`.

It reduces the model from **97.41 MB to 24.87 MB**, decreases median CPU latency from **89.72 ms to 50.41 ms** (**1.78× speedup**), and preserves accuracy almost exactly (**84.54% → 84.59%**). It also maintains **96.94% agreement** with the FP32 model.

`pc-percentile` is slightly faster and has slightly higher measured top-1 accuracy, but the differences are small enough that they are difficult to interpret without repeated calibration seeds or confidence intervals. MinMax is also simpler to understand, reproduce, and maintain.

For **MobileNetV2**, I would also deploy `pc-minmax`.

It reduces the model from **13.34 MB to 3.72 MB**, decreases median latency from **7.99 ms to 5.57 ms** (**1.43× speedup**), and loses only **0.23 percentage points** of top-1 accuracy.

`pc-entropy` gives essentially the same accuracy and a very small latency improvement, but the difference is not large enough to justify preferring the more complicated calibration method based on this experiment alone.

The per-tensor MobileNetV2 configuration would not be acceptable because its accuracy falls by **5.73 percentage points**.

Overall, **per-channel MinMax provides the best engineering trade-off in this study**: large reductions in model size, meaningful CPU speedups, minimal accuracy loss, and a relatively simple and reproducible quantization pipeline.

## Limitations

- CPU-only, one machine, batch size 1; INT8 speedups depend heavily on CPU instruction support
  (e.g. VNNI), so results do not transfer to GPUs or edge NPUs.
- Imagenette is an easy 10-class subset of ImageNet; absolute accuracy is not comparable to full ImageNet.
- Static PTQ only; quantization-aware training (QAT) and 4-bit / mixed-precision search are not covered.
- Single calibration seed; no confidence intervals.

## Possible extensions

- C++ inference of the same ONNX models with the ONNX Runtime C++ API.
- TensorRT INT8 on an NVIDIA GPU; OpenVINO on Intel CPUs.
- Quantize a Transformer vision model (e.g. a small ViT) and compare the failure modes.
