"""Toy quantization in pure numpy: see WHY the tricks in the real experiment exist.

Run:  python scripts/00_toy_quantization.py
Needs only numpy. Read the printed output next to STUDY_GUIDE_zh-TW.md, part 2.
"""
import numpy as np


def quantize_affine(x, lo, hi, bits=8):
    """x -> integers in [0, 2^bits-1] using range [lo, hi]. Returns (q, scale, zero_point)."""
    qmax = 2**bits - 1
    scale = (hi - lo) / qmax
    zp = int(np.clip(round(-lo / scale), 0, qmax))
    q = np.clip(np.round(x / scale) + zp, 0, qmax)
    return q.astype(np.int32), scale, zp


def dequantize(q, scale, zp):
    return (q - zp) * scale


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


print("=== 1. One number by hand: range [-1.2, 3.4], 8 bits, x = 0.5 ===")
q, s, zp = quantize_affine(np.array([0.5]), -1.2, 3.4)
print(f"scale = (3.4 - (-1.2)) / 255 = {s:.6f}, zero_point = {zp}")
print(f"q = round(0.5/scale) + zp = {q[0]}   ->  back to float: {dequantize(q, s, zp)[0]:.5f}  "
      f"(rounding error {abs(dequantize(q, s, zp)[0] - 0.5):.5f})")

rng = np.random.default_rng(0)
x = rng.standard_normal(100_000).astype(np.float32)

print("\n=== 2. Fewer bits -> bigger error (Gaussian data, min/max range) ===")
for bits in (8, 6, 4):
    q, s, zp = quantize_affine(x, x.min(), x.max(), bits)
    print(f"{bits}-bit  RMSE = {rmse(x, dequantize(q, s, zp)):.5f}")

print("\n=== 3. ONE outlier ruins min/max calibration ===")
x_out = x.copy()
x_out[0] = 50.0  # a single extreme activation
q, s, zp = quantize_affine(x_out, x_out.min(), x_out.max())
err_minmax = rmse(x_out[1:], dequantize(q, s, zp)[1:])
lo, hi = np.percentile(x_out, [0.005, 99.995])  # clip the extreme tail instead
q, s2, zp2 = quantize_affine(np.clip(x_out, lo, hi), lo, hi)
err_pct = rmse(x_out[1:], dequantize(q, s2, zp2)[1:])
print(f"min/max range [{x_out.min():.1f}, {x_out.max():.1f}]  scale={s:.4f}  RMSE on normal values = {err_minmax:.5f}")
print(f"percentile range [{lo:.2f}, {hi:.2f}]  scale={s2:.4f}  RMSE on normal values = {err_pct:.5f}")
print("(percentile clips the outlier itself, but everyone else gets a much finer grid)")

print("\n=== 4. Per-tensor vs per-channel weights (channels have very different ranges) ===")
w = rng.standard_normal((64, 128)).astype(np.float32)
w *= np.logspace(-1, 1, 64, dtype=np.float32)[:, None]  # channel 0 ~0.1x ... channel 63 ~10x
q, s, zp = quantize_affine(w, w.min(), w.max())
err_tensor = rmse(w, dequantize(q, s, zp))
w_hat = np.empty_like(w)
for c in range(w.shape[0]):  # each output channel gets its own scale
    qc, sc, zc = quantize_affine(w[c], w[c].min(), w[c].max())
    w_hat[c] = dequantize(qc, sc, zc)
err_channel = rmse(w, w_hat)
small = slice(0, 8)  # the small-range channels are the ones per-tensor destroys
print(f"per-tensor  RMSE (all channels)        = {err_tensor:.5f}")
print(f"per-channel RMSE (all channels)        = {err_channel:.5f}")
print(f"per-tensor  RMSE (8 smallest channels) = {rmse(w[small], dequantize(*quantize_affine(w, w.min(), w.max()))[small]):.5f}")
print(f"per-channel RMSE (8 smallest channels) = {rmse(w[small], w_hat[small]):.5f}")
