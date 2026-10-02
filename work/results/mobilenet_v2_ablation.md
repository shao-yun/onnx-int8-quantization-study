<!-- AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD | 16 logical CPUs | onnxruntime 1.23.2 | Windows-10-10.0.26200-SP0 -->

| model | size (MB) | top-1 (%) | Δ vs FP32 (pp) | agrees w/ FP32 (%) | latency ms, median (1 thread) | p95 ms | speedup | Q/DQ nodes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mobilenet_v2_fp32 | 13.34 | 78.8 | 0.0 | 100.0 | 7.97 | 8.43 | 1.0x | 0 |
| mobilenet_v2_int8_pt-minmax | 3.49 | 77.15 | -1.65 | 86.29 | 5.38 | 5.6 | 1.48x | 238 |
| mobilenet_v2_int8_pc-minmax | 3.72 | 68.76 | -10.04 | 76.31 | 5.41 | 5.59 | 1.47x | 238 |
| mobilenet_v2_int8_pc-percentile | 3.72 | 72.48 | -6.32 | 83.75 | 5.42 | 5.57 | 1.47x | 238 |
| mobilenet_v2_int8_pc-entropy | 3.72 | 68.76 | -10.04 | 76.31 | 5.42 | 5.6 | 1.47x | 238 |
| mobilenet_v2_int8_pc-minmax-keep-first-last-fp32 | 7.37 | 74.01 | -4.79 | 84.05 | 5.56 | 5.88 | 1.43x | 228 |

Images evaluated: 3925
