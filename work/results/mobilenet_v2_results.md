<!-- AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD | 16 logical CPUs | onnxruntime 1.23.2 | Windows-10-10.0.26200-SP0 -->

| model | size (MB) | top-1 (%) | Δ vs FP32 (pp) | agrees w/ FP32 (%) | latency ms, median (1 thread) | p95 ms | speedup | Q/DQ nodes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mobilenet_v2_fp32 | 13.34 | 78.8 | 0.0 | 100.0 | 7.99 | 9.38 | 1.0x | 0 |
| mobilenet_v2_int8_pc-entropy | 3.72 | 78.57 | -0.23 | 90.22 | 5.5 | 5.75 | 1.45x | 238 |
| mobilenet_v2_int8_pc-minmax-keep-first-last-fp32 | 7.37 | 78.5 | -0.3 | 90.19 | 5.41 | 5.9 | 1.48x | 228 |
| mobilenet_v2_int8_pc-minmax | 3.72 | 78.57 | -0.23 | 90.22 | 5.57 | 6.08 | 1.43x | 238 |
| mobilenet_v2_int8_pc-percentile | 3.72 | 78.04 | -0.76 | 91.59 | 5.71 | 6.78 | 1.4x | 238 |
| mobilenet_v2_int8_pt-minmax | 3.49 | 73.07 | -5.73 | 79.39 | 5.42 | 6.1 | 1.47x | 238 |

Images evaluated: 3925
