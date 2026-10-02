<!-- AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD | 16 logical CPUs | onnxruntime 1.23.2 | Windows-10-10.0.26200-SP0 -->

| model | size (MB) | top-1 (%) | Δ vs FP32 (pp) | agrees w/ FP32 (%) | latency ms, median (1 thread) | p95 ms | speedup | Q/DQ nodes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| resnet50_fp32 | 97.41 | 84.54 | 0.0 | 100.0 | 89.72 | 128.44 | 1.0x | 0 |
| resnet50_int8_pc-entropy | 24.87 | 84.59 | 0.05 | 96.94 | 50.87 | 55.21 | 1.76x | 256 |
| resnet50_int8_pc-minmax-keep-first-last-fp32 | 30.74 | 84.15 | -0.39 | 97.86 | 54.61 | 61.55 | 1.64x | 244 |
| resnet50_int8_pc-minmax | 24.87 | 84.59 | 0.05 | 96.94 | 50.41 | 63.26 | 1.78x | 256 |
| resnet50_int8_pc-percentile | 24.87 | 84.71 | 0.17 | 96.87 | 49.11 | 54.45 | 1.83x | 256 |
| resnet50_int8_pt-minmax | 24.53 | 84.87 | 0.33 | 93.15 | 49.2 | 54.92 | 1.82x | 256 |

Images evaluated: 3925
