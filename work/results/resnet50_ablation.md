<!-- AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD | 16 logical CPUs | onnxruntime 1.23.2 | Windows-10-10.0.26200-SP0 -->

| model | size (MB) | top-1 (%) | Δ vs FP32 (pp) | agrees w/ FP32 (%) | latency ms, median (1 thread) | p95 ms | speedup | Q/DQ nodes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| resnet50_fp32 | 97.41 | 84.54 | 0.0 | 100.0 | 87.85 | 89.9 | 1.0x | 0 |
| resnet50_int8_pt-minmax | 24.53 | 82.14 | -2.4 | 92.0 | 46.52 | 48.83 | 1.89x | 256 |
| resnet50_int8_pc-minmax | 24.87 | 79.06 | -5.48 | 87.77 | 46.53 | 48.26 | 1.89x | 256 |
| resnet50_int8_pc-percentile | 24.87 | 79.59 | -4.95 | 88.08 | 46.52 | 48.81 | 1.89x | 256 |
| resnet50_int8_pc-entropy | 24.87 | 79.06 | -5.48 | 87.77 | 46.49 | 48.7 | 1.89x | 256 |
| resnet50_int8_pc-minmax-keep-first-last-fp32 | 30.74 | 84.28 | -0.26 | 98.45 | 51.85 | 55.26 | 1.69x | 244 |

Images evaluated: 3925
