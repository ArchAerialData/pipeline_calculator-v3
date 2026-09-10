# Performance and output comparison

| Fixture | Before s | After s | Change | Peak RSS before/after MiB | Output equal |
| --- | --- | --- | --- | --- | --- |
| sparse | 0.217 | 0.235 | +8.0% | 126.3 / 126.6 | True |
| dense | 1.960 | 1.990 | +1.5% | 158.3 / 158.1 | True |
| chain | 0.873 | 0.880 | +0.8% | 126.3 / 126.5 | True |
| short_paths | 0.120 | 0.128 | +7.0% | 126.3 / 126.6 | True |
| linked | 14.256 | 13.873 | -2.7% | 533.9 / 533.9 | True |
| curved | 0.833 | 0.836 | +0.5% | 126.3 / 126.5 | True |

Dense workload context/progress overhead vs the final disabled mode: **2.30%**. Five timed runs per workload after warm-up; profiler runs excluded. Subsecond changes are noisy. Output comparisons exclude only three additive corridor metadata fields and normalize input directory names.
