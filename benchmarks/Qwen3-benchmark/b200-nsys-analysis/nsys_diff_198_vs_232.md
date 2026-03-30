# Diff: 30b 64×GPU TP1PP1VP1 vs 30b 64×GPU TP1PP1VP1

## Key Metrics

| Metric | A | B | Delta | Change |
|--------|---|---|-------|--------|
| Step time | 17.87s | 16.94s | -0.93s (-5.2%) | better |
| Throughput | 938933.17tok/s | 990366.70tok/s | +51433.53tok/s (+5.5%) | better |
| MFU | 27.39% | 28.89% | +1.50% (+5.5%) | better |
| Pipeline bubble | 31.04% | 23.80% | -7.23% (-23.3%) | better |
| Compute ratio | 58.10% | 38.53% | -19.57% (-33.7%) | worse |
| Comm ratio | 41.90% | 61.47% | +19.57% (+46.7%) | worse |

## Per-PP-Stage

### PP0
| Metric | A | B | Delta |
|--------|---|---|-------|
| Compute | 41.09s | 34.06s | -7.03s |
| Comm | 29.63s | 54.33s | +24.70s |
| PP SendRecv | 0.00s | 0.00s | 0.00s |
| Idle % | 31.04% | 23.80% | -7.23% |
| Eff. Compute | 59.06% | 46.61% | -12.45% |
