# 30b Profile — 64×B200 SXM 192GB

**Config**: BF16 | 8 nodes × 8 B200 SXM 192GB | TP=1 PP=1 CP=1 VP=1 DP=8 | GBS=4096 MBS=2 Seq=4096

## Key Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Step time | 16.94s (±9.779s) | Stable across profiled steps |
| Throughput | 990367 tokens/sec | 15474 tokens/sec/GPU |
| MFU | 28.9% | Below typical 35-45% target (2250 TFLOPS peak BF16) |
| Pipeline bubble | 23.8% avg | Theoretical 0.0% |
| Compute : Communication | 39% : 61% | Communication-bound |

## Communication Breakdown by PP Stage

| PP Stage | Compute (s) | All Comm (s) | PP SendRecv (s) | Idle % | Exposed Comm (s) | Eff. Compute |
|----------|-------------|-------------|-----------------|--------|-------------------|--------------|
| PP0 | 34.1 | 54.3 | 0.0 | 23.8% | 39.1 | 46.6% |



## Compute-Communication Overlap

| Comm Type | PP0/PP1 Time (s) | PP0/PP1 Hidden | PP2/PP3 Time (s) | PP2/PP3 Hidden |
|-----------|-------------------|----------------|-------------------|----------------|
| TP | 42.5 | 34% | 0.0 | 0% |
| DP | 14.2 | 7% | 0.0 | 0% |
| **Total** | **54** | **28%** | **0** | **0%** |

## Root Cause

Communication overhead accounts for 61% of GPU kernel time, with pipeline-parallel SendRecv being the dominant contributor. The effective compute efficiency ranges from 46.6% to 46.6% across PP stages.


## Recommendations

1. **Consider VP=2** if pipeline bubble exceeds 5%
2. **Current PP=1 is already minimal**
3. **Enable `batch_p2p_comm=True`** — batches all P2P ops into one NCCL group call
4. **Enable `overlap_p2p_comm=True`** — overlaps PP SendRecv with compute on separate CUDA streams
5. **Verify EFA/network configuration** — confirm `FI_EFA_USE_DEVICE_RDMA=1` and placement groups

---

<div style="page-break-after: always;"></div>

## Appendix: Analysis Plots

![Compute-Communication Overlap Timeline](plots/07_overlap_timeline.png)

![PP Stage Comparison](plots/04_pp_stage_comparison.png)

![Overlap by Communication Type](plots/05_overlap_by_type.png)