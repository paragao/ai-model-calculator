# Export Formats Guide

The model calculator automatically exports analysis results to CSV and JSON formats for further analysis in Excel, Python, R, or other data analysis tools.

## Output Files Generated

When you run `python3 model_calculations.py`, the following files are created.
Sizes are from a full default run (22 variants x 32 hardware presets) and are
indicative, not fixed.

### Phase 1: Memory Analysis
- **phase1_memory_results.csv** (~110 KB, 704 rows) - Memory breakdown per variant/hardware
- **phase1_memory_results.json** (~770 KB) - Detailed memory analysis with nested data

### Phase 2: Batch Configuration
- **phase2_batch_results.csv** (~1.1 MB, 13,728 rows) - One row per micro x accum combination

### Phase 3: Training Time Estimation
- **phase3_training_results.csv** (~970 KB, 5,084 rows) - Training duration estimates

### Phase 4: Communication Overhead
- **phase4_zero2_comm_results.csv** (~76 KB, 899 rows) - ZeRO-2 reduce-scatter overhead
- **phase4_1_zero1_comm_results.csv** (~72 KB, 857 rows) - ZeRO-1 all-gather overhead
- **phase4_2_zero3_comm_results.csv** (~133 KB, 986 rows) - ZeRO-3: two all-gathers plus reduce-scatter

### Phase 5: MoE Communication
- **phase5_alltoall_comm_results.csv** (~218 KB, 3,520 rows) - All-to-all routing overhead

Phase 6 (PP SendRecv) is reported inline on the terminal and writes no CSV.

## CSV Format Details

Column names below are the ACTUAL header strings. Several are abbreviated in the
file (`gpus`, `zero`, `micro`, `accum`, `tok_batch`, `steps`, `est_months`), so
read them verbatim rather than guessing a longer form.

### Phase 1 Memory Results (phase1_memory_results.csv)

**Columns:**
- `variant` - Model variant name (e.g., "Qwen3-30B-A3B")
- `hardware` - Hardware preset name (e.g., "64 p5" -- 64 INSTANCES, 512 GPUs)
- `zero_strategy` - Best strategy that fits: "ZeRO-1", "ZeRO-2", "ZeRO-3", or "OOM"
- `micro_batch` - Largest viable micro batch (1, 2, 4, 8 or 16), or "N/A" when OOM
- `model_gb` - Model parameter memory (GB). Held at `PARAM_STORAGE_BYTES` (2, BF16)
  **regardless of `PRECISION`**: `--compute_dtype fp8` is a GEMM precision, so
  Transformer Engine keeps the weights resident in BF16 and adds FP8 cast copies on
  top. This column does not halve at FP8.
- `grad_gb` - Gradient memory (GB). Held at `GRAD_BYTES_PER_PARAM` (2, BF16)
  **regardless of `PRECISION`**: an FP8 recipe uses FP8 only for the GEMM inputs and
  reduces gradients in BF16 or higher, so this column does not halve at FP8.
- `optim_gb` - Optimizer state memory (GB)
- `activation_gb` - Activation memory (GB)
- `buffer_gb` - Communication buffer memory (GB)
- `logits_gb` - Output-logits memory (GB): the `(micro, seq_len, vocab)` tensor plus
  its cross-entropy gradient, at `LOGITS_BYTES_PER_ELEMENT` (8) bytes per element.
  **Not** part of `activation_gb`, which is per-layer in `d` and never touches
  `vocab`. Under the default fused-CE assumption only `LOGITS_CHUNK_TOKENS` (1024)
  rows are live at once, so above that this column is **flat in `micro_batch`**;
  with `LOGITS_FUSED_CE = False` it grows linearly and becomes the largest term.
  See "Output logits" below.
- `zero3_transient_gb` - ZeRO-3 transient all-gather buffer (GB). **0.0 for ZeRO-1
  and ZeRO-2**, which do not gather parameters. DP-invariant, and for 18 of 22
  variants the largest gathered unit is the embedding table.
- `total_gb` - Total memory required (GB), including a 1.24x fragmentation factor
- `gpu_capacity_gb` - GPU memory capacity (GB)
- `headroom_gb` - Remaining memory (GB)
- `utilization_pct` - Memory utilization percentage
- `memory_efficiency` - Memory efficiency score
- `status` - "OK" or "OOM"

### Phase 2 Batch Results (phase2_batch_results.csv)

One row per (variant, hardware, micro, accum) combination that survives the
derived batch cap. **Three flag columns carry information the `priority` does not.**

**Columns:**
- `variant`, `hardware` - as Phase 1
- `seq_len` - Sequence length for this variant
- `micro_batch`, `grad_accum` - The swept combination
- `tokens_per_batch_M` - Global batch in millions of tokens
- `training_steps` - `TOTAL_TOKENS / tokens_per_batch`
- `assessment` - `reasonable`, `outlier (fewest steps)` or `outlier (most steps)`
- `priority` - **1 (reasonable) or 3 (outlier) only.** RELATIVE to the other
  configurations in the same (variant, hardware) pair: the smallest and largest
  step counts are outliers and the rest are reasonable. There is no absolute step
  band and no priority 0 or 2.
- `fits_in_memory` - **Tri-state**: `True`, `False`, or **blank** meaning "not
  assessed" because Phase 1 found no viable micro batch for that pair. Blank must
  NOT be rendered as "does not fit". Phase 2 has no memory model of its own.
- `exceeds_batch_cap` - Against the cap DERIVED per configuration, not the fixed
  `MAX_TOKENS_PER_BATCH`. Uniformly `False` in the default export because the
  pipeline filters capped rows out before writing.
- `steps_absurd` - The **absolute** axis: `True` when `training_steps` exceeds
  `TOTAL_TOKENS / TOKENS_PER_BATCH`, i.e. when the batch is smaller than any
  published frontier run used at any point. **Independent of `priority`** -- on a
  default run 7,260 rows are `priority` 1 and `steps_absurd` True at once, because
  "reasonable" only compares within its own pair. Filtering on `priority` alone
  keeps absurd rows.

### Phase 3 Training Time Results (phase3_training_results.csv)

**Columns:**
- `variant` - Model variant name
- `platform` - Hardware preset plus config, e.g. `64 p5 (m1/a1, 0.3M)`
- `gpus` - Number of GPUs
- `zero` - ZeRO stage as an integer: 1, 2 or 3
- `micro`, `accum` - Micro batch and gradient accumulation
- `seq_len` - Sequence length
- `gbs` - Global batch in SEQUENCES
- `tok_batch` - Global batch in TOKENS (`gbs * seq_len`)
- `steps` - Training steps for `TOTAL_TOKENS` (15e12 by default)
- `est_months` - Overhead-inclusive estimate (compute + PP bubble + exposed comm)
- `rel_speed` - How many times faster than the **SLOWEST** configuration for the
  same variant. Always `>= 1.0`, and exactly `1.0` for the slowest config itself.
  The anchor is per-variant, so the value is a within-variant speed-up and is not a
  cross-variant figure: the fastest config's `rel_speed` equals that variant's full
  spread, which ranges from 32.5x to 20,642x depending on the variant.
- `months_low` / `months_high` - 0.75x and 1.25x multipliers on `est_months`
- `bubble_pct` - PP bubble as % of total time
- `comm_overhead_pct` - Exposed communication as % of total time
- `efficiency_rating` - Star rating 0-3 from the config's **position** within its
  variant's speed range, NOT a threshold on `rel_speed` and NOT a step-count
  rating (see "Efficiency rating" below). Empty when unset.

### Phase 4 Communication (phase4_*.csv)

All three share `hardware`, `variant`, `model_mem_gb`, `viable_micros`, `dp` and
`overhead_rating`, then differ in the volume/time columns:
- **ZeRO-1** (`phase4_1_zero1_comm_results.csv`): `ag_volume_mb`, `ag_time_ms`
- **ZeRO-2** (`phase4_zero2_comm_results.csv`): `rs_volume_mb`, `rs_time_ms`
- **ZeRO-3** (`phase4_2_zero3_comm_results.csv`): `fwd_ag_volume_mb`,
  `bwd_ag_volume_mb`, `rs_volume_mb`, `total_volume_mb`, `total_time_ms`,
  `baseline_ratio` (1.5x the ZeRO-1/2 baseline: 3*Psi per step vs 2*Psi)

### Phase 5 MoE All-to-All (phase5_alltoall_comm_results.csv)

**Columns:** `hardware`, `variant`, `d`, `micro`, `fwd_bwd_gb`, `a2a_time_ms`,
`performance_rating`. Dense variants are absent.

## JSON Format Details

The JSON files contain the same data but in a hierarchical format suitable for programmatic access:

```json
{
  "hardware": ["64 p5", "128 p5", ...],
  "variants": {
    "Qwen3-30B-A3B": [
      {
        "viable_micros_z1": [1, 2, 4, 8],
        "viable_micros_z2": [1, 2, 4, 8],
        "viable_micros_z3": [1, 2, 4, 8, 16],
        "best_zero": 1,
        "best_micro": 8,
        "memory_breakdown": {
          "total": 44.22,
          "model": 10.28,
          "grad": 10.28,
          "optim": 0.02,
          "activation": 17.39,
          "buffer": 6.15,
          "zero3_transient": 0.0,
          "zero3_transient_detail": {}
        },
        "efficiency_metrics": {
          "utilization_pct": 31.36,
          "memory_efficiency": 0.44
        },
        "oom_diagnostics": {},
        "model_mem_gb": 10.28,
        "hardware": "64 p5"
      }
    ]
  }
}
```

Note `viable_micros_z3`, `zero3_transient`, `zero3_transient_detail`,
`oom_diagnostics`, `model_mem_gb` and `hardware`: all present per entry. The
numeric values above are illustrative -- run the calculator for current figures.

## Using the Exported Data

### Excel / Google Sheets
1. Open CSV files directly in Excel/Sheets
2. Use pivot tables to analyze by variant or hardware
3. Create charts comparing memory usage, training time, etc.

### Python (Pandas)
```python
import pandas as pd

# Load memory analysis
df = pd.read_csv('phase1_memory_results.csv')

# Filter for Qwen3 models on p5 presets.
# NOTE: `hardware` holds preset NAMES like "64 p5", not accelerator names, so a
# .str.contains('H200') filter matches NOTHING. Filter on the preset suffix.
qwen3_p5 = df[(df['variant'].str.contains('Qwen3')) &
              (df['hardware'].str.endswith('p5'))]

# Analyze memory usage
print(qwen3_p5[['variant', 'total_gb', 'headroom_gb', 'utilization_pct']])

# Load training time estimates
train_df = pd.read_csv('phase3_training_results.csv')
print(train_df[train_df['variant'] == 'Qwen3-30B-A3B'])
```

### Python (JSON)
```python
import json

# Load detailed memory analysis
with open('phase1_memory_results.json', 'r') as f:
    data = json.load(f)

# Access Qwen3-30B-A3B results
qwen3_30b = data['variants']['Qwen3-30B-A3B']
print(f"Best micro batch: {qwen3_30b[0]['best_micro']}")
print(f"Memory breakdown: {qwen3_30b[0]['memory_breakdown']}")
```

### R
```r
library(tidyverse)

# Load and analyze memory results
memory_df <- read_csv('phase1_memory_results.csv')

# Plot memory usage by variant
memory_df %>%
  filter(str_detect(variant, "Qwen3")) %>%
  ggplot(aes(x = variant, y = total_gb, fill = hardware)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "Qwen3 Models: Memory Usage by Hardware",
       x = "Model Variant", y = "Total Memory (GB)")
```

## Key Metrics Explained

### Memory Efficiency
- **0.0 - 0.3**: Poor utilization (underutilized GPU)
- **0.3 - 0.6**: Good utilization
- **0.6 - 0.8**: Optimal utilization
- **0.8 - 1.0**: Very high utilization (less headroom for spikes)

### Output logits (Phase 1 `logits_gb`)

The LM head produces a `(micro, seq_len, vocab)` tensor and the cross-entropy
backward needs its gradient. `vocab` runs 128,256 to 248,320 in this catalog, so the
term is large, and it is **separate from `activation_gb`**, which is per-layer in `d`.

`LOGITS_BYTES_PER_ELEMENT = 8` covers a bf16 logits tensor plus the fp32 softmax and
its gradient.

**The fused-CE assumption is the dominant knob, and it is an assumption.** A fused or
chunked cross-entropy (Liger's `fused_linear_cross_entropy`, a compiled chunked loss)
walks the flattened `micro * seq_len` rows in chunks and keeps only one chunk live, so
above the chunk size the term stops scaling with `micro` altogether:

| `micro` | fused, 1024-token chunk | unfused |
|---|---|---|
| 1 | 1.05 GB | 4.20 GB |
| 4 | 1.05 GB | 16.81 GB |
| 8 | 1.05 GB | 33.62 GB |
| 16 | 1.05 GB | 67.24 GB |

Figures for `Llama-3.1-8B` at `seq_len` 4096, `vocab` 128,256.

Measured over all 10,560 variant × hardware × ZeRO × micro combinations, **3,880
configurations fit under the fused assumption against 3,252 unfused** — so the
assumption is worth 628 configurations and should not be treated as incidental. Set
`LOGITS_FUSED_CE = False` for a plain `nn.CrossEntropyLoss` over a materialised
logits tensor, and set `LOGITS_CHUNK_TOKENS` to your implementation's chunk size
rather than trusting the 1024 default, which is a placeholder and not a measured
property of any model.

The LM head **weight** gradient is not double-counted: the head's parameters are in
`model_gb` already, so `grad_gb` covers them.

### Precision (`PRECISION` = `BF16` or `FP8`) does not change any memory column

Every Phase 1 memory column is **byte-identical** at both settings. A full sweep of all
704 variant x hardware pairs produces the same numbers either way, and a test pins it.

This is deliberate and measured, not an oversight. `--compute_dtype fp8` selects a GEMM
precision; it is not a storage dtype. Transformer Engine keeps parameters resident in
BF16 and adds cached FP8 cast copies on top, gradients reduce in BF16, optimizer state
is fp32, and activations are BF16. On the one controlled comparison available -- Llama
70B LoRA finetune, H200 32 GPUs, TP=4 PP=2 VP=20 MBS=1, identical cluster, container
and parallelism with precision the only difference -- FP8 used **more** memory:

| | allocated | reserved |
|---|---|---|
| BF16 | 37.8 GB | 38.5 GB |
| FP8 | 41.1 GB | 41.3 GB |
| | **+8.73%** | +7.27% |

**Read the FP8 numbers as a lower bound.** Predicting equal memory is ~8% low against
that measurement, because the FP8 cast copies and scaling metadata are not modelled.
Before this was corrected the model predicted FP8 *cheaper*, which was 17% low and
wrong in sign. `PRECISION` still changes Phase 3 training time, via the throughput peak.

### Context parallelism (`CP`) divides the sequence-dependent memory

`CP` shards the sequence, so activation memory, the a2a/dispatch/PP buffers and the
output logits all scale as `1/CP` per GPU. Raising `CP` also shrinks
`DP = gpus / (TP * PP * CP)`, which raises the per-GPU optimizer and gradient share, so
the net effect on `total_gb` depends on which dominates.

Earlier releases applied only the `DP` half, so raising `CP` *increased* reported memory
-- the opposite of what context parallelism is for. Against a measured run (Llama 70B
BF16 pretrain, H200 64 GPUs, TP=2 PP=4 CP=2 VP=5, seq 8192, 108.08 GB reserved) the
ZeRO-1 estimate moved from 161.37 GB (1.49x) to 126.75 GB (1.17x), and ZeRO-2 -- the
closer analogue of a NeMo distributed optimizer -- to 108.88 GB, i.e. 1.01x measured.

### Efficiency rating (Phase 3 `efficiency_rating`)

This is a **relative SPEED** rating, not a step-count rating. It is the
configuration's log-scaled **position** within its own variant's
`[slowest, fastest]` range -- `0.0` at the slowest config, `1.0` at the fastest:

- **0** (⭐⭐⭐): position `>= 0.80`
- **1** (⭐⭐): position `>= 0.60`
- **2** (⭐): position `>= 0.40`
- **3** (no stars): position `< 0.40`

Empty means the config was never classified, which is distinct from rating 3.

**Why a position and not a threshold on `rel_speed`.** Per-variant spread
(slowest/fastest) runs from 32.5x for `Llama-3.1-405B` to 20,642x for
`Llama-3.2-3B` -- a 635x difference in the scale being measured. A fixed multiplier
would therefore mean something different for every variant: against a slowest
anchor, `rel_speed >= 2.0` is the top few percent of a 32x spread but the bottom
few percent of a 20,642x one. Positions are log-scaled because `est_months` values
within a variant are spread geometrically (GPU counts double), so a linear position
would put nearly every config against the fast end.

Measured over all 5,084 rows the bands are 26.6% / 36.1% / 21.0% / 16.3%, so every
band carries information. An earlier scheme anchored `rel_speed` on the first
platform in sweep order -- mid-pack for all 22 variants -- and put **78% of rows on
rating 0** simply by exceeding 2x an arbitrary entry.

`est_months` remains the figure to rank on when you want absolute time; the rating
tells you where a config sits among the alternatives for that variant.

### Step assessment (Phase 2 `priority` / `assessment`)

**There is no absolute step band.** Earlier versions graded step counts against
fixed ranges (300K-800K "optimal" and so on); those were removed because the
optimal arm was unreachable and the remaining bands condemned 80.8% of
configurations. See the Phase 2 column notes above: `priority` is 1 or 3, relative
to the pair, and the absolute check is the separate `steps_absurd` column.

### Communication Overhead Ratings
- **Excellent**: <2ms (ZeRO-2), <3ms (ZeRO-1), <1ms (all-to-all)
- **Good**: 2-5ms, 3-8ms, 1-3ms respectively
- **Moderate**: 5-10ms, 8-15ms, 3-5ms respectively
- **High**: >10ms, >15ms, >5ms respectively

## Getting current per-model figures

An earlier version of this guide embedded a per-variant results summary. It is
removed rather than updated, because it went stale in three ways at once: it quoted
`Qwen3-0.6B`, which is **not in the catalog**; it stated training times "for 30T
tokens" when `TOTAL_TOKENS` is **15e12**; and it used hardware labels like
`4,096 H200`, whereas preset names are of the form `64 p5` (64 **instances**, 512
GPUs -- the number is instances, not GPUs).

For current numbers, run the calculator and read the CSVs. To filter by accelerator
rather than preset name, match the preset suffix:

```python
import pandas as pd

df = pd.read_csv('phase1_memory_results.csv')

# Hardware is a preset NAME like "64 p5", so filter on the suffix, not "H200".
p5 = df[df['hardware'].str.endswith('p5')]
print(p5[['variant', 'hardware', 'zero_strategy', 'total_gb', 'status']])

# Rank by absolute time. efficiency_rating is a position WITHIN each variant's own
# range, so it ranks alternatives for one variant and is not a cross-variant sort key.
train = pd.read_csv('phase3_training_results.csv')
best = train.sort_values('est_months').groupby('variant').first()
print(best[['platform', 'zero', 'micro', 'accum', 'est_months']])
```

## Tips for Analysis

1. **Compare hardware**: Load results across presets (`64 p5` vs `64 p6-b200` etc.) to see memory differences
2. **Batch size impact**: Look at how micro batch size affects memory and training time
3. **ZeRO strategy trade-offs**: ZeRO-2 saves memory but increases communication overhead
4. **MoE vs Dense**: Compare Qwen3-30B-A3B (MoE) vs Qwen3-32B (dense) - similar params but different memory profiles
5. **Communication bottlenecks**: Check Phase 4/5 results for high-overhead configurations

## Visualizations

You can create various visualizations from the exported data:

- **Memory usage bar charts**: Compare variants across hardware
- **Training time comparison**: Bar/line charts showing time vs GPU count
- **Memory breakdown stacked bars**: Show model/grad/optim/activation/buffer proportions
- **Communication overhead heatmaps**: Visualize overhead by variant and hardware
- **Efficiency scatter plots**: Plot utilization vs memory efficiency

## Next Steps

1. **Load CSV files** in your preferred analysis tool
2. **Filter for specific models** (e.g., all Qwen3 models)
3. **Create visualizations** to identify trends
4. **Export findings** for presentations or reports
5. **Iterate on configurations** based on insights

---

The raw terminal output is not written to a file. Redirect it if you want one:
`python3 model_calculations.py > analysis.txt 2>&1` (ANSI colour codes are
preserved, so use `less -R` to read it).

For the full analysis with all variants, review the individual CSV files.
