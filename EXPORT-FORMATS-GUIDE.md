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
- `model_gb` - Model parameter memory (GB)
- `grad_gb` - Gradient memory (GB)
- `optim_gb` - Optimizer state memory (GB)
- `activation_gb` - Activation memory (GB)
- `buffer_gb` - Communication buffer memory (GB)
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
- `rel_speed` - Speed relative to a baseline. **The baseline is the FIRST platform
  configuration in sweep order, not the fastest or the slowest**, and it differs
  per variant, so `rel_speed` is not comparable across variants.
- `months_low` / `months_high` - 0.75x and 1.25x multipliers on `est_months`
- `bubble_pct` - PP bubble as % of total time
- `comm_overhead_pct` - Exposed communication as % of total time
- `efficiency_rating` - Star rating 0-3 derived from `rel_speed`, NOT from the step
  count (see "Efficiency rating" below)

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

### Efficiency rating (Phase 3 `efficiency_rating`)

This is a **relative SPEED** rating, not a step-count rating, and it is derived
from `rel_speed` alone:

- **0** (⭐⭐⭐): `rel_speed >= 2.0`
- **1** (⭐⭐): `rel_speed >= 1.5`
- **2** (⭐): `rel_speed >= 1.0`
- **3** (no stars): slower than the baseline

**Read it with care.** `rel_speed` is measured against the FIRST platform
configuration in sweep order, which is neither the fastest nor the slowest -- on a
default run it is mid-pack for all 22 variants, `rel_speed` spans 0.002 to 51.35,
and **78% of rows (3,968 of 5,084) land on rating 0**. So rating 0 means "at least
twice an arbitrary mid-pack anchor", not "optimal", and ratings are not comparable
between variants. Rank by `est_months` instead.

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

# Rank configurations by actual estimated time, not by efficiency_rating.
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
