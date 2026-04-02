# Export Formats Guide

The model calculator automatically exports analysis results to CSV and JSON formats for further analysis in Excel, Python, R, or other data analysis tools.

## Output Files Generated

When you run `python3 model_calculations.py`, the following files are automatically created:

### Phase 1: Memory Analysis
- **phase1_memory_results.csv** (8.6 KB) - Memory breakdown per variant/hardware
- **phase1_memory_results.json** (40 KB) - Detailed memory analysis with nested data

### Phase 2: Batch Configuration
- **phase2_batch_results.csv** (2.7 KB) - Optimal batch size configurations

### Phase 3: Training Time Estimation
- **phase3_training_results.csv** (12 KB) - Training duration estimates

### Phase 4: Communication Overhead
- **phase4_zero2_comm_results.csv** (15 KB) - ZeRO-2 reduce-scatter overhead
- **phase4_1_zero1_comm_results.csv** (11 KB) - ZeRO-1 all-gather overhead

### Phase 5: MoE Communication
- **phase5_alltoall_comm_results.csv** (12 KB) - All-to-all routing overhead

## CSV Format Details

### Phase 1 Memory Results (phase1_memory_results.csv)

**Columns:**
- `variant` - Model variant name (e.g., "Qwen3-30B-A3B")
- `hardware` - Hardware configuration (e.g., "4,096 H200")
- `zero_strategy` - Optimization strategy (ZeRO-1 or ZeRO-2)
- `micro_batch` - Micro batch size (1, 2, 4, or 8)
- `model_gb` - Model parameter memory (GB)
- `grad_gb` - Gradient memory (GB)
- `optim_gb` - Optimizer state memory (GB)
- `activation_gb` - Activation memory (GB)
- `buffer_gb` - Communication buffer memory (GB)
- `total_gb` - Total memory required (GB)
- `gpu_capacity_gb` - GPU memory capacity (GB)
- `headroom_gb` - Remaining memory (GB)
- `utilization_pct` - Memory utilization percentage
- `memory_efficiency` - Memory efficiency score
- `status` - "OK" or "OOM"

**Example rows (Qwen3 models):**
```csv
variant,hardware,zero_strategy,micro_batch,model_gb,total_gb,headroom_gb,utilization_pct
Qwen3-0.6B,"4,096 H200",ZeRO-1,8,1.71,13.57,127.43,9.62
Qwen3-30B-A3B,"4,096 H200",ZeRO-1,8,10.28,44.22,96.78,31.36
Qwen3-235B-A22B,"4,096 H200",ZeRO-2,4,68.20,108.52,32.48,76.98
```

### Phase 3 Training Time Results (phase3_training_results.csv)

**Columns:**
- `variant` - Model variant name
- `platform` - Hardware + batch configuration
- `num_gpus` - Number of GPUs
- `zero_strategy` - ZeRO strategy
- `micro_batch` - Micro batch size
- `grad_accum` - Gradient accumulation steps
- `tokens_per_batch` - Total tokens per batch
- `total_steps` - Training steps for 30T tokens
- `training_time_months` - Estimated training time (months)
- `relative_speed` - Speed relative to fastest config
- `time_low_months` - Optimistic estimate (75% multiplier)
- `time_high_months` - Pessimistic estimate (125% multiplier)
- `assessment` - Configuration quality (0-3, lower is better)

**Example rows:**
```csv
variant,platform,num_gpus,tokens_per_batch,total_steps,training_time_months
Qwen3-0.6B,"4,096 H200 (67.1M)",4096,67100000.0,447093,0.027
Qwen3-30B-A3B,"4,096 H200 (67.1M)",4096,67100000.0,447093,1.302
```

## JSON Format Details

The JSON files contain the same data but in a hierarchical format suitable for programmatic access:

```json
{
  "hardware": ["4,096 H200", "4,096 H100", ...],
  "variants": {
    "Qwen3-30B-A3B": [
      {
        "viable_micros_z1": [1, 2, 4, 8],
        "viable_micros_z2": [1, 2, 4, 8],
        "best_zero": 1,
        "best_micro": 8,
        "memory_breakdown": {
          "total": 44.22,
          "model": 10.28,
          "grad": 10.28,
          "optim": 0.02,
          "activation": 17.39,
          "buffer": 6.15
        },
        "efficiency_metrics": {
          "utilization_pct": 31.36,
          "memory_efficiency": 0.44
        }
      }
    ]
  }
}
```

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

# Filter for Qwen3 models on H200
qwen3_h200 = df[(df['variant'].str.contains('Qwen3')) & 
                (df['hardware'].str.contains('H200'))]

# Analyze memory usage
print(qwen3_h200[['variant', 'total_gb', 'headroom_gb', 'utilization_pct']])

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

### Training Time Assessment
- **0**: Optimal (300K-800K steps)
- **1**: Good (200K-300K or 800K-1.2M steps)
- **2**: Borderline (outside optimal ranges)
- **3**: Poor (batch too small/large)

### Communication Overhead Ratings
- **Excellent**: <2ms (ZeRO-2), <3ms (ZeRO-1), <1ms (all-to-all)
- **Good**: 2-5ms, 3-8ms, 1-3ms respectively
- **Moderate**: 5-10ms, 8-15ms, 3-5ms respectively
- **High**: >10ms, >15ms, >5ms respectively

## Qwen3 Model Results Summary

Based on the generated CSV files:

### Qwen3-0.6B
- **Memory**: 13.6 GB on H200, fits easily with ZeRO-1, micro=8
- **Training time (30T tokens)**: ~0.027 months (0.8 days) on 4,096 H200

### Qwen3-8B
- **Memory**: 70.0 GB on H200, fits with ZeRO-1, micro=8
- **Training time**: ~0.44 months (13 days) on 4,096 H200

### Qwen3-30B-A3B (MoE)
- **Memory**: 44.2 GB on H200, fits with ZeRO-1, micro=8
- **Training time**: ~1.3 months on 4,096 H200
- **Active params**: 2.9B (only 8 of 128 experts active)
- **MoE overhead**: 0.67-5.37ms all-to-all communication

### Qwen3-32B (Dense)
- **Memory**: 98.9 GB on H200, requires ZeRO-2, micro=4
- **OOM on H100**: Does not fit on 80GB GPUs

### Qwen3-235B-A22B (MoE)
- **Memory**: 108.5 GB on H200, requires ZeRO-2, micro=4
- **Training time**: ~9.5 months on 4,096 H200
- **Active params**: 19.04B (8 of 128 experts)
- **OOM on H100**: Does not fit on 80GB GPUs

## Tips for Analysis

1. **Compare hardware**: Load results for both H200 and H100 to see memory differences
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

For the raw terminal output with colors, see: `qwen3_full_analysis.txt`

For the full analysis with all variants, review the individual CSV files.
