# Report Integration

This file describes the automatic HTML report generation that runs after every calculator execution, using the `aws-branded-report` Kiro Power.

## Detection

Before generating, check if the `aws-branded-report` power is available. Look for its POWER.md in the project's `.kiro/powers/` directory or installed powers:

```
Check if file exists: .kiro/powers/aws-branded-report/POWER.md
```

- If the file **exists**: proceed with report generation below.
- If the file **does not exist**: skip this entire workflow silently. Do not warn the user or suggest installing the power.

## Loading the report power

Load the `aws-branded-report` power's steering files as needed:
- `report-workflow.md` for the assembly steps
- `chart-generation.md` for the `generate_chart.py` helper functions
- `template-reference.md` for the HTML template and placeholder reference
- `css-classes.md` for table, callout, metric card CSS classes

The chart helper script (`generate_chart.py`) must be saved locally before generating charts. Follow the instructions in the report power's `chart-generation.md` to extract and save it.

## Data sources

The report draws from CSV files exported by the calculator in `/tmp/ai-model-calculator/`:

| File | Report usage |
|------|-------------|
| `phase1_memory_results.csv` | Memory breakdown table, OOM identification, metric cards |
| `phase1_memory_results.json` | Structured data for chart generation (memory bars) |
| `phase2_batch_results.csv` | Batch configuration table (top 5 by priority) |
| `phase3_training_results.csv` | Training time table, metric cards, bar chart data |
| `phase4_zero2_comm_results.csv` | ZeRO-2 communication overhead table |
| `phase4_1_zero1_comm_results.csv` | ZeRO-1 communication overhead table |
| `phase5_alltoall_comm_results.csv` | MoE all-to-all routing table (skip for dense-only runs) |

Read these CSV files to extract the data needed for each report section.

## Charts to generate (3-4)

Use the `generate_chart.py` helper from the `aws-branded-report` power:

```python
import sys
sys.path.insert(0, "/tmp")  # or wherever generate_chart.py was saved
from generate_chart import bar_chart, line_chart, to_base64
```

**Chart 1: GPU Memory Breakdown (bar chart)**
- Source: `phase1_memory_results.json`
- X-axis: Model variant names
- Y-axis: Memory in GB (grouped: model, gradient, optimizer, activation, buffer)
- One group per hardware platform (if comparing multiple)
- Title: "Per-GPU Memory Breakdown by Model Variant"

**Chart 2: Training Time Comparison (bar chart)**
- Source: `phase3_training_results.csv`
- X-axis: Platform names (e.g., "64 p5en", "128 p5en", "256 p5")
- Y-axis: Training time in months (use midpoint of low-high range)
- One bar group per model variant (if comparing multiple)
- Title: "Estimated Training Time by Platform"

**Chart 3: Communication Overhead vs DP (line chart)**
- Source: `phase4_zero2_comm_results.csv` and `phase4_1_zero1_comm_results.csv`
- X-axis: DP rank count (64, 128, 256, 512, 1024, ...)
- Y-axis: Communication time in ms
- Two lines: ZeRO-1 (all-gather) and ZeRO-2 (reduce-scatter)
- Title: "Communication Overhead vs Data Parallelism Degree"

**Chart 4 (MoE only): All-to-All Routing Time (bar chart)**
- Source: `phase5_alltoall_comm_results.csv`
- X-axis: Micro batch size (1, 2, 4, 8, 16)
- Y-axis: All-to-all time in ms
- Skip this chart entirely if all selected models are dense (expert_ffn=0)
- Title: "MoE All-to-All Communication vs Micro Batch Size"

After generating each chart PNG, convert to base64 with `to_base64()` for embedding in the HTML report.

## Report sections mapping

The HTML report follows this section structure:

| # | Section Title | Content |
|---|--------------|---------|
| 1 | Executive Summary | 3-4 metric cards: model name(s), total GPUs, best training time estimate, recommended ZeRO stage. 2-3 sentence overview. |
| 2 | Memory Analysis | Phase 1 data table (variant, model mem, grad, optim, activation, buffer, total, headroom, ZeRO, micro). Chart 1. OOM callout boxes. |
| 3 | Batch Configuration | Phase 2 top 5 configurations per hardware (priority 0-1 only). Table with micro, accum, tokens/batch, steps, assessment. |
| 4 | Training Time Estimates | Phase 3 table (platform, GPUs, ZeRO, time range, relative speed). Chart 2. Metric card for fastest config. |
| 5 | Communication Overhead | Phase 4 combined table (DP, ZeRO-1 all-gather ms, ZeRO-2 reduce-scatter ms). Chart 3. Warning callout if >5ms. |
| 6 | MoE Routing Analysis | Phase 5 table (micro, fwd+bwd volume, time, rating). Chart 4. **Skip for dense-only runs.** |
| 7 | Recommendations | Aggregated recommendations from all phases. Info callouts for positive, warning callouts for concerns. |

## Report metadata

Set these values when assembling the report:

- **Title:** `"LLM Training Infrastructure Analysis: {model_names}"`
- **Subtitle:** `"{date} | {hardware_platforms} | {total_gpus} GPUs"`
- **Badge:** `"Internal"`
- **Footer:** `"Amazon Web Services · LLM Training Analysis · Internal Use Only"`
- **Output path:** Same directory as the CSV exports, named `training-analysis-{model_slug}.html`

## Assembly workflow

1. Load the `aws-branded-report` power's steering files (`report-workflow.md`, `chart-generation.md`, `template-reference.md`, `css-classes.md`)
2. Save `generate_chart.py` to `/tmp/` if not already present
3. Read the CSV/JSON export files from `/tmp/ai-model-calculator/`
4. Generate charts 1-3 (and chart 4 if MoE) using `generate_chart.py`
5. Convert chart PNGs to base64
6. Read the HTML template from the `aws-branded-report` power's `template-reference.md`
7. Build each section's HTML content using the CSS classes from `css-classes.md`
8. Replace template placeholders: `{{TITLE}}`, `{{SUBTITLE}}`, `{{BADGE}}`, `{{TOC_ITEMS}}`, `{{SECTIONS}}`, `{{FOOTER_TEXT}}`
9. Write the final HTML file to `/tmp/ai-model-calculator/training-analysis-{model_slug}.html`
10. Verify no placeholders remain (read back and check)
11. Report the file path to the user
