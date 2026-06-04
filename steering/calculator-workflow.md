# Calculator Workflow

## How to run an analysis

Follow these steps in order for every analysis run.

### Step 0: Determine use case

Ask the user which mode they want:

**Mode A — "How many instances do I need to finish in time X?"**
The user provides:
- Model to train
- Dataset size (in tokens)
- Target training time (e.g., "2 months", "30 days", "1 week")

The calculator determines the minimum number of instances/GPUs needed to meet the time target.

**Mode B — "How long will training take?"**
The user provides:
- Model to train
- Dataset size (in tokens)
- Number of instances (and instance type)

The calculator estimates the training duration.

**Mode C — "What's the minimum to fit this model?"**
The user provides:
- Model to train (or serve/fine-tune)
- Instance type

The calculator determines the minimum number of instances required just to hold the model in GPU memory (before any training begins). This is useful for capacity planning and understanding the memory floor.

In Modes A and B, the user must provide the **dataset size in number of tokens**. Mode C does not require a dataset size — it's purely a memory-fit calculation.

After determining the mode, proceed to Step 1.

### Step 1: Select model architecture

Ask the user which model(s) to analyze. Present the pre-defined list grouped by family:

**Dense Models:** Llama-3.1-8B, Llama-3.1-70B, Llama-3.1-405B, Llama-3.2-3B, Llama-3.3-70B, Qwen2.5-7B, Qwen2.5-32B, Qwen2.5-72B, Qwen3-8B, Qwen3-32B, Qwen3.5-27B
**MoE Models:** Qwen3-30B-A3B, Qwen3-235B-A22B, Qwen3.5-35B-A3B, Qwen3.5-122B-A10B, Qwen3.5-397B-A17B, Llama-4-Scout-17B-16E, Llama-4-Maverick-17B-128E, DeepSeek-V3, DeepSeek-V3.1, DeepSeek-V3.2, DeepSeek-V4
**Custom:** Enter architecture details manually

The user can select multiple models for comparison. If "Custom", gather the 10 required parameters from `model-catalog.md`, then compute derived fields.

### Step 2: Select hardware platform

Ask the user which hardware platform(s) to analyze:

**Training-class (P-family):**
- **p5.48xlarge (H100 80GB)** -- node count: 64, 128, 256, 512, or custom
- **p5en.48xlarge (H200 141GB)** -- node count: 64, 128, 256, 512, or custom
- **p6-b200.48xlarge (B200 180GB)** -- node count: 64, 128, 256, 512, or custom
- **p6-b300.48xlarge (B300 Ultra 268GB)** -- node count: 64, 128, 256, 512, or custom
- **p6e-gb200.36xlarge (GB200 NVL 185GB)** -- UltraServer configs: 36, 72, 144, 288 GPUs

**Inference/fine-tuning (G-family):**
- **g5.48xlarge (A10G 24GB)** -- node count: 1, 4, 8, 16, or custom
- **g6.48xlarge (L4 24GB)** -- node count: 1, 4, 8, 16, or custom
- **g6e.48xlarge (L40S 48GB)** -- node count: 1, 4, 8, 16, or custom

**Custom** -- gather the 7 hardware parameters from `hardware-catalog.md`

The user can select multiple platforms for comparison.

**For Mode A (time-target):** If the user hasn't specified a platform, ask which instance type they prefer. Then the calculator will sweep multiple node counts to find the minimum that meets the time target.

**For Mode B (fixed-nodes):** The user provides both the instance type AND the node count.

### Step 3: Configure training parameters

Show the current defaults and ask if the user wants to change anything. Apply smart defaults based on model selection (see `training-config.md` for the full table):

- For **Llama 3.1/3.2/3.3 dense**: set `EP=1`, `N_EXPERTS=0`, `TOPK=0`, `VOCAB=128256`
- For **Llama 4 MoE**: set `VOCAB=202048`, `EP=8` (Scout: `N_EXPERTS=16`, `TOPK=1`; Maverick: `N_EXPERTS=128`, `TOPK=1`)
- For **Qwen2.5 dense**: set `EP=1`, `N_EXPERTS=0`, `TOPK=0`, `VOCAB=152064`
- For **Qwen3 dense**: set `EP=1`, `N_EXPERTS=0`, `TOPK=0`, `VOCAB=151936`
- For **Qwen3 MoE**: set `VOCAB=151936`, `N_EXPERTS=128`, `TOPK=8`, `EP=8`
- For **Qwen3.5 dense**: set `EP=1`, `N_EXPERTS=0`, `TOPK=0`, `VOCAB=248320`
- For **Qwen3.5 MoE**: set `VOCAB=248320`, `EP=8` (35B: `N_EXPERTS=256`, `TOPK=8`; 122B: `N_EXPERTS=256`, `TOPK=8`; 397B: `N_EXPERTS=512`, `TOPK=10`)
- For **DeepSeek-V3/V3.1/V3.2**: set `N_EXPERTS=256`, `TOPK=8`, `VOCAB=129280`, `EP=8`
- For **DeepSeek-V4**: set `N_EXPERTS=384`, `TOPK=6`, `VOCAB=129280`, `EP=8`

**Critical: Set TOTAL_TOKENS from user input.** The user provides the dataset size in tokens. Set `TOTAL_TOKENS` to this value (e.g., `30e9` for 30B tokens).

**For Mode A (time-target):** Also record the target time. After running the calculator, compare Phase 3 results against the target to identify the minimum node count.

**For Mode B (fixed-nodes):** Just run normally and report the estimated time.

If the user doesn't want to change anything else, proceed with defaults.

### Step 4: Modify configuration files

Before running the calculator, update the configuration files in `/tmp/ai-model-calculator/configuration/`:

1. **`variants_config.py`** -- Set the `VARIANTS` list to include only the selected model(s). Use templates from `model-catalog.md`.
2. **`hardware_config.py`** -- Set the `HARDWARE` list to include only the selected platform(s). Use templates from `hardware-catalog.md`.
3. **`project_config.py`** -- Update VOCAB, SEQ_LEN, N_EXPERTS, TOPK, PP, TP, CP, EP, TOTAL_TOKENS, PRECISION, MFU based on user selections and smart defaults. Recompute `EXPERTS_PER_GPU`.
4. **`advanced_config.py`** -- Update `TOKENS_PER_BATCH` if the user specified a target batch size.

### Step 5: Run the calculator

Execute from the cloned repo directory:

```bash
cd /tmp/ai-model-calculator && python3 model_calculations.py
```

The script runs all 5 phases sequentially and produces:
The script runs all 6 phases sequentially and produces:
- Console output with colored tables and recommendations
- CSV/JSON export files in the working directory:
  - `phase1_memory_results.csv` / `phase1_memory_results.json`
  - `phase2_batch_results.csv`
  - `phase3_training_results.csv`
  - `phase4_zero2_comm_results.csv`
  - `phase4_1_zero1_comm_results.csv`
  - `phase5_alltoall_comm_results.csv`
  - Phase 6: PP SendRecv analysis (inline output)

If the script fails with a validation error, check that `PP` divides `layers` evenly and `EP` divides `N_EXPERTS` evenly for all selected variants.

### Step 6: Present results

Present a structured summary to the user organized by mode and phase:

**For Mode A (time-target), lead with the answer:**
- "To complete training in ≤ X time, you need **N nodes (M GPUs)** of instance-type."
- Show a comparison table of all node counts vs estimated time
- Then present the detailed phase results for the recommended configuration

**For Mode B (fixed-nodes), lead with the answer:**
- "With N nodes (M GPUs), training will take approximately **X days (Y-Z range)**."
- Then present the detailed phase results

**Detailed phase results (both modes):**

**Phase 1 -- Memory Analysis:**
- For each hardware x variant: model memory, gradient, optimizer, activation, buffer, total, headroom, utilization %, best ZeRO stage, best micro batch size
- Flag any OOM cases

**Phase 2 -- Batch Configuration:**
- Top 3-5 batch configurations per hardware ranked by priority
- Each shows: micro batch size, gradient accumulation, tokens/batch, training steps, assessment

**Phase 3 -- Training Time:**
- For each variant x hardware: estimated time (low-high range), relative speed
- Compare across platforms if multiple selected

**Phase 4 -- ZeRO Communication Overhead:**
- ZeRO-2: reduce-scatter volume and time at various DP values
- ZeRO-1: all-gather volume and time at various DP values

**Phase 5 -- All-to-All Communication (MoE only):**
- Forward+backward all-to-all volume and time for each micro batch size

**Phase 6 -- PP SendRecv Communication:**
- PP communication type: intra-node (NVLink) vs inter-node (EFA)
- Time per send, pipeline bubble %, exposed overhead
- VP impact on send count (interleaved vs standard schedule)

After presenting the summary:
1. Re-run with different parameters (if user requests)
2. Export detailed results (point to CSV/JSON files)
3. Dive deeper into any specific phase (if user requests)

Then, **automatically proceed to generate the HTML report** if the `aws-branded-report` power is available. See `report-integration.md` for the full workflow.

## Calculation phases explained

### Phase 1: Memory Analysis

Computes per-GPU memory breakdown for every (variant, hardware, ZeRO stage, micro batch size) combination:
- **Model parameters**: `(attn + FFN + expert + embed) * PARAM_BYTES / (TP * PP) + (ln + router) * PARAM_BYTES / PP`. TP shards attention projections, FFN matrices, expert weights, and embeddings. Layer norm and router weights are small and replicated across the TP group. Further divided by DP for ZeRO-3.
- **Gradients**: same size as model parameters per GPU; sharded under ZeRO-2 and ZeRO-3
- **Optimizer states**: Split by EP/DP fix: `model_mem * dense_frac * OPTIM_BYTES / dp + model_mem * moe_frac * OPTIM_BYTES / eff_dp_moe`. When `dp > ep`, `eff_dp_moe = dp / ep`; otherwise `eff_dp_moe = dp`. This correctly models that MoE expert params are only shared within the EP group.
- **Activations**: MoE-aware per-layer cost: MoE layers use `seq * micro * (d * 12 + topk * expert_ffn * 2 * param_bytes)`, dense layers use `seq * micro * d * 12`. Multiplied by `SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER (0.45)`.
- **VP activation overhead** (when PP>1 and VP>1): `(PP-1) * VP * layers_per_virtual_chunk * avg_act_per_layer`. These are full (non-checkpointed) activations for in-flight micro-batches during interleaved 1F1B warmup.
- **Communication buffers**: `NCCL_MEM_BUF (6 GB)` + `NCCL_EP_SCALING_FACTOR * (EP-1)` + MoE dispatch buffers + PP send/recv buffers.
  - MoE dispatch: `moe_layers_per_gpu * 2 * micro * seq * d * param_bytes * (topk/experts_per_gpu)`
  - PP buffers: `2 * micro * seq * d * param_bytes` (send + recv for one micro-batch activation)
- **Fragmentation factor**: `1.24×` applied to raw total. Empirical PyTorch caching allocator overhead calibrated against Qwen3-235B on 64 B300 GPUs.

Finds the minimum ZeRO stage and maximum micro batch size that fits within `GPU_MEM * GPU_MEM_UTILIZATION_THRESHOLD`.

### Phase 2: Batch Configuration

Sweeps all combinations of micro batch size x gradient accumulation steps:
- `tokens_per_batch = micro * SEQ_LEN * accum * DP`
- `training_steps = TOTAL_TOKENS / tokens_per_batch`
- Assigns priority 0 (optimal) through 3 (poor) based on training step count relative to configured bounds

### Phase 3: Training Time

Estimates wall-clock training time:
- `flops_per_token = 6 * active_params_B * 1e9`
- `total_flops = flops_per_token * TOTAL_TOKENS`
- `effective_tflops = peak_tflops * MFU * zero_efficiency`
- `time_seconds = total_flops / (gpus * effective_tflops * 1e12)`
- Applies `TIME_ESTIMATE_LOW_MULTIPLIER` and `TIME_ESTIMATE_HIGH_MULTIPLIER` for range

### Phase 4: ZeRO Communication Overhead

**ZeRO-2 (reduce-scatter):** After backward pass, gradients are reduced and scattered.
- Volume per micro-batch: `model_size_bytes / DP`
- Time: `volume / inter_node_bandwidth`

**ZeRO-1 (all-gather):** Before forward pass, parameters are gathered from shards.
- Volume per micro-batch: `model_size_bytes * (DP - 1) / DP`
- Time: `volume / inter_node_bandwidth`

### Phase 5: All-to-All Communication (MoE)

For MoE models, tokens are routed to expert-holding GPUs via all-to-all collectives:
- Volume per direction: `micro * SEQ_LEN * d * TOPK * PARAM_BYTES / EP`
- Forward + backward = `FWD_BWD_ROUTING_BUFF_PASSES` passes
- Time: `total_volume / intra_node_bandwidth`

For dense models, this phase reports zero overhead.

### Phase 6: PP SendRecv Communication

For models using Pipeline Parallelism (PP>1), computes point-to-point communication overhead:
- **Activation size per send**: `micro * SEQ_LEN * d * dtype_bytes` (one micro-batch's hidden states)
- **Sends per micro-batch**: VP>1 (interleaved): `2*(PP*VP - 1)`; VP=1 (standard): `2*(PP-1)`
- **Communication type detection**: PP ≤ gpus_per_node → intra-node (NVLink); PP > gpus_per_node → inter-node (EFA)
- **Effective bandwidth**:
  - Intra-node: `intra_node_bw_gbps` (e.g., 1800 GB/s NVLink for B300) → ~18 µs per send
  - Inter-node: `inter_node_bw_gb * EFA_PP_EFFICIENCY` (EFA P2P uses only 1-2 of 16 NICs; `EFA_PP_EFFICIENCY = 0.016`) → ~5.2 ms per send
- **Pipeline bubble**: `(PP-1) / (num_microbatches * VP)` %
- **Exposed communication time**: Only warmup (fwd) + cooldown (bwd) sends are serialized. `exposed_sends = 2*(PP-1)`. `estimated_pp_time_ms = exposed_sends * time_per_send_us / 1000`
- **Total traffic per step**: `total_sends * activation_size_bytes`

For PP=1, this phase reports zero overhead.

## Rules

1. **Always ask for model, dataset size, and use case mode before running.** Never assume defaults without confirming with the user.
2. **Apply smart defaults silently.** When the user picks "Llama-3.1-70B", set VOCAB=128256 automatically -- don't ask.
3. **Modify config files in place.** Update `variants_config.py`, `hardware_config.py`, and `project_config.py` before running.
4. **Never modify utility modules.** The files in `utils/` and `model_calculations.py` are the calculator engine -- do not edit them.
5. **Run from the repo directory.** The calculator uses relative imports; run with `/tmp/ai-model-calculator/` as the working directory.
6. **For dense models, set MoE params to zero.** Set `N_EXPERTS=0`, `TOPK=0`, `EP=1` in project_config.py. Set `expert_ffn=0` in the variant dict.
7. **Validate PP divides layers.** Check that `layers % PP == 0` for every variant. If not, suggest valid PP values (divisors of layers).
8. **Validate EP divides N_EXPERTS.** For MoE models, check that `N_EXPERTS % EP == 0`. If not, suggest valid EP values.
9. **Present results in structured format.** Use tables and clear section headers. Highlight OOM cases, best configurations, and warnings.
10. **Offer comparison when multiple models or platforms are selected.** Cross-reference Phase 1 (does it fit?) with Phase 3 (how long?) for a holistic recommendation.
11. **Auto-generate HTML report.** After every calculator run, check if the `aws-branded-report` power is available. If yes, generate a report from the CSV exports. If not installed, skip silently.
12. **For Mode A, sweep node counts.** Run the calculator with multiple hardware configs (e.g., 64, 128, 256, 512 nodes) and find the minimum that meets the time target.
13. **For Mode B, report time clearly.** Present the estimated training time with low/high confidence range and highlight whether it's reasonable.
14. **Always require dataset size.** If the user hasn't specified the number of tokens, ask before proceeding (Modes A and B only; Mode C doesn't need it).
15. **For Mode C, present multiple scenarios.** Always show inference, fine-tuning, and full training memory requirements side by side so the user understands the range.
16. **For Mode C, account for MoE total params.** MoE models need ALL expert parameters in memory even though only top-K are active per token. Use `total_params_B` not `active_params_B` for memory calculations.

## Mode A: Finding minimum instances for a time target

When the user wants to know "how many instances do I need to finish in X time":

1. Run the calculator with ALL available node counts for the selected instance type
2. From Phase 3 results, find the smallest node count where `time_estimate_high <= target_time`
3. Present a table showing all node counts with their estimated times
4. Recommend the minimum node count that meets the target
5. If NO configuration meets the target, say so and suggest the fastest option available

**Quick estimate formula (use before running the full calculator):**
```
time_seconds = (6 * active_params_B * 1e9 * TOTAL_TOKENS) / (gpus * peak_tflops_bf16 * MFU * zero_eff * 1e12)
min_gpus = (6 * active_params_B * 1e9 * TOTAL_TOKENS) / (target_time_seconds * peak_tflops_bf16 * MFU * zero_eff * 1e12)
min_nodes = ceil(min_gpus / gpus_per_node)
```

## Mode B: Estimating training time for fixed instances

When the user provides a fixed number of instances:

1. Configure the hardware with the exact node count provided
2. Run the calculator normally
3. Present the estimated training time with low/high range
4. Contextualize: "This is approximately X days / X weeks / X months"
5. If the model doesn't fit in memory (OOM), flag it and suggest alternatives (more TP, different ZeRO stage, or larger instance type)

## Mode C: Minimum instances to fit a model in memory

When the user wants to know the minimum hardware to hold a model:

1. Determine the memory requirement per GPU based on the parallelism strategy
2. Calculate the minimum number of GPUs needed
3. Convert to number of instances (nodes)
4. Present results for multiple scenarios (inference-only, fine-tuning, full training)

**Memory calculation for different use cases:**

**Inference (weights only):**
```
model_memory_gb = total_params_B * PARAM_BYTES
min_gpus = ceil(model_memory_gb / (mem_gb * GPU_MEM_UTILIZATION_THRESHOLD))
```
- BF16: `PARAM_BYTES = 2` → memory = total_params_B × 2 GB
- FP8: `PARAM_BYTES = 1` → memory = total_params_B × 1 GB

**Fine-tuning (weights + gradients + optimizer for trainable params):**
```
training_memory_gb = total_params_B * (PARAM_BYTES + PARAM_BYTES + OPTIM_BYTES)
                   = total_params_B * (2 + 2 + 6) = total_params_B * 10 GB  (BF16)
min_gpus = ceil(training_memory_gb / (mem_gb * GPU_MEM_UTILIZATION_THRESHOLD))
```
Note: For LoRA/PEFT, only adapter params need gradients+optimizer. Estimate adapter as ~1-5% of total params.

**Full pre-training (weights + gradients + optimizer + activations + buffers):**
```
base_memory_gb = total_params_B * (PARAM_BYTES + PARAM_BYTES + OPTIM_BYTES)
activation_memory_gb = SEQ_LEN * micro_batch * d * EMPIRICAL_ACT_MULTIPLIER * layers * SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER / 1e9
buffer_memory_gb = NCCL_MEM_BUF  (typically 4 GB)
total_memory_gb = base_memory_gb + activation_memory_gb + buffer_memory_gb
min_gpus = ceil(total_memory_gb / (mem_gb * GPU_MEM_UTILIZATION_THRESHOLD))
```

**ZeRO sharding reduces per-GPU memory:**
- ZeRO-1: optimizer states sharded → `optimizer_per_gpu = optimizer_total / DP`
- ZeRO-2: + gradients sharded → `gradients_per_gpu = gradients_total / DP`
- ZeRO-3: + parameters sharded → `params_per_gpu = params_total / DP`

**Presentation format:**

Present a table like this:
```
Model: Qwen3-235B-A22B (235B params, 22B active)
Instance: p5en.48xlarge (8x H200, 141 GB/GPU)

| Use Case        | Precision | Memory Needed | Min GPUs | Min Nodes | ZeRO Stage |
|-----------------|-----------|---------------|----------|-----------|------------|
| Inference       | BF16      | 470 GB        | 4        | 1         | N/A        |
| Inference       | FP8       | 235 GB        | 2        | 1         | N/A        |
| Fine-tuning     | BF16      | 2,350 GB      | 19       | 3         | ZeRO-3     |
| Full training   | BF16      | 2,820 GB      | 22       | 3         | ZeRO-3     |
```

**Additional guidance to include:**
- Recommend the parallelism strategy (TP, PP) that minimizes nodes while maintaining efficiency
- Flag if the model requires multi-node even for inference
- For MoE models, note that only active params matter for compute but ALL params must be in memory
- Suggest the most cost-effective instance type if the user hasn't chosen one

## Output behavior

The calculator prints colored terminal output and exports CSV/JSON files. When presenting results:

- **Summarize, don't dump raw output.** Parse the console output and present key findings in markdown tables.
- **Highlight actionable items:** OOM warnings, recommended ZeRO stages, best batch configurations, estimated training time ranges.
- **For comparisons:** Create side-by-side tables showing how different hardware platforms or model variants compare.
- **CSV files** are written to `/tmp/ai-model-calculator/`. Mention their paths so the user can find them.
- **Auto-generate HTML report.** After presenting markdown results, check for the `aws-branded-report` power and generate an HTML report from the CSV exports. See `report-integration.md` for details.
