# Code Review: model_calculations.py - Improvements

## Critical Issues (Fix Immediately)

### 1. Inconsistent PP/TP usage in PHASE 1 (Lines 114-117)
**Current:**
```python
pp = 1
tp = 1
dp_effective = total_gpus // (tp * pp)
```

**Should be:**
```python
dp_effective = total_gpus // (TP * PP)
```

**Impact:** Ignores user-configured PP/TP values, breaks when PP>1 or TP>1

---

### 2. Inconsistent layers_per_gpu calculation (Line 130)
**Current:**
```python
layers_per_gpu = layers  # PP=1
```

**Should be:**
```python
layers_per_gpu = layers // PP
```

**Impact:** Incorrect memory calculations when PP>1

---

### 3. Fragile hardware detection (Lines 153-156)
**Current:**
```python
if hw["label"] == "H200":
    micro = MBS_H200
else:
    micro = MBS_H100
```

**Should be:**
```python
if hw["mem_gb"] >= 141:
    micro = MBS_H200
else:
    micro = MBS_H100
```

**Impact:** Breaks if hardware label format changes, less maintainable

---

### 4. Hardcoded memory threshold (Line 237)
**Current:**
```python
if hw["mem_gb"] >= 141:  # H200 with high memory
```

**Should add constant:**
```python
H200_MEM_THRESHOLD_GB = 141  # at top of file
# Then use:
if hw["mem_gb"] >= H200_MEM_THRESHOLD_GB:
```

**Impact:** Magic number repeated, hard to maintain

---

## Moderate Issues (Should Fix)

### 5. MARGIN defined in wrong location (Line 196)
**Current:** Defined inside PHASE 2 section

**Should:** Move to constants section (lines 6-106) with comment:
```python
MARGIN = 0.02  # Tolerance for matching target batch sizes (2%)
```

---

### 6. Unused variable (Line 185)
**Current:**
```python
fits = total_best <= gpu_mem
```

**Action:** Remove - variable calculated but never used

---

### 7. Poor variable naming (Lines 18-20)
**Current:**
```python
BASE_MONTHS=4.0
LOW_RANGE=0.75
UPPER_RANGE=1.25
```

**Better:**
```python
BASE_TRAINING_TIME_MONTHS = 4.0
ESTIMATE_UNCERTAINTY_LOW = 0.75
ESTIMATE_UNCERTAINTY_HIGH = 1.25
```

---

### 8. Outdated comment (Line 133)
**Current:**
```python
# Model params (BF16)
```

**Should be:**
```python
# Model parameters memory (precision determined by PARAM_BYTES)
```

---

### 9. Incorrect section title (Line 227)
**Current:**
```python
print("TRAINING TIME COMPARISON: ALL PLATFORMS (Variant B)")
```

**Should be:**
```python
print("TRAINING TIME COMPARISON: ALL PLATFORMS")
```

---

### 10. Confusing constant comment (Line 89)
**Current:**
```python
FFN_WEIGHT_MATRICES=3 # for query, key, value projections in attention and for the two matrices in dense FFN layers
```

**Issue:** Mentions attention Q/K/V but this is for FFN, not attention

**Better:**
```python
FFN_WEIGHT_MATRICES=3 # 3 weight matrices in dense FFN: gate, up, down projections (SwiGLU)
```

---

### 11. Typo in variable name (Line 93)
**Current:**
```python
SELECTIVE_ACT_CHEKPOINTING_MULTIPLIER
```

**Should be:**
```python
SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER  # Fixed spelling
```

---

## Code Quality Improvements

### 12. Add hardware memory constants
```python
# Hardware specifications
H200_MEM_GB = 141
H100_MEM_GB = 80
H200_NVLINK_GBPS = 900
H100_NVLINK_GBPS = 400
```

---

### 13. Code duplication - model memory calculation
**Issue:** Model memory calculation repeated 3 times (lines 134-142, 309-317)

**Suggestion:** Create helper function:
```python
def calculate_model_memory_gb(variant, layers_per_gpu, experts_per_gpu, param_bytes):
    """
    Calculate model memory in GB for a given variant.

    Args:
        variant: Variant dictionary with architecture specs
        layers_per_gpu: Number of layers per GPU (total_layers // PP)
        experts_per_gpu: Number of experts per GPU (N_EXPERTS // EP)
        param_bytes: Bytes per parameter (2 for BF16, 1 for FP8)

    Returns:
        float: Model memory in GB
    """
    d = variant["d"]
    layers = variant["layers"]
    moe_layers = layers - variant["dense_layers"]

    attn_mem = layers_per_gpu * variant["attn_per_layer"] * param_bytes
    routed_mem = experts_per_gpu * moe_layers * variant["expert_each"] * param_bytes
    shared_mem = moe_layers * variant["shared_each"] * param_bytes
    router_mem = moe_layers * variant["router_each"] * param_bytes
    ln_mem = layers * LAYER_NORM * d * param_bytes
    dense_ffn_mem = variant["dense_layers"] * FFN_WEIGHT_MATRICES * d * variant["dense_ffn"] * param_bytes
    embed_mem = NUM_EMBEDDINGS_TABLES * VOCAB * d * param_bytes

    model_mem_bytes = attn_mem + routed_mem + shared_mem + router_mem + ln_mem + dense_ffn_mem + embed_mem
    return model_mem_bytes / 1e9
```

---

### 14. Missing validation
Add validation at start:
```python
# Validation
assert EP > 0 and N_EXPERTS % EP == 0, f"N_EXPERTS ({N_EXPERTS}) must be divisible by EP ({EP})"
assert PP > 0, f"PP must be positive, got {PP}"
assert TP > 0, f"TP must be positive, got {TP}"
assert PRECISION in ["BF16", "FP8"], f"PRECISION must be BF16 or FP8, got {PRECISION}"

for hw in HARDWARE:
    total_gpus = hw["gpus"]
    required_gpus = TP * PP * EP
    assert total_gpus >= required_gpus, f"{hw['name']}: {total_gpus} GPUs < minimum required {required_gpus} (TP={TP} * PP={PP} * EP={EP})"
    assert total_gpus % required_gpus == 0, f"{hw['name']}: {total_gpus} GPUs not divisible by {required_gpus}"
```

---

### 15. Add type hints and docstrings
For future Python 3.9+ compatibility:
```python
from typing import Dict, List

def calculate_dp(total_gpus: int, tp: int, pp: int, ep: int) -> int:
    """Calculate data parallelism degree."""
    return total_gpus // (tp * pp * ep)
```

---

### 16. Improve output formatting
Add hardware name to batch size analysis output (Line 200):
```python
print(f"\n--- {hw['name']}: All micro/accum combinations (DP={dp}) ---")
```

---

## Recommended File Structure

```python
#!/usr/bin/python3
# 1. Imports (if any)

# 2. USER CONFIGURATION - Model Architecture
# 3. USER CONFIGURATION - Hardware
# 4. CONSTANTS - Training hyperparameters
# 5. CONSTANTS - Memory estimations
# 6. CONSTANTS - Assessment ranges
# 7. DERIVED CONSTANTS
# 8. VALIDATION
# 9. HELPER FUNCTIONS (if added)
# 10. PHASE 1: Memory Analysis
# 11. PHASE 2: Batch Size Analysis
# 12. PHASE 3: Training Time Estimates
# 13. PHASE 4: ZeRO-2 Communication Analysis
# 14. PHASE 5: All-to-All Communication Analysis
```

---

## Priority Fixes

**High Priority (Do First):**
1. Fix PP/TP usage in PHASE 1 (Issue #1, #2)
2. Fix hardware detection logic (Issue #3)
3. Fix memory threshold magic number (Issue #4)

**Medium Priority:**
4. Fix variable names (Issue #7)
5. Move MARGIN constant (Issue #5)
6. Fix typo in CHECKPOINTING (Issue #11)

**Low Priority (Nice to Have):**
7. Refactor duplicate code (Issue #13)
8. Add validation (Issue #14)
9. Improve comments and output (Issues #8, #9, #10, #16)

---

## Testing After Changes

After applying fixes, verify:
1. Run with PP=1, TP=1 (default) - should match current output
2. Run with PP=2, TP=1 - verify layers_per_gpu is correctly halved
3. Run with different PRECISION="FP8" - verify PARAM_BYTES=1 used
4. Add a new hardware config - verify it's automatically picked up
5. Change MICROS list - verify all phases use new values
