# DGX Spark Memory Profiling Methodology

How to safely determine maximum batch sizes on DGX Spark unified memory.

## The Problem

Standard profiling approaches fail on DGX Spark:
- **OOM kills the system**, not just the process
- **Memory monitoring is reactive** - by the time you detect high usage, it's too late
- **Sweeps are dangerous** - running multiple configs risks bricking the machine

## Methodology: Incremental Profiling

### Step 1: Clean Slate

```bash
# Use the OOM protection script
./scripts/oom_protection.sh preflight 80
```

Or manually:
```bash
ps aux | grep -E "(ray|vllm|python)" | grep -v grep
pkill -9 -f "ray::" 2>/dev/null || true
pkill -9 -f "vllm" 2>/dev/null || true
sleep 15
free -h  # Should show ~115GB free
```

### Step 2: Start Small

Run with **minimum viable batch size** first:
- SFT: `batch_size=1, grad_accum=1`
- DPO: `batch_size=1, grad_accum=1`
- GRPO: `batch_size=1, grad_accum=1, vllm_gpu_memory_utilization=0.1`

### Step 3: Monitor During Training

In a separate terminal:
```bash
watch -n 5 'free -h | head -2; echo "---"; nvidia-smi --query-gpu=memory.used --format=csv 2>/dev/null || echo "nvidia-smi N/A"'
```

### Step 4: Document Peak Memory

After each successful run, record:
1. **Peak memory used** (from monitoring)
2. **Batch size and settings**
3. **Memory headroom** (128GB - peak)

### Step 5: Increment Carefully

Double one parameter at a time:
1. `batch_size`: 1 → 2 → 4 → 8 → 16 → 32
2. `grad_accum`: if batch_size hits limit
3. `max_seq_length`: 1024 → 2048 → 4096

**Stop when**: Peak memory exceeds 100GB (leaving 28GB headroom for OS/SSH/agents)

### Step 6: Record Last Working Config

When you hit OOM, the **previous configuration** is your safe maximum.

---

## Results Template

Copy this for your own profiling:

### [Model Name] [Training Type]

| batch | grad_accum | seq_len | peak_mem | headroom | status | notes |
|-------|------------|---------|----------|----------|--------|-------|
| 1 | 1 | 1024 | ?GB | ?GB | pending | baseline |
| 2 | 1 | 1024 | ?GB | ?GB | pending | |
| 4 | 1 | 1024 | ?GB | ?GB | pending | |
| 8 | 1 | 1024 | ?GB | ?GB | pending | |

---

## Memory Estimation Formulas

Rough estimates (approximations):

```
SFT memory  ≈ model_params_B × 6 + activation_memory
DPO memory  ≈ SFT × 1.3 (policy + frozen reference)
GRPO memory ≈ SFT + vllm_gpu_memory_utilization × 119GB
LoRA memory ≈ model_params_B × 2 + trainable_params × 6 + activation_memory
```

Where:
- `model_params_B × 6`: weights (bf16) + optimizer states + gradients
- `activation_memory`: scales with batch × seq_len × hidden_dim

---

## Key Insights from Profiling

1. **Memory scales super-linearly with batch size**
   - Doubling batch from 8→16 may add 34GB, not 24GB
   - This is due to activation memory growth

2. **DPO uses ~1.3x SFT memory** (not 2x as expected)
   - Reference model is frozen, no optimizer states

3. **LoRA is very memory-efficient**
   - Can use larger batches or longer sequences
   - Gradient checkpointing often unnecessary

4. **Gradient accumulation doesn't increase peak memory**
   - Use grad_accum to increase effective batch size safely
