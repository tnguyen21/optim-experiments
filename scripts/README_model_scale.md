# Model Scale Sweep

This directory contains scripts for analyzing how optimizer trends (effective rank, sparsity, generalization) change across different model scales.

## Motivation

The original Pasanu et al. findings were on 1-2 layer MLPs. We want to validate whether second-order optimizers (like Muon) continue to produce lower-rank solutions and better generalization as we scale to larger Vision Transformers.

## Model Scales

The script tests 5 different model scales:

| Scale | Parameters | Config | Target Use |
|-------|------------|--------|------------|
| **Tiny** | 2.7M | dim=192, depth=6, heads=3 | Baseline (current) |
| **Small** | 10.7M | dim=384, depth=6, heads=6 | ~4x scaling |
| **Medium** | 25.3M | dim=512, depth=8, heads=8 | ~10x scaling |
| **Large** | 56.8M | dim=768, depth=8, heads=12 | ~20x scaling |
| **XL** | 100.9M | dim=1024, depth=8, heads=16 | ~40x scaling |

## Usage

### Quick Test (3 scales, 15 epochs each)
```bash
uv run python scripts/model_scale_sweep.py --quick
```

### Full Sweep (5 scales, 30 epochs each)
```bash
uv run python scripts/model_scale_sweep.py
```

### Single Optimizer
```bash
uv run python scripts/model_scale_sweep.py --optimizer muon
uv run python scripts/model_scale_sweep.py --optimizer adamw --epochs 20
```

### Parameter Counting
```bash
uv run python scripts/count_model_params.py
```

## Adaptive Features

The script automatically adjusts:

- **Batch Size**: Larger models use smaller batch sizes to fit in GPU memory
  - <10M params: batch_size=4096
  - <50M params: batch_size=2048  
  - ≥50M params: batch_size=1024

- **Learning Rates**: Uses linear scaling rule based on batch size
  - SGD: 0.1 × (batch_size/4096)
  - AdamW: 0.001 × (batch_size/4096)
  - Muon: 0.01 × (batch_size/4096)

## Expected Results

Based on scaling laws and Pasanu et al., we expect:

1. **Performance**: All optimizers should improve with scale, but Muon may maintain its advantage
2. **Rank Trends**: Second-order methods should continue producing lower-rank solutions at scale
3. **Sparsity**: Activation sparsity patterns may change with model capacity
4. **Generalization Gap**: Larger models may show bigger train/val gaps, testing robustness

## Analysis Integration

Results can be analyzed with existing tools:

```bash
# After running model scale sweep, analyze the trained models
uv run python scripts/summarize_optimizer_differences.py

# If checkpoints are saved, analyze temporal dynamics  
uv run python scripts/analyze_checkpoint_evolution.py
```

## Memory Requirements

Estimated GPU memory usage:
- **Tiny (2.7M)**: ~2GB
- **Small (10.7M)**: ~4GB  
- **Medium (25.3M)**: ~6GB
- **Large (56.8M)**: ~10GB
- **XL (100.9M)**: ~16GB

For systems with limited GPU memory, use `--quick` mode or test individual scales.

## Research Questions

This sweep helps answer:

1. **Scale Robustness**: Do second-order optimizer advantages hold at 100M+ parameters?
2. **Rank Scaling**: How does effective rank scale with model capacity across optimizers?
3. **Efficiency**: At what scale do second-order methods become prohibitively expensive?
4. **Generalization**: Do larger Muon-trained models generalize better than AdamW equivalents?

## Output

The script generates:
- `model_scale_sweep_TIMESTAMP/` directory
- `scale_sweep_results.json` with detailed metrics
- `model_scale_sweep.png` with performance vs scale plots
- Console summary table with key results