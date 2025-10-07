# Configuration Files

This directory contains optimized configuration files for training ViT-Tiny on CIFAR-10.

## Optimizer-Specific Configs (Recommended)

These configs use learning rates optimized via comprehensive LR sweep for **batch size 4096**:

| Config File | Optimizer | Optimal LR | Best Val Acc | Description |
|------------|-----------|------------|--------------|-------------|
| `muon_optimal_config.yaml` | Muon | 0.01 | **75.95%** | 🏆 Best performing |
| `adamw_optimal_config.yaml` | AdamW | 0.001 | 71.49% | Solid baseline |
| `sgd_optimal_config.yaml` | SGD | 0.1 | 53.97% | Classical optimizer |

## Usage Examples

```bash
# Use best performing optimizer (Muon)
python scripts/run_experiment.py --config config/muon_optimal_config.yaml

# Compare with AdamW
python scripts/run_experiment.py --config config/adamw_optimal_config.yaml

# Test SGD baseline
python scripts/run_experiment.py --config config/sgd_optimal_config.yaml
```

## LR Sweep Results Summary

The learning rates were optimized through a systematic sweep across multiple values:

- **SGD**: Tested `[0.01, 0.03, 0.1, 0.3, 1.0, 3.0]` → Optimal: `0.1`
- **AdamW**: Tested `[1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]` → Optimal: `0.001`  
- **Muon**: Tested `[0.001, 0.003, 0.01, 0.03, 0.1, 0.3]` → Optimal: `0.01`

All training runs were stable and completed 50 epochs successfully.

## Batch Size Dependency

⚠️ **Important**: These learning rates are optimized specifically for **batch size 4096**. 

For different batch sizes, you may need to adjust learning rates:
- Smaller batch sizes → Lower LRs
- Larger batch sizes → Higher LRs

## Base Config

`base_config.yaml` - Template configuration with Muon defaults. Use optimizer-specific configs for best results.

## Other Files

- `test_config.yaml` - Configuration for testing/debugging (if present)

## Reproducing LR Sweep

To re-run the learning rate sweep:

```bash
# Full sweep (takes ~6-8 hours)
python scripts/lr_sweep.py

# Quick test (takes ~1 hour)
python scripts/lr_sweep.py --quick

# Single optimizer
python scripts/lr_sweep.py --optimizer muon
```