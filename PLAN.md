# Optimizer Quality Experiments - Initial Setup Instructions

## Project Overview
Build a minimal training pipeline to compare how different optimizers (SGD, AdamW) produce qualitatively different solutions when training ResNet-8 on CIFAR-10. This is Phase 0: establish the training infrastructure with deterministic, reproducible experiments.

## Repository Structure
Create the following file structure:

```
optimizer-quality-experiments/
├── README.md                 # Project description and usage
├── requirements.txt          # Python dependencies
├── config/
│   └── base_config.yaml     # Hyperparameters and experiment settings
├── src/
│   ├── __init__.py
│   ├── models.py            # ResNet-8 architecture
│   ├── data.py              # CIFAR-10 data loading with deterministic splits
│   ├── train.py             # Training loop
│   └── utils.py             # Logging, seed setting, metrics
├── experiments/
│   ├── logs/                # Trackio logs
│   └── results/             # Final model weights and metrics
└── scripts/
    └── run_experiment.py    # Main entry point
```

## Implementation Requirements

### 1. `requirements.txt`
Include minimal dependencies:
```
torch
torchvision
numpy
pyyaml
trackio
tqdm
```

### 2. `config/base_config.yaml`
Create a config file with these settings:
```yaml
# Experiment settings
experiment_name: "resnet8_cifar10_baseline"
seed: 42
device: "cuda"  # or "cpu"

# Data settings
dataset: "cifar10"
data_dir: "./data"
num_workers: 4
train_val_split: 0.9  # 90% train, 10% val from training set

# Model settings
model: "resnet8"
num_classes: 10

# Training settings
batch_size: 128
num_epochs: 200
eval_every: 10  # Evaluate every N epochs

# Optimizer settings (will override for each experiment)
optimizer:
  type: "sgd"  # or "adamw"
  lr: 0.1
  momentum: 0.9  # For SGD
  weight_decay: 0.0001
  # betas: [0.9, 0.999]  # For AdamW

# Logging
log_dir: "./experiments/logs"
```

### 3. `src/models.py`
Implement a ResNet-8 architecture. Requirements:
- Use standard ResNet building blocks (BasicBlock with 3x3 convolutions)
- Architecture: Initial conv -> 3 stages (each with 2 BasicBlocks) -> avgpool -> fc
- Channel progression: 16 -> 32 -> 64
- Total layers: 1 (initial conv) + 2*3 (6 blocks with 2 conv each = 6 layers) + 1 (fc) = 8 layers
- Include batch normalization
- Make it easy to extract intermediate activations for later analysis

Example structure:
```python
class ResNet8(nn.Module):
    def __init__(self, num_classes=10):
        # Stage 1: 16 channels, 2 blocks
        # Stage 2: 32 channels, 2 blocks  
        # Stage 3: 64 channels, 2 blocks
        # avgpool + fc
```

### 4. `src/data.py`
Implement deterministic CIFAR-10 loading:
- Use `torch.manual_seed()` before creating train/val split
- Standard CIFAR-10 preprocessing:
  - Training: RandomCrop(32, padding=4), RandomHorizontalFlip, Normalize
  - Validation/Test: Normalize only
  - Mean: [0.4914, 0.4822, 0.4465], Std: [0.2023, 0.1994, 0.2010]
- Create a deterministic train/val split from the training set
- Keep test set separate for final evaluation
- Log the exact indices used for train/val split (save to file)

Function signatures:
```python
def get_cifar10_dataloaders(config):
    """
    Returns: train_loader, val_loader, test_loader
    """
    pass

def save_split_indices(train_indices, val_indices, save_path):
    """Save split indices for reproducibility"""
    pass
```

### 5. `src/train.py`
Implement the main training loop:

**Key requirements:**
- Set ALL random seeds at the start (torch, numpy, random)
- Set `torch.backends.cudnn.deterministic = True`
- Set `torch.backends.cudnn.benchmark = False`
- Standard training loop with:
  - Forward pass
  - Loss computation (CrossEntropyLoss)
  - Backward pass
  - Optimizer step (no LR scheduling)
- Track metrics: train loss, val loss, train acc, val acc
- Log to Trackio every epoch
- Save final model weights only

Function signatures:
```python
def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Returns: avg_loss, avg_acc"""
    pass

def validate(model, val_loader, criterion, device):
    """Returns: avg_loss, avg_acc"""
    pass

def train(model, train_loader, val_loader, test_loader, optimizer, config, tracker):
    """Main training loop - returns final metrics"""
    pass
```

### 6. `src/utils.py`
Utility functions:

```python
def set_seed(seed):
    """Set all random seeds for reproducibility"""
    pass

def get_optimizer(model, config):
    """Create optimizer based on config"""
    pass

def setup_logging(config):
    """Setup Trackio tracker and create directories"""
    pass

def save_final_model(model, filepath, metrics):
    """Save final model weights and summary metrics"""
    pass
```

### 7. `scripts/run_experiment.py`
Main entry point to run experiments:

```python
"""
Usage:
python scripts/run_experiment.py --config config/base_config.yaml --optimizer sgd
python scripts/run_experiment.py --config config/base_config.yaml --optimizer adamw
"""

def main():
    # Parse arguments
    # Load config
    # Override optimizer if specified
    # Set seeds
    # Create dataloaders
    # Create model
    # Create optimizer
    # Setup Trackio logging
    # Train
    # Save final model and results
    pass
```

Should support command-line override of key parameters:
- `--config`: path to config file
- `--optimizer`: optimizer type (sgd, adamw)
- `--lr`: learning rate
- `--seed`: random seed
- `--experiment-name`: name for this run

## Minimal Test Plan

Once implemented, verify the setup works:

### Test 1: Deterministic Training
```bash
# Run twice with same seed and optimizer
python scripts/run_experiment.py --optimizer sgd --seed 42
python scripts/run_experiment.py --optimizer sgd --seed 42
# Verify: losses should be IDENTICAL at each epoch
```

### Test 2: Different Optimizers
```bash
# Run with different optimizers
python scripts/run_experiment.py --optimizer sgd --seed 42
python scripts/run_experiment.py --optimizer adamw --seed 42
# Verify: training curves should differ
```

### Test 3: Results Saved
```bash
# After training completes
ls experiments/results/
# Verify: final model weights and metrics.json exist
```

## Success Criteria for Phase 0

✅ Code runs without errors  
✅ Training is fully deterministic (same seed = identical results)  
✅ Both SGD and AdamW train successfully to >70% test accuracy  
✅ Final models save correctly with summary metrics  
✅ Trackio UI shows clear training curves  
✅ Total training time < 30 minutes on GPU (< 3 hours on CPU)  

## Expected Outputs After Running

```
experiments/
├── logs/
│   ├── sgd_seed42/
│   │   └── trackio_logs.json
│   └── adamw_seed42/
│       └── trackio_logs.json
└── results/
    ├── sgd_seed42/
    │   ├── final_model.pt
    │   └── metrics.json
    └── adamw_seed42/
        ├── final_model.pt
        └── metrics.json
```

## Implementation Notes

- **Start simple**: Get SGD working first, then add AdamW
- **Verify determinism early**: Run same experiment twice before proceeding
- **Minimal logging**: Just epoch-level metrics (train/val loss and accuracy)
- **Use descriptive experiment names**: Include optimizer, seed, date
- **Don't optimize prematurely**: Focus on correctness first, speed later
- **Trackio setup**: Use `trackio.Tracker()` to log metrics - it will auto-generate a UI

## Next Steps (After Phase 0)

Once this minimal pipeline works:
1. Add hyperparameter sweep functionality
2. Implement basic analysis: plot loss curves, compare final accuracies
3. Add model inspection: save weight statistics, activation norms
4. Begin feature visualization pipeline

## Questions to Resolve During Implementation

- What GPU is available? (Affects batch size choice)
- Should we track gradient norms per layer? (Yes, minimal overhead, useful later)

## Common Pitfalls to Avoid

❌ Not setting cudnn.deterministic - will cause non-reproducible results  
❌ Different data augmentation between runs - breaks comparisons  
❌ Forgetting to set eval() mode during validation  
❌ Logging too frequently - slows down training unnecessarily  

---

**Start with**: Get the basic training loop working for SGD on CIFAR-10 with ResNet-8. Keep it simple: no LR scheduling, no checkpointing, just train to completion and save the final model. Everything should be minimal, clean, and reproducible. We'll add complexity incrementally.
