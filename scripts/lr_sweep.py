#!/usr/bin/env python3
"""
Learning Rate Sweep Script

Sweeps across different learning rates for each optimizer to find optimal LRs
for batch size 4096. Trains for ~50 epochs to get reliable estimates.

Usage:
python scripts/lr_sweep.py [--optimizer OPTIMIZER] [--quick]
"""

import os
import sys
import json
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from models import ViTTiny
from data import get_cifar10_dataloaders
from utils import get_optimizer, set_seed
from train import train_one_epoch, validate


# Learning rate ranges for each optimizer (optimized for batch size 4096)
LR_RANGES = {
    "sgd": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],  # SGD typically needs higher LRs
    "adamw": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],  # AdamW prefers lower LRs
    "muon": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3],  # Muon somewhere in between
}

# Reduced ranges for quick testing
LR_RANGES_QUICK = {
    "sgd": [0.03, 0.1, 0.3],
    "adamw": [1e-4, 3e-4, 1e-3],
    "muon": [0.003, 0.01, 0.03],
}


def create_model():
    """Create a fresh model instance"""
    return ViTTiny(num_classes=10)


def train_for_lr_sweep(model, train_loader, val_loader, optimizer_config, num_epochs=50, device="cuda"):
    """
    Train model for learning rate sweep

    Returns:
        Dict with training metrics
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Create optimizer
    config = {"optimizer": optimizer_config}
    optimizer = get_optimizer(model, config)

    # Training history
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epochs": []}

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        # Training
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)

        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Track best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

        # Store metrics
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epochs"].append(epoch)

        # Early stopping if training becomes unstable
        if train_loss > 10.0 or np.isnan(train_loss):
            print(f"  Training became unstable at epoch {epoch}, stopping early")
            break

    # Final metrics
    final_metrics = {
        "final_train_acc": history["train_acc"][-1] if history["train_acc"] else 0.0,
        "final_val_acc": history["val_acc"][-1] if history["val_acc"] else 0.0,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else float("inf"),
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else float("inf"),
        "epochs_completed": len(history["epochs"]),
        "training_stable": not (np.isnan(history["train_loss"][-1]) if history["train_loss"] else True),
    }

    return final_metrics, history


def run_lr_sweep(optimizer_name, lr_values, num_epochs=50, seed=42, quick=False):
    """
    Run learning rate sweep for a given optimizer

    Returns:
        List of results for each LR
    """
    print(f"\n{'=' * 60}")
    print(f"Running LR sweep for {optimizer_name.upper()}")
    print(f"Learning rates: {lr_values}")
    print(f"Training for {num_epochs} epochs each")
    print(f"{'=' * 60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup data loaders (batch size 4096)
    batch_size = 4096 if not quick else 1024  # Reduce batch size for quick testing
    config = {
        "data_dir": "data",
        "batch_size": batch_size,
        "num_workers": 4,
        "train_val_split": 0.8,
        "seed": seed,
        "log_dir": "temp",
    }

    print(f"Loading dataset with batch size {batch_size}...")
    train_loader, val_loader, _ = get_cifar10_dataloaders(config)

    results = []

    for lr in tqdm(lr_values, desc=f"Testing LRs for {optimizer_name}"):
        print(f"\n--- Testing LR: {lr} ---")

        # Set seed for reproducibility
        set_seed(seed)

        # Create fresh model
        model = create_model()

        # Create optimizer config
        optimizer_config = {
            "type": optimizer_name,
            "lr": lr,
            "weight_decay": 0.1 if optimizer_name == "muon" else 0.0001,
        }

        if optimizer_name == "sgd":
            optimizer_config["momentum"] = 0.9
        elif optimizer_name == "adamw":
            optimizer_config["betas"] = [0.9, 0.999]
        elif optimizer_name == "muon":
            optimizer_config["momentum"] = 0.95
            optimizer_config["nesterov"] = True

        try:
            # Train model
            final_metrics, history = train_for_lr_sweep(model, train_loader, val_loader, optimizer_config, num_epochs, device)

            # Store results
            result = {
                "optimizer": optimizer_name,
                "lr": lr,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "seed": seed,
                **final_metrics,
                "history": history,
            }
            results.append(result)

            print(f"  Final val acc: {final_metrics['final_val_acc']:.2f}%")
            print(f"  Best val acc: {final_metrics['best_val_acc']:.2f}% (epoch {final_metrics['best_epoch']})")
            print(f"  Training stable: {final_metrics['training_stable']}")

        except Exception as e:
            print(f"  Error training with LR {lr}: {e}")
            # Store failed result
            result = {
                "optimizer": optimizer_name,
                "lr": lr,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "seed": seed,
                "final_train_acc": 0.0,
                "final_val_acc": 0.0,
                "best_val_acc": 0.0,
                "best_epoch": 0,
                "final_train_loss": float("inf"),
                "final_val_loss": float("inf"),
                "epochs_completed": 0,
                "training_stable": False,
                "error": str(e),
            }
            results.append(result)

    return results


def plot_lr_sweep_results(all_results, output_dir):
    """
    Create visualizations of LR sweep results
    """
    plt.style.use("seaborn-v0_8")
    sns.set_palette("Set2")

    # Convert to DataFrame
    data = []
    for result in all_results:
        if "error" not in result:  # Skip failed runs
            data.append(
                {
                    "optimizer": result["optimizer"],
                    "lr": result["lr"],
                    "final_val_acc": result["final_val_acc"],
                    "best_val_acc": result["best_val_acc"],
                    "training_stable": result["training_stable"],
                }
            )

    if not data:
        print("No successful results to plot")
        return

    df = pd.DataFrame(data)

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Final validation accuracy vs LR
    ax1 = axes[0]
    for optimizer in df["optimizer"].unique():
        opt_data = df[df["optimizer"] == optimizer]
        stable_data = opt_data[opt_data["training_stable"]]
        unstable_data = opt_data[~opt_data["training_stable"]]

        if not stable_data.empty:
            ax1.plot(stable_data["lr"], stable_data["final_val_acc"], "o-", label=f"{optimizer.upper()}", linewidth=2, markersize=8)

        if not unstable_data.empty:
            ax1.scatter(unstable_data["lr"], unstable_data["final_val_acc"], marker="x", s=100, alpha=0.7, color="red")

    ax1.set_xscale("log")
    ax1.set_xlabel("Learning Rate")
    ax1.set_ylabel("Final Validation Accuracy (%)")
    ax1.set_title("Final Validation Accuracy vs Learning Rate")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Best validation accuracy vs LR
    ax2 = axes[1]
    for optimizer in df["optimizer"].unique():
        opt_data = df[df["optimizer"] == optimizer]
        stable_data = opt_data[opt_data["training_stable"]]
        unstable_data = opt_data[~opt_data["training_stable"]]

        if not stable_data.empty:
            ax2.plot(stable_data["lr"], stable_data["best_val_acc"], "o-", label=f"{optimizer.upper()}", linewidth=2, markersize=8)

        if not unstable_data.empty:
            ax2.scatter(unstable_data["lr"], unstable_data["best_val_acc"], marker="x", s=100, alpha=0.7, color="red")

    ax2.set_xscale("log")
    ax2.set_xlabel("Learning Rate")
    ax2.set_ylabel("Best Validation Accuracy (%)")
    ax2.set_title("Best Validation Accuracy vs Learning Rate")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "lr_sweep_results.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Create summary table
    print("\n" + "=" * 80)
    print("LR SWEEP SUMMARY")
    print("=" * 80)

    for optimizer in df["optimizer"].unique():
        opt_data = df[df["optimizer"] == optimizer].sort_values("best_val_acc", ascending=False)
        stable_data = opt_data[opt_data["training_stable"]]

        print(f"\n{optimizer.upper()}:")
        if not stable_data.empty:
            best_row = stable_data.iloc[0]
            print(f"  Best LR: {best_row['lr']:.6f}")
            print(f"  Best Val Acc: {best_row['best_val_acc']:.2f}%")
            print(f"  Final Val Acc: {best_row['final_val_acc']:.2f}%")
        else:
            print("  No stable training runs found!")

        print(f"  Stable runs: {len(stable_data)}/{len(opt_data)}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Learning Rate Sweep")
    parser.add_argument(
        "--optimizer", type=str, choices=["sgd", "adamw", "muon", "all"], default="all", help="Optimizer to sweep (default: all)"
    )
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer LRs and epochs")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"lr_sweep_results_{timestamp}")
    output_dir.mkdir(exist_ok=True)

    print("LR Sweep Analysis")
    print(f"Output directory: {output_dir}")
    print(f"Quick mode: {args.quick}")
    print(f"Epochs per LR: {args.epochs}")

    # Choose LR ranges
    lr_ranges = LR_RANGES_QUICK if args.quick else LR_RANGES
    num_epochs = 10 if args.quick else args.epochs

    # Determine which optimizers to test
    if args.optimizer == "all":
        optimizers_to_test = list(lr_ranges.keys())
    else:
        optimizers_to_test = [args.optimizer]

    # Run sweeps
    all_results = []

    for optimizer_name in optimizers_to_test:
        lr_values = lr_ranges[optimizer_name]
        results = run_lr_sweep(optimizer_name, lr_values, num_epochs, args.seed, args.quick)
        all_results.extend(results)

    # Save results
    results_file = output_dir / "lr_sweep_results.json"
    print(f"\nSaving results to {results_file}...")

    # Remove history from saved results to keep file size manageable
    results_to_save = []
    for result in all_results:
        result_copy = result.copy()
        if "history" in result_copy:
            del result_copy["history"]  # Remove detailed history
        results_to_save.append(result_copy)

    with open(results_file, "w") as f:
        json.dump(results_to_save, f, indent=2)

    # Generate visualizations
    print("Generating visualizations...")
    try:
        plot_lr_sweep_results(all_results, output_dir)
        print("✓ Visualizations complete!")
    except Exception as e:
        print(f"Warning: Could not generate visualizations: {e}")

    print("\n✓ LR sweep complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
