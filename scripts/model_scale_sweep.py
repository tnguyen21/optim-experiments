#!/usr/bin/env python3
"""
Model Scale Sweep Script

Trains models at different parameter scales (5M, 10M, 50M, 100M+) to examine how
optimizer trends (effective rank, sparsity) scale with model size.

Usage:
python scripts/model_scale_sweep.py [--optimizer OPTIMIZER] [--quick]
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
from data import get_cifar10_dataloaders
from utils import get_optimizer, set_seed
from train import train_one_epoch, validate


def count_parameters(model):
    """Count trainable parameters in model"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Detailed breakdown
    param_breakdown = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_breakdown[name] = param.numel()
    
    return total_params, param_breakdown


class ViTScaled(nn.Module):
    """Scalable Vision Transformer for different parameter counts"""
    
    def __init__(self, img_size=32, patch_size=4, num_classes=10, 
                 embed_dim=192, depth=6, num_heads=3, mlp_ratio=4.0):
        super(ViTScaled, self).__init__()
        
        # Same structure as ViTTiny but with configurable dimensions
        from models import PatchEmbedding, TransformerBlock
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Learnable position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio) 
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Store config for analysis
        self.config = {
            'embed_dim': embed_dim,
            'depth': depth, 
            'num_heads': num_heads,
            'mlp_ratio': mlp_ratio
        }
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights following ViT paper"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x, _ = block(x)
        
        # Classification
        x = self.norm(x)
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        
        return x
    
    def get_features(self, x):
        """Extract intermediate features for analysis (same as ViTTiny)"""
        features = {}
        B = x.shape[0]
        
        x = self.patch_embed(x)
        features["patch_embed"] = x.clone()
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        features["pos_embed"] = x.clone()
        
        attn_weights = []
        for i, block in enumerate(self.blocks):
            x, attn = block(x)
            features[f"block_{i}"] = x.clone()
            attn_weights.append(attn)
        
        x_norm = self.norm(x)
        features["norm"] = x_norm.clone()
        features["cls_token_final"] = x_norm[:, 0].clone()
        
        output = self.head(x_norm[:, 0])
        features["output"] = output.clone()
        features["attention_weights"] = attn_weights
        
        return output, features


def design_model_scales():
    """
    Design ViT variants at different parameter scales
    
    Returns:
        Dict with model configurations for different scales
    """
    # Current ViTTiny: ~5.5M parameters
    configs = {
        "tiny": {
            "embed_dim": 192,
            "depth": 6,
            "num_heads": 3,
            "mlp_ratio": 4.0,
            "target_params": "~5M"
        },
        "small": {
            "embed_dim": 384,  # 2x embed dim
            "depth": 6,
            "num_heads": 6,    # Keep head_dim = 64
            "mlp_ratio": 4.0,
            "target_params": "~10M"
        },
        "medium": {
            "embed_dim": 512,
            "depth": 8,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "target_params": "~25M"
        },
        "large": {
            "embed_dim": 768,
            "depth": 8,
            "num_heads": 12,
            "mlp_ratio": 4.0,
            "target_params": "~50M"
        },
        "xl": {
            "embed_dim": 1024,
            "depth": 8,
            "num_heads": 16,
            "mlp_ratio": 4.0,
            "target_params": "~100M"
        }
    }
    
    return configs


def create_model(scale_name, config):
    """Create model for given scale configuration"""
    return ViTScaled(
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_ratio"]
    )


def train_model_at_scale(scale_name, model_config, optimizer_name, num_epochs=30, seed=42):
    """
    Train a model at given scale and return training metrics
    
    Returns:
        Dict with training results and model info
    """
    print(f"\n{'='*60}")
    print(f"Training {scale_name.upper()} model with {optimizer_name.upper()}")
    print(f"Config: {model_config}")
    print(f"{'='*60}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    
    # Create model
    model = create_model(scale_name, model_config)
    total_params, param_breakdown = count_parameters(model)
    
    print(f"Model parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # Adjust batch size based on model size (to fit in memory)
    if total_params < 10e6:
        batch_size = 4096
    elif total_params < 50e6:
        batch_size = 2048
    else:
        batch_size = 1024
    
    print(f"Using batch size: {batch_size}")
    
    # Setup data
    config = {
        "data_dir": "data",
        "batch_size": batch_size,
        "num_workers": 4,
        "train_val_split": 0.8,
        "seed": seed,
        "log_dir": "temp",
    }
    
    train_loader, val_loader, _ = get_cifar10_dataloaders(config)
    
    # Create optimizer with scale-adjusted LR
    base_lrs = {"sgd": 0.1, "adamw": 0.001, "muon": 0.01}
    
    # Scale LR based on batch size (linear scaling rule)
    lr_scale = batch_size / 4096
    adjusted_lr = base_lrs[optimizer_name] * lr_scale
    
    optimizer_config = {
        "type": optimizer_name,
        "lr": adjusted_lr,
        "weight_decay": 0.1 if optimizer_name == "muon" else 0.0001,
    }
    
    if optimizer_name == "sgd":
        optimizer_config["momentum"] = 0.9
    elif optimizer_name == "adamw":
        optimizer_config["betas"] = [0.9, 0.999]
    elif optimizer_name == "muon":
        optimizer_config["momentum"] = 0.95
        optimizer_config["nesterov"] = True
    
    print(f"Optimizer config: {optimizer_config}")
    
    # Create optimizer
    model = model.to(device)
    optimizer = get_optimizer(model, {"optimizer": optimizer_config})
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [], 
        "val_acc": [],
        "epochs": []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in tqdm(range(num_epochs), desc=f"Training {scale_name}"):
        # Training
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
        
        # Store metrics
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epochs"].append(epoch)
        
        # Early stopping for unstable training
        if train_loss > 10.0 or np.isnan(train_loss):
            print(f"Training became unstable at epoch {epoch}")
            break
        
        # Progress update
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: train_acc={train_acc:.2f}%, val_acc={val_acc:.2f}%")
    
    # Final results
    results = {
        "scale_name": scale_name,
        "optimizer": optimizer_name,
        "model_config": model_config,
        "total_params": total_params,
        "param_breakdown": param_breakdown,
        "batch_size": batch_size,
        "adjusted_lr": adjusted_lr,
        "final_train_acc": history["train_acc"][-1] if history["train_acc"] else 0.0,
        "final_val_acc": history["val_acc"][-1] if history["val_acc"] else 0.0,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "epochs_completed": len(history["epochs"]),
        "training_stable": not (np.isnan(history["train_loss"][-1]) if history["train_loss"] else True),
        "history": history
    }
    
    print(f"✓ {scale_name} training complete:")
    print(f"  Final val acc: {results['final_val_acc']:.2f}%")
    print(f"  Best val acc: {results['best_val_acc']:.2f}% (epoch {best_epoch})")
    print(f"  Parameters: {total_params:,}")
    
    return results


def plot_scale_sweep_results(all_results, output_dir):
    """
    Create visualizations showing how performance scales with model size
    """
    plt.style.use("seaborn-v0_8")
    sns.set_palette("Set2")
    
    # Convert to DataFrame
    data = []
    for result in all_results:
        if result["training_stable"]:
            data.append({
                "scale": result["scale_name"],
                "optimizer": result["optimizer"],
                "params_millions": result["total_params"] / 1e6,
                "final_val_acc": result["final_val_acc"],
                "best_val_acc": result["best_val_acc"],
                "batch_size": result["batch_size"],
                "adjusted_lr": result["adjusted_lr"]
            })
    
    if not data:
        print("No stable results to plot")
        return
    
    df = pd.DataFrame(data)
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Final validation accuracy vs model size
    ax1 = axes[0]
    for optimizer in df["optimizer"].unique():
        opt_data = df[df["optimizer"] == optimizer].sort_values("params_millions")
        ax1.plot(opt_data["params_millions"], opt_data["final_val_acc"],
                'o-', label=f"{optimizer.upper()}", linewidth=2, markersize=8)
    
    ax1.set_xscale("log")
    ax1.set_xlabel("Model Parameters (Millions)")
    ax1.set_ylabel("Final Validation Accuracy (%)")
    ax1.set_title("Model Performance vs Scale")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Best validation accuracy vs model size
    ax2 = axes[1]
    for optimizer in df["optimizer"].unique():
        opt_data = df[df["optimizer"] == optimizer].sort_values("params_millions")
        ax2.plot(opt_data["params_millions"], opt_data["best_val_acc"],
                'o-', label=f"{optimizer.upper()}", linewidth=2, markersize=8)
    
    ax2.set_xscale("log")
    ax2.set_xlabel("Model Parameters (Millions)")
    ax2.set_ylabel("Best Validation Accuracy (%)")
    ax2.set_title("Best Performance vs Scale")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "model_scale_sweep.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Print summary table
    print("\n" + "="*80)
    print("MODEL SCALE SWEEP SUMMARY")
    print("="*80)
    
    for optimizer in df["optimizer"].unique():
        print(f"\n{optimizer.upper()}:")
        opt_data = df[df["optimizer"] == optimizer].sort_values("params_millions")
        for _, row in opt_data.iterrows():
            print(f"  {row['scale']:<8} ({row['params_millions']:5.1f}M params): "
                  f"{row['best_val_acc']:5.2f}% val acc")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Model Scale Sweep")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adamw", "muon", "all"], 
                       default="all", help="Optimizer to test (default: all)")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer scales and epochs")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train (default: 30)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"model_scale_sweep_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    print("Model Scale Sweep Analysis")
    print(f"Output directory: {output_dir}")
    print(f"Quick mode: {args.quick}")
    print(f"Epochs per scale: {args.epochs}")
    
    # Get model configurations
    model_configs = design_model_scales()
    
    # Choose scales to test
    if args.quick:
        scales_to_test = ["tiny", "small", "medium"]  # 5M, 10M, 25M
        num_epochs = 15
    else:
        scales_to_test = list(model_configs.keys())  # All scales
        num_epochs = args.epochs
    
    # Choose optimizers
    if args.optimizer == "all":
        optimizers_to_test = ["sgd", "adamw", "muon"]
    else:
        optimizers_to_test = [args.optimizer]
    
    # Print planned experiments
    print("\nPlanned experiments:")
    for scale in scales_to_test:
        config = model_configs[scale]
        print(f"  {scale}: {config['target_params']} params")
    print(f"Optimizers: {optimizers_to_test}")
    
    # Run experiments
    all_results = []
    
    for scale_name in scales_to_test:
        model_config = model_configs[scale_name]
        
        for optimizer_name in optimizers_to_test:
            try:
                result = train_model_at_scale(
                    scale_name, model_config, optimizer_name, 
                    num_epochs, args.seed
                )
                all_results.append(result)
            except Exception as e:
                print(f"Error training {scale_name} with {optimizer_name}: {e}")
                continue
    
    # Save results
    results_file = output_dir / "scale_sweep_results.json"
    print(f"\nSaving results to {results_file}...")
    
    # Remove history to save space
    results_to_save = []
    for result in all_results:
        result_copy = result.copy()
        if "history" in result_copy:
            del result_copy["history"]
        if "param_breakdown" in result_copy:
            del result_copy["param_breakdown"]  # Too detailed for JSON
        results_to_save.append(result_copy)
    
    with open(results_file, "w") as f:
        json.dump(results_to_save, f, indent=2)
    
    # Generate visualizations
    print("Generating visualizations...")
    try:
        plot_scale_sweep_results(all_results, output_dir)
        print("✓ Visualizations complete!")
    except Exception as e:
        print(f"Warning: Could not generate visualizations: {e}")
    
    print("\n✓ Model scale sweep complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()