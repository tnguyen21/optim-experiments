#!/usr/bin/env python3
"""
Model Analysis Script for Optimizer Comparison

Analyzes trained ViT-Tiny models to compare:
1. Weight matrix effective ranks (via singular value entropy)
2. Activation statistics and patterns
3. Optimization behavior across different optimizers

Usage:
python scripts/analyze_models.py
"""

import os
import sys
import json
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from models import ViTTiny
from data import get_cifar10_dataloaders

# Import model scale functionality
try:
    from model_scale_sweep import ViTScaled, design_model_scales
    SCALE_SUPPORT = True
except ImportError:
    SCALE_SUPPORT = False
    print("Note: Model scale analysis not available (model_scale_sweep.py not found)")


MEMORY_LIMIT = 10000  # For activation covariance computation
DEFAULT_NUM_BATCHES = 5  # Default number of batches for activation collection
SPARSITY_THRESHOLD = 0.01  # Threshold for sparsity computation
SVD_EPSILON = 1e-10  # Small epsilon to avoid log(0) in entropy computation


def parse_experiment_name(exp_name):
    """
    Parse experiment name to extract scale, optimizer, lr, seed
    
    Supports both formats:
    - Original: optimizer_lr{value}_seed{value}_{timestamp}
    - Scale-aware: {scale}_{optimizer}_lr{value}_seed{value}_{timestamp}
    
    Returns:
        Dict with parsed components
    """
    parts = exp_name.split("_")
    
    # Try scale-aware format first
    if len(parts) >= 5 and SCALE_SUPPORT:
        try:
            scale = parts[0]
            optimizer = parts[1]
            lr_str = parts[2]
            seed_str = parts[3]
            
            # Validate it's actually a scale name
            model_configs = design_model_scales()
            if scale.lower() in model_configs:
                lr = float(lr_str.replace("lr", ""))
                seed = int(seed_str.replace("seed", ""))
                return {
                    "scale": scale.lower(),
                    "optimizer": optimizer,
                    "lr": lr,
                    "seed": seed,
                    "has_scale": True
                }
        except (ValueError, IndexError):
            pass
    
    # Fall back to original format
    if len(parts) >= 4:
        try:
            optimizer = parts[0]
            lr_str = parts[1]
            seed_str = parts[2]
            
            lr = float(lr_str.replace("lr", ""))
            seed = int(seed_str.replace("seed", ""))
            return {
                "scale": "tiny",  # Default for original format
                "optimizer": optimizer,
                "lr": lr,
                "seed": seed,
                "has_scale": False
            }
        except (ValueError, IndexError):
            pass
    
    # Couldn't parse
    return None


def get_optimizer_name(exp_name):
    """Extract optimizer name from experiment name (backward compatibility)"""
    parsed = parse_experiment_name(exp_name)
    return parsed["optimizer"] if parsed else exp_name.split("_")[0]


def discover_trained_models(results_dir="experiments/results"):
    """
    Discover all trained models in the results directory
    
    Now supports both original and scale-aware experiment formats.

    Returns:
        List of dicts with model info: [{scale, optimizer, lr, seed, model_path, metrics_path, has_scale}, ...]
    """
    models = []
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return models

    for exp_dir in results_path.glob("*"):
        if not exp_dir.is_dir():
            continue

        nested_dir = exp_dir / exp_dir.name
        if nested_dir.exists():
            model_path = nested_dir / "final_model.pt"
            metrics_path = nested_dir / "metrics.json"
        else:
            model_path = exp_dir / "final_model.pt"
            metrics_path = exp_dir / "metrics.json"

        if model_path.exists() and metrics_path.exists():
            exp_name = exp_dir.name
            
            # Parse experiment name
            parsed = parse_experiment_name(exp_name)
            if parsed:
                models.append({
                    "scale": parsed["scale"],
                    "optimizer": parsed["optimizer"],
                    "lr": parsed["lr"],
                    "seed": parsed["seed"],
                    "has_scale": parsed["has_scale"],
                    "exp_name": exp_name,
                    "model_path": str(model_path),
                    "metrics_path": str(metrics_path),
                })
            else:
                print(f"Could not parse experiment name: {exp_name}")

    # Group and summarize discoveries
    by_scale = defaultdict(lambda: defaultdict(list))
    for model in models:
        by_scale[model["scale"]][model["optimizer"]].append(model)
    
    print(f"Found {len(models)} trained models:")
    for scale, optimizers in by_scale.items():
        total_for_scale = sum(len(models) for models in optimizers.values())
        print(f"  {scale.upper()}: {total_for_scale} models")
        for optimizer, models_list in optimizers.items():
            seeds = [m["seed"] for m in models_list]
            print(f"    {optimizer}: {len(models_list)} models (seeds: {seeds})")

    return models


def create_model_by_scale(scale):
    """
    Create model instance based on scale name
    
    Args:
        scale: Scale name (tiny, small, medium, large, xl)
        
    Returns:
        Model instance
    """
    if scale == "tiny" or not SCALE_SUPPORT:
        # Use original ViTTiny for backward compatibility
        return ViTTiny(num_classes=10)
    
    # Use scaled model
    model_configs = design_model_scales()
    if scale not in model_configs:
        print(f"Warning: Unknown scale '{scale}', falling back to tiny")
        return ViTTiny(num_classes=10)
    
    config = model_configs[scale]
    return ViTScaled(
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_ratio"],
        num_classes=10
    )


def load_model_and_metrics(model_info, device="cpu"):
    """
    Load a trained model and its metrics
    
    Now supports dynamic model creation based on scale.

    Args:
        model_info: Dict with model_path, metrics_path, and scale
        device: Device to load model on

    Returns:
        (model, metrics) tuple
    """
    with open(model_info["metrics_path"], "r") as f:
        metrics = json.load(f)

    # Create model based on scale
    scale = model_info.get("scale", "tiny")
    model = create_model_by_scale(scale)
    
    # Load weights
    state_dict = torch.load(model_info["model_path"], map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, metrics


def compute_effective_rank(matrix, normalize=False):
    """
    Compute effective rank via singular value entropy
    
    Args:
        matrix: 2D tensor (weight matrix)
        normalize: If True, normalize by maximum possible rank

    Returns:
        Effective rank (float), optionally normalized by max possible rank
    """
    if matrix.dim() != 2:
        raise ValueError(f"Expected 2D matrix, got {matrix.dim()}D")

    U, S, V = torch.svd(matrix)

    s_norm = S / S.sum()

    entropy = -(s_norm * torch.log(s_norm + SVD_EPSILON)).sum()

    effective_rank = torch.exp(entropy).item()
    
    if normalize:
        max_possible_rank = min(matrix.shape[0], matrix.shape[1])
        return effective_rank / max_possible_rank
    
    return effective_rank


def analyze_model_weights(model):
    """
    Analyze weight matrices of the model

    Returns:
        Dict with effective ranks (absolute and relative) and singular values for each layer
    """
    weight_analysis = {}

    for name, param in model.named_parameters():
        if param.dim() == 2:  # Only 2D weight matrices
            U, S, V = torch.svd(param.data)
            singular_values = S.detach().cpu().numpy()

            # Compute absolute effective rank
            eff_rank_abs = compute_effective_rank(param.data, normalize=False)
            
            # Compute relative effective rank (normalized by max possible rank)
            eff_rank_rel = compute_effective_rank(param.data, normalize=True)
            
            # Compute max possible rank for reference
            max_possible_rank = min(param.shape[0], param.shape[1])

            weight_analysis[name] = {
                "effective_rank": eff_rank_abs,  # Keep original field for backward compatibility
                "effective_rank_absolute": eff_rank_abs,
                "effective_rank_relative": eff_rank_rel,
                "max_possible_rank": max_possible_rank,
                "singular_values": singular_values.tolist(),
                "shape": list(param.shape),
                "num_params": param.numel(),
            }

    return weight_analysis


def collect_activations(model, dataloader, device, num_batches=DEFAULT_NUM_BATCHES):
    """
    Collect intermediate activations for analysis

    Args:
        model: The model to analyze
        dataloader: DataLoader for input data
        device: Device to run on
        num_batches: Number of batches to process

    Returns:
        Dict of activations by layer name
    """
    activations = {}
    hooks = []

    def get_activation(name):
        def hook(module, input, output):
            # Store activation (detached to save memory)
            if isinstance(output, tuple):
                output = output[0]  # For attention layers that return (output, weights)
            activations[name] = output.detach().cpu()

        return hook

    for name, module in model.named_modules():
        if any(layer_type in name for layer_type in ["attn", "mlp", "norm"]):
            hooks.append(module.register_forward_hook(get_activation(name)))

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            images = images.to(device)
            _ = model(images)

    for hook in hooks:
        hook.remove()

    return activations


def compute_activation_stats(activations):
    """
    Compute statistics on collected activations

    Returns:
        Dict of statistics by layer
    """
    stats = {}

    for name, acts in activations.items():
        flat_acts = acts.flatten(1)

        stats[name] = {
            "mean_magnitude": acts.abs().mean().item(),
            "sparsity": (acts.abs() < SPARSITY_THRESHOLD).float().mean().item(),
            "variance": acts.var().item(),
            "shape": list(acts.shape),
            "num_activations": acts.numel(),
        }

        if flat_acts.size(1) <= MEMORY_LIMIT:  # Avoid memory issues with very large matrices
            try:
                centered = flat_acts - flat_acts.mean(dim=0, keepdim=True)
                cov = torch.mm(centered.t(), centered) / (centered.size(0) - 1)
                stats[name]["cov_effective_rank"] = compute_effective_rank(cov)
            except Exception as e:
                print(f"Could not compute covariance effective rank for {name}: {e}")
                stats[name]["cov_effective_rank"] = None
        else:
            stats[name]["cov_effective_rank"] = None

    return stats


def compute_class_representation_similarity(model, dataloader, device, num_batches=None, representation="cls_token"):
    """Compute class-level representation statistics for covariance plots"""

    model.eval()
    num_classes = model.head.out_features
    rep_dim = model.head.in_features

    class_sum = torch.zeros(num_classes, rep_dim, device=device)
    class_count = torch.zeros(num_classes, device=device)

    batches_processed = 0
    seen_classes = set()

    with torch.no_grad():
        for images, labels in dataloader:
            if num_batches is not None and batches_processed >= num_batches and len(seen_classes) == num_classes:
                break

            images = images.to(device)
            labels = labels.to(device)

            _, features = model.get_features(images)

            if representation == "cls_token":
                reps = features["cls_token_final"]
            elif representation == "mean_patch":
                patches = features["norm"][:, 1:]
                reps = patches.mean(dim=1)
            else:
                raise ValueError(f"Unsupported representation type: {representation}")

            class_sum.index_add_(0, labels, reps)
            class_count.index_add_(0, labels, torch.ones(labels.size(0), device=device))

            seen_classes.update(labels.tolist())
            batches_processed += 1

    nonzero_mask = class_count > 0
    if not nonzero_mask.all():
        missing = (~nonzero_mask).nonzero(as_tuple=False).flatten().tolist()
        print(f"Warning: Missing samples for classes {missing}; covariance stats may be incomplete")

    safe_counts = class_count.clone()
    safe_counts[~nonzero_mask] = 1.0

    class_means = class_sum / safe_counts.unsqueeze(1)
    class_means[~nonzero_mask] = 0.0

    cosine_sim = F.cosine_similarity(class_means.unsqueeze(1), class_means.unsqueeze(0), dim=2)

    try:
        covariance = torch.cov(class_means.T)
    except RuntimeError:
        covariance = torch.zeros(num_classes, num_classes, device=device)

    return {
        "representation": representation,
        "class_means": class_means.cpu().tolist(),
        "class_counts": [int(c.item()) for c in class_count.cpu()],
        "cosine_similarity": cosine_sim.cpu().tolist(),
        "covariance": covariance.cpu().tolist(),
    }


def plot_singular_value_spectra(results, output_dir):
    """
    Plot singular value spectra for key layers across optimizers
    """

    plt.style.use("seaborn-v0_8")
    sns.set_palette("Set2")

    weight_analysis = results["weight_analysis"]

    by_optimizer = defaultdict(list)
    for exp_name, analysis in weight_analysis.items():
        optimizer = get_optimizer_name(exp_name)
        by_optimizer[optimizer].append((exp_name, analysis))

    all_layer_names = set()
    for exp_name, analysis in weight_analysis.items():
        all_layer_names.update(analysis.keys())

    key_layers = [name for name in all_layer_names if any(layer_type in name for layer_type in ["qkv", "proj", "fc1", "fc2", "head"])]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, layer_name in enumerate(key_layers[:6]):  # Top 6 layers
        ax = axes[idx]

        for optimizer, models in by_optimizer.items():
            all_sv = []
            for exp_name, analysis in models:
                if layer_name in analysis:
                    sv = np.array(analysis[layer_name]["singular_values"])
                    sv_norm = sv / sv.sum()
                    sv_sorted = np.sort(sv_norm)[::-1]
                    all_sv.append(sv_sorted)

            if all_sv:
                max_len = max(len(sv) for sv in all_sv)
                padded = [np.pad(sv, (0, max_len - len(sv))) for sv in all_sv]
                mean_sv = np.mean(padded, axis=0)
                std_sv = np.std(padded, axis=0)

                x = np.arange(len(mean_sv))
                ax.plot(x, mean_sv, label=f"{optimizer}", linewidth=2)
                ax.fill_between(x, mean_sv - std_sv, mean_sv + std_sv, alpha=0.3)

        ax.set_title(f"Singular Value Spectrum: {layer_name}", fontsize=10)
        ax.set_xlabel("Singular Value Index")
        ax.set_ylabel("Normalized Singular Value")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

    for idx in range(len(key_layers), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(output_dir / "singular_value_spectra.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_effective_ranks(results, output_dir):
    """
    Plot effective ranks by layer and optimizer (both absolute and relative)
    """

    weight_analysis = results["weight_analysis"]

    data = []
    for exp_name, analysis in weight_analysis.items():
        optimizer = get_optimizer_name(exp_name)
        for layer_name, layer_data in analysis.items():
            data.append({
                "optimizer": optimizer, 
                "layer": layer_name, 
                "effective_rank": layer_data["effective_rank"],  # Absolute (backward compatibility)
                "effective_rank_relative": layer_data.get("effective_rank_relative", layer_data["effective_rank"]),  # Relative
                "max_possible_rank": layer_data.get("max_possible_rank", min(layer_data["shape"])),
                "exp_name": exp_name
            })

    if not data:
        print("No data for effective rank plots")
        return

    df = pd.DataFrame(data)

    # Create 2x2 subplot for absolute and relative ranks
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))

    # Absolute effective rank plots
    sns.boxplot(data=df, x="optimizer", y="effective_rank", ax=ax1)
    ax1.set_title("Absolute Effective Rank Distribution by Optimizer")
    ax1.set_ylabel("Effective Rank (Absolute)")

    pivot_df_abs = df.groupby(["layer", "optimizer"])["effective_rank"].mean().unstack()
    sns.heatmap(pivot_df_abs, annot=True, fmt=".2f", cmap="viridis", ax=ax2)
    ax2.set_title("Mean Absolute Effective Rank by Layer and Optimizer")

    # Relative effective rank plots (normalized)
    sns.boxplot(data=df, x="optimizer", y="effective_rank_relative", ax=ax3)
    ax3.set_title("Relative Effective Rank Distribution by Optimizer")
    ax3.set_ylabel("Effective Rank (Normalized)")
    ax3.set_ylim(0, 1)  # Relative ranks are between 0 and 1

    pivot_df_rel = df.groupby(["layer", "optimizer"])["effective_rank_relative"].mean().unstack()
    sns.heatmap(pivot_df_rel, annot=True, fmt=".3f", cmap="viridis", ax=ax4, vmin=0, vmax=1)
    ax4.set_title("Mean Relative Effective Rank by Layer and Optimizer")

    plt.tight_layout()
    plt.savefig(output_dir / "effective_ranks.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_activation_statistics(results, output_dir):
    """
    Plot activation statistics across optimizers
    """

    activation_analysis = results["activation_analysis"]

    data = []
    for exp_name, analysis in activation_analysis.items():
        optimizer = get_optimizer_name(exp_name)
        for layer_name, stats in analysis.items():
            data.append(
                {
                    "optimizer": optimizer,
                    "layer": layer_name,
                    "mean_magnitude": stats["mean_magnitude"],
                    "sparsity": stats["sparsity"],
                    "variance": stats["variance"],
                    "cov_effective_rank": stats.get("cov_effective_rank"),
                    "exp_name": exp_name,
                }
            )

    if not data:
        print("No data for activation plots")
        return

    df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    sns.boxplot(data=df, x="optimizer", y="mean_magnitude", ax=axes[0])
    axes[0].set_title("Activation Mean Magnitude by Optimizer")
    axes[0].set_yscale("log")

    sns.boxplot(data=df, x="optimizer", y="sparsity", ax=axes[1])
    axes[1].set_title("Activation Sparsity by Optimizer")

    sns.boxplot(data=df, x="optimizer", y="variance", ax=axes[2])
    axes[2].set_title("Activation Variance by Optimizer")
    axes[2].set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_dir / "activation_statistics.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_class_similarity_heatmaps(results, output_dir):
    """Plot class-level cosine similarity heatmaps across optimizers"""

    class_similarity = results.get("class_similarity", {})

    if not class_similarity:
        print("No class similarity data for covariance plots")
        return

    by_optimizer = defaultdict(list)
    for exp_name, stats in class_similarity.items():
        optimizer = get_optimizer_name(exp_name)

        if "cosine_similarity" in stats:
            by_optimizer[optimizer].append(np.array(stats["cosine_similarity"]))

    if not by_optimizer:
        print("No cosine similarity matrices found")
        return

    optimizers = sorted(by_optimizer.keys())
    fig, axes = plt.subplots(1, len(optimizers), figsize=(6 * len(optimizers), 5), squeeze=False)

    vmin, vmax = -1.0, 1.0

    for idx, optimizer in enumerate(optimizers):
        matrices = by_optimizer[optimizer]
        mean_matrix = np.mean(matrices, axis=0)

        ax = axes[0, idx]
        sns.heatmap(mean_matrix, ax=ax, cmap="coolwarm", vmin=vmin, vmax=vmax, square=True, cbar=idx == len(optimizers) - 1)
        ax.set_title(f"{optimizer.title()} Class Similarity")
        ax.set_xlabel("Class")
        ax.set_ylabel("Class")

    plt.tight_layout()
    plt.savefig(output_dir / "class_similarity_heatmaps.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_scale_vs_optimizer_heatmap(results, output_dir):
    """
    Create heatmaps showing metrics across (scale, optimizer) combinations
    """
    if not SCALE_SUPPORT:
        return
    
    plt.style.use("seaborn-v0_8")
    
    # Extract data with scale information
    data = []
    for exp_name, analysis in results["weight_analysis"].items():
        parsed = parse_experiment_name(exp_name)
        if parsed:
            effective_ranks_abs = [layer_data["effective_rank"] for layer_data in analysis.values()]
            effective_ranks_rel = [layer_data.get("effective_rank_relative", layer_data["effective_rank"]) for layer_data in analysis.values()]
            data.append({
                "scale": parsed["scale"],
                "optimizer": parsed["optimizer"],
                "exp_name": exp_name,
                "mean_effective_rank_absolute": np.mean(effective_ranks_abs),
                "mean_effective_rank_relative": np.mean(effective_ranks_rel)
            })
    
    # Add activation data
    for exp_name, analysis in results["activation_analysis"].items():
        parsed = parse_experiment_name(exp_name)
        if parsed:
            sparsities = [stats["sparsity"] for stats in analysis.values()]
            # Find corresponding entry
            for entry in data:
                if entry["exp_name"] == exp_name:
                    entry["mean_sparsity"] = np.mean(sparsities)
                    break
    
    if not data:
        print("No scale data available for heatmap")
        return
    
    df = pd.DataFrame(data)
    
    # Create subplots for different metrics (now 3 panels)
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    
    # Plot 1: Absolute Effective Rank Heatmap
    rank_abs_pivot = df.groupby(["scale", "optimizer"])["mean_effective_rank_absolute"].mean().unstack()
    if not rank_abs_pivot.empty:
        sns.heatmap(rank_abs_pivot, annot=True, fmt=".2f", cmap="viridis", ax=axes[0])
        axes[0].set_title("Mean Absolute Effective Rank by Scale and Optimizer")
        axes[0].set_xlabel("Optimizer")
        axes[0].set_ylabel("Model Scale")
    
    # Plot 2: Relative Effective Rank Heatmap
    rank_rel_pivot = df.groupby(["scale", "optimizer"])["mean_effective_rank_relative"].mean().unstack()
    if not rank_rel_pivot.empty:
        sns.heatmap(rank_rel_pivot, annot=True, fmt=".3f", cmap="viridis", ax=axes[1], vmin=0, vmax=1)
        axes[1].set_title("Mean Relative Effective Rank by Scale and Optimizer")
        axes[1].set_xlabel("Optimizer")
        axes[1].set_ylabel("Model Scale")
    
    # Plot 3: Sparsity Heatmap
    if "mean_sparsity" in df.columns:
        sparsity_pivot = df.groupby(["scale", "optimizer"])["mean_sparsity"].mean().unstack()
        if not sparsity_pivot.empty:
            sns.heatmap(sparsity_pivot, annot=True, fmt=".3f", cmap="plasma", ax=axes[2])
            axes[2].set_title("Mean Activation Sparsity by Scale and Optimizer")
            axes[2].set_xlabel("Optimizer")
            axes[2].set_ylabel("Model Scale")
    
    plt.tight_layout()
    plt.savefig(output_dir / "scale_optimizer_heatmaps.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_scaling_trends(results, output_dir):
    """
    Create line plots showing how metrics scale with model size
    """
    if not SCALE_SUPPORT:
        return
    
    plt.style.use("seaborn-v0_8")
    sns.set_palette("Set2")
    
    # Get model parameter counts
    model_configs = design_model_scales()
    scale_to_params = {}
    for scale, config in model_configs.items():
        model = create_model_by_scale(scale)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        scale_to_params[scale] = total_params
    
    # Extract and organize data
    by_optimizer = defaultdict(lambda: defaultdict(list))
    
    for exp_name, analysis in results["weight_analysis"].items():
        parsed = parse_experiment_name(exp_name)
        if parsed and parsed["scale"] in scale_to_params:
            effective_ranks_abs = [layer_data["effective_rank"] for layer_data in analysis.values()]
            effective_ranks_rel = [layer_data.get("effective_rank_relative", layer_data["effective_rank"]) for layer_data in analysis.values()]
            by_optimizer[parsed["optimizer"]]["params"].append(scale_to_params[parsed["scale"]])
            by_optimizer[parsed["optimizer"]]["effective_rank_absolute"].append(np.mean(effective_ranks_abs))
            by_optimizer[parsed["optimizer"]]["effective_rank_relative"].append(np.mean(effective_ranks_rel))
            by_optimizer[parsed["optimizer"]]["scale"].append(parsed["scale"])
    
    # Add activation data
    for exp_name, analysis in results["activation_analysis"].items():
        parsed = parse_experiment_name(exp_name)
        if parsed and parsed["scale"] in scale_to_params:
            sparsities = [stats["sparsity"] for stats in analysis.values()]
            # Find index for this experiment
            try:
                idx = len(by_optimizer[parsed["optimizer"]]["sparsity"])
                by_optimizer[parsed["optimizer"]]["sparsity"].append(np.mean(sparsities))
            except:
                pass
    
    if not by_optimizer:
        print("No scaling data available")
        return
    
    # Create plots (now 3 subplots)
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    
    # Plot 1: Absolute Effective Rank vs Parameters
    ax1 = axes[0]
    for optimizer, data in by_optimizer.items():
        if "effective_rank_absolute" in data and "params" in data:
            # Group by scale and average across seeds
            scale_data = defaultdict(list)
            for scale, rank in zip(data["scale"], data["effective_rank_absolute"]):
                scale_data[scale].append(rank)
            
            scales = sorted(scale_data.keys(), key=lambda s: scale_to_params[s])
            params = [scale_to_params[s] / 1e6 for s in scales]  # Convert to millions
            ranks = [np.mean(scale_data[s]) for s in scales]
            
            ax1.plot(params, ranks, 'o-', label=optimizer.upper(), linewidth=2, markersize=8)
    
    ax1.set_xscale("log")
    ax1.set_xlabel("Model Parameters (Millions)")
    ax1.set_ylabel("Mean Absolute Effective Rank")
    ax1.set_title("Absolute Effective Rank Scaling")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Relative Effective Rank vs Parameters
    ax2 = axes[1]
    for optimizer, data in by_optimizer.items():
        if "effective_rank_relative" in data and "params" in data:
            # Group by scale and average across seeds
            scale_data = defaultdict(list)
            for scale, rank in zip(data["scale"], data["effective_rank_relative"]):
                scale_data[scale].append(rank)
            
            scales = sorted(scale_data.keys(), key=lambda s: scale_to_params[s])
            params = [scale_to_params[s] / 1e6 for s in scales]
            ranks = [np.mean(scale_data[s]) for s in scales]
            
            ax2.plot(params, ranks, 'o-', label=optimizer.upper(), linewidth=2, markersize=8)
    
    ax2.set_xscale("log")
    ax2.set_xlabel("Model Parameters (Millions)")
    ax2.set_ylabel("Mean Relative Effective Rank")
    ax2.set_title("Relative Effective Rank Scaling")
    ax2.set_ylim(0, 1)  # Relative ranks are normalized
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Sparsity vs Parameters
    ax3 = axes[2]
    for optimizer, data in by_optimizer.items():
        if "sparsity" in data and "params" in data and len(data["sparsity"]) == len(data["scale"]):
            # Group by scale and average across seeds
            scale_data = defaultdict(list)
            for scale, sparsity in zip(data["scale"], data["sparsity"]):
                scale_data[scale].append(sparsity)
            
            scales = sorted(scale_data.keys(), key=lambda s: scale_to_params[s])
            params = [scale_to_params[s] / 1e6 for s in scales]
            sparsities = [np.mean(scale_data[s]) for s in scales]
            
            ax3.plot(params, sparsities, 'o-', label=optimizer.upper(), linewidth=2, markersize=8)
    
    ax3.set_xscale("log")
    ax3.set_xlabel("Model Parameters (Millions)")
    ax3.set_ylabel("Mean Activation Sparsity")
    ax3.set_title("Sparsity Scaling")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "scaling_trends.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_summary_table(results, output_dir):
    """
    Create a summary comparison table with scale awareness
    """

    # Organize data by (scale, optimizer) when possible, fallback to optimizer only
    summary = defaultdict(lambda: defaultdict(list))
    scale_aware_summary = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    has_scale_data = False

    for exp_name, analysis in results["weight_analysis"].items():
        parsed = parse_experiment_name(exp_name)
        if parsed:
            optimizer = parsed["optimizer"]
            scale = parsed["scale"]
            if parsed["has_scale"] and SCALE_SUPPORT:
                has_scale_data = True
            
            effective_ranks_abs = [layer_data["effective_rank"] for layer_data in analysis.values()]
            effective_ranks_rel = [layer_data.get("effective_rank_relative", layer_data["effective_rank"]) for layer_data in analysis.values()]
            
            # Original grouping (backward compatibility)
            summary[optimizer]["weight_effective_rank_mean"].append(np.mean(effective_ranks_abs))
            summary[optimizer]["weight_effective_rank_std"].append(np.std(effective_ranks_abs))
            summary[optimizer]["weight_effective_rank_relative_mean"].append(np.mean(effective_ranks_rel))
            summary[optimizer]["weight_effective_rank_relative_std"].append(np.std(effective_ranks_rel))
            
            # Scale-aware grouping
            if SCALE_SUPPORT:
                scale_aware_summary[scale][optimizer]["weight_effective_rank_mean"].append(np.mean(effective_ranks_abs))
                scale_aware_summary[scale][optimizer]["weight_effective_rank_std"].append(np.std(effective_ranks_abs))
                scale_aware_summary[scale][optimizer]["weight_effective_rank_relative_mean"].append(np.mean(effective_ranks_rel))
                scale_aware_summary[scale][optimizer]["weight_effective_rank_relative_std"].append(np.std(effective_ranks_rel))

    for exp_name, analysis in results["activation_analysis"].items():
        parsed = parse_experiment_name(exp_name)
        if parsed:
            optimizer = parsed["optimizer"]
            scale = parsed["scale"]

            magnitudes = [stats["mean_magnitude"] for stats in analysis.values()]
            sparsities = [stats["sparsity"] for stats in analysis.values()]
            variances = [stats["variance"] for stats in analysis.values()]

            # Original grouping
            summary[optimizer]["activation_magnitude_mean"].append(np.mean(magnitudes))
            summary[optimizer]["activation_sparsity_mean"].append(np.mean(sparsities))
            summary[optimizer]["activation_variance_mean"].append(np.mean(variances))
            
            # Scale-aware grouping
            if SCALE_SUPPORT:
                scale_aware_summary[scale][optimizer]["activation_magnitude_mean"].append(np.mean(magnitudes))
                scale_aware_summary[scale][optimizer]["activation_sparsity_mean"].append(np.mean(sparsities))
                scale_aware_summary[scale][optimizer]["activation_variance_mean"].append(np.mean(variances))

    # Create final summaries
    final_summary = {}
    for optimizer, metrics in summary.items():
        final_summary[optimizer] = {}
        for metric, values in metrics.items():
            final_summary[optimizer][f"{metric}_across_seeds"] = {"mean": float(np.mean(values)), "std": float(np.std(values))}

    # Scale-aware final summary
    scale_summary = {}
    if SCALE_SUPPORT:
        for scale, optimizers in scale_aware_summary.items():
            scale_summary[scale] = {}
            for optimizer, metrics in optimizers.items():
                scale_summary[scale][optimizer] = {}
                for metric, values in metrics.items():
                    scale_summary[scale][optimizer][f"{metric}_across_seeds"] = {
                        "mean": float(np.mean(values)), 
                        "std": float(np.std(values))
                    }

    # Save summaries
    with open(output_dir / "summary_table.json", "w") as f:
        json.dump(final_summary, f, indent=2)
    
    if scale_summary:
        with open(output_dir / "scale_summary_table.json", "w") as f:
            json.dump(scale_summary, f, indent=2)

    # Print results
    print("\n" + "=" * 80)
    print("OPTIMIZER COMPARISON SUMMARY")
    print("=" * 80)

    for optimizer in sorted(final_summary.keys()):
        print(f"\n{optimizer.upper()}:")
        metrics = final_summary[optimizer]

        if "weight_effective_rank_mean_across_seeds" in metrics:
            eff_rank = metrics["weight_effective_rank_mean_across_seeds"]
            print(f"  Weight Effective Rank (Absolute): {eff_rank['mean']:.3f} ± {eff_rank['std']:.3f}")

        if "weight_effective_rank_relative_mean_across_seeds" in metrics:
            eff_rank_rel = metrics["weight_effective_rank_relative_mean_across_seeds"]
            print(f"  Weight Effective Rank (Relative): {eff_rank_rel['mean']:.3f} ± {eff_rank_rel['std']:.3f}")

        if "activation_sparsity_mean_across_seeds" in metrics:
            sparsity = metrics["activation_sparsity_mean_across_seeds"]
            print(f"  Activation Sparsity: {sparsity['mean']:.3f} ± {sparsity['std']:.3f}")

        if "activation_variance_mean_across_seeds" in metrics:
            variance = metrics["activation_variance_mean_across_seeds"]
            print(f"  Activation Variance: {variance['mean']:.6f} ± {variance['std']:.6f}")

    # Print scale-aware summary
    if scale_summary and has_scale_data:
        print("\n" + "=" * 80)
        print("SCALE-AWARE ANALYSIS SUMMARY")
        print("=" * 80)
        
        for scale in sorted(scale_summary.keys()):
            print(f"\n{scale.upper()} MODELS:")
            for optimizer in sorted(scale_summary[scale].keys()):
                print(f"  {optimizer.upper()}:")
                metrics = scale_summary[scale][optimizer]
                
                if "weight_effective_rank_mean_across_seeds" in metrics:
                    eff_rank = metrics["weight_effective_rank_mean_across_seeds"]
                    print(f"    Effective Rank (Absolute): {eff_rank['mean']:.3f} ± {eff_rank['std']:.3f}")
                
                if "weight_effective_rank_relative_mean_across_seeds" in metrics:
                    eff_rank_rel = metrics["weight_effective_rank_relative_mean_across_seeds"]
                    print(f"    Effective Rank (Relative): {eff_rank_rel['mean']:.3f} ± {eff_rank_rel['std']:.3f}")
                
                if "activation_sparsity_mean_across_seeds" in metrics:
                    sparsity = metrics["activation_sparsity_mean_across_seeds"]
                    print(f"    Sparsity: {sparsity['mean']:.3f} ± {sparsity['std']:.3f}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Analyze trained models to compare optimizers and model scales",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze models in default experiments/results directory
  python scripts/summarize_optimizer_differences.py
  
  # Analyze models from model scale sweep
  python scripts/summarize_optimizer_differences.py --experiments model_scale_sweep_20251008_123456
  
  # Analyze models with custom output directory
  python scripts/summarize_optimizer_differences.py --experiments experiments/results --output custom_analysis
        """
    )
    
    parser.add_argument(
        "--experiments", 
        type=str, 
        default="experiments/results",
        help="Directory containing trained model results (default: experiments/results)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="analysis",
        help="Output directory for analysis results (default: analysis)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for analysis (default: auto)"
    )
    
    return parser.parse_args()


def main():
    """Main analysis function"""
    args = parse_args()
    
    print("Starting model analysis...")
    print(f"Experiments directory: {args.experiments}")
    print(f"Output directory: {args.output}")

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    models = discover_trained_models(args.experiments)
    if not models:
        print("No trained models found!")
        return

    all_results = {"weight_analysis": {}, "activation_analysis": {}, "class_similarity": {}, "model_info": models}

    print("Loading test dataset...")
    config = {
        "data_dir": "data",
        "batch_size": 32,
        "num_workers": 0,  # Avoid multiprocessing issues
        "train_val_split": 0.8,
        "seed": 42,
        "log_dir": "temp",  # Won't be used
    }
    _, _, test_loader = get_cifar10_dataloaders(config)

    for model_info in tqdm(models, desc="Analyzing models"):
        exp_name = model_info["exp_name"]

        print(f"\nAnalyzing {exp_name}...")

        try:
            model, metrics = load_model_and_metrics(model_info, device)

            print("  Computing weight matrix effective ranks...")
            weight_analysis = analyze_model_weights(model)
            all_results["weight_analysis"][exp_name] = weight_analysis

            print("  Collecting and analyzing activations...")
            activations = collect_activations(model, test_loader, device)
            activation_stats = compute_activation_stats(activations)
            all_results["activation_analysis"][exp_name] = activation_stats

            print("  Computing class covariance statistics...")
            class_stats = compute_class_representation_similarity(model, test_loader, device)
            all_results["class_similarity"][exp_name] = class_stats

            print(f"  ✓ Analysis complete for {exp_name}")

        except Exception as e:
            print(f"  ✗ Error analyzing {exp_name}: {e}")
            continue

    results_file = output_dir / "analysis_results.json"
    print(f"\nSaving results to {results_file}...")

    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nGenerating visualizations...")
    try:
        # Original visualizations
        plot_singular_value_spectra(all_results, output_dir)
        plot_effective_ranks(all_results, output_dir)
        plot_activation_statistics(all_results, output_dir)
        plot_class_similarity_heatmaps(all_results, output_dir)
        
        # New scale-aware visualizations
        if SCALE_SUPPORT:
            print("  Generating scale analysis plots...")
            plot_scale_vs_optimizer_heatmap(all_results, output_dir)
            plot_scaling_trends(all_results, output_dir)
        
        # Summary tables (both original and scale-aware)
        create_summary_table(all_results, output_dir)
        
        print("✓ Visualizations complete!")
    except Exception as e:
        print(f"Warning: Could not generate some visualizations: {e}")

    print("✓ Analysis complete!")
    print(f"Results saved to: {results_file}")
    print(f"Visualizations saved to: {output_dir}")

    return all_results


if __name__ == "__main__":
    results = main()
