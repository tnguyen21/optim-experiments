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
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from models import ViTTiny
from data import get_cifar10_dataloaders


MEMORY_LIMIT = 10000  # For activation covariance computation
DEFAULT_NUM_BATCHES = 5  # Default number of batches for activation collection
SPARSITY_THRESHOLD = 0.01  # Threshold for sparsity computation
SVD_EPSILON = 1e-10  # Small epsilon to avoid log(0) in entropy computation


def get_optimizer_name(exp_name):
    """Extract optimizer name from experiment name"""
    return exp_name.split("_")[0]


def discover_trained_models(results_dir="experiments/results"):
    """
    Discover all trained models in the results directory

    Returns:
        List of dicts with model info: [{optimizer, lr, seed, model_path, metrics_path}, ...]
    """
    models = []
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return models

    # Find all experiment directories
    for exp_dir in results_path.glob("*"):
        if not exp_dir.is_dir():
            continue

        # Look for nested directory with same name (current structure)
        nested_dir = exp_dir / exp_dir.name
        if nested_dir.exists():
            model_path = nested_dir / "final_model.pt"
            metrics_path = nested_dir / "metrics.json"
        else:
            # Fallback to direct structure
            model_path = exp_dir / "final_model.pt"
            metrics_path = exp_dir / "metrics.json"

        if model_path.exists() and metrics_path.exists():
            # Parse experiment info from directory name
            exp_name = exp_dir.name
            parts = exp_name.split("_")

            if len(parts) >= 4:  # Expected format: optimizer_lr{value}_seed{value}_{timestamp}
                optimizer = parts[0]
                lr_str = parts[1]  # lr{value}
                seed_str = parts[2]  # seed{value}

                try:
                    lr = float(lr_str.replace("lr", ""))
                    seed = int(seed_str.replace("seed", ""))

                    models.append(
                        {
                            "optimizer": optimizer,
                            "lr": lr,
                            "seed": seed,
                            "exp_name": exp_name,
                            "model_path": str(model_path),
                            "metrics_path": str(metrics_path),
                        }
                    )
                except ValueError:
                    print(f"Could not parse experiment name: {exp_name}")

    print(f"Found {len(models)} trained models")
    for model in models:
        print(f"  {model['optimizer']} (lr={model['lr']}, seed={model['seed']})")

    return models


def load_model_and_metrics(model_info, device="cpu"):
    """
    Load a trained model and its metrics

    Args:
        model_info: Dict with model_path and metrics_path
        device: Device to load model on

    Returns:
        (model, metrics) tuple
    """
    # Load metrics
    with open(model_info["metrics_path"], "r") as f:
        metrics = json.load(f)

    # Create and load model
    model = ViTTiny(num_classes=10)
    state_dict = torch.load(model_info["model_path"], map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, metrics


def compute_effective_rank(matrix):
    """
    Compute effective rank via singular value entropy

    Args:
        matrix: 2D tensor (weight matrix)

    Returns:
        Effective rank (float)
    """
    if matrix.dim() != 2:
        raise ValueError(f"Expected 2D matrix, got {matrix.dim()}D")

    # Compute SVD
    U, S, V = torch.svd(matrix)

    # Normalize singular values to create probability distribution
    s_norm = S / S.sum()

    # Compute entropy (with small epsilon to avoid log(0))
    entropy = -(s_norm * torch.log(s_norm + SVD_EPSILON)).sum()

    # Effective rank is exp(entropy)
    return torch.exp(entropy).item()


def analyze_model_weights(model):
    """
    Analyze weight matrices of the model

    Returns:
        Dict with effective ranks and singular values for each layer
    """
    weight_analysis = {}

    for name, param in model.named_parameters():
        if param.dim() == 2:  # Only 2D weight matrices
            # Compute SVD once
            U, S, V = torch.svd(param.data)
            singular_values = S.detach().cpu().numpy()

            # Compute effective rank from singular values
            s_norm = S / S.sum()
            entropy = -(s_norm * torch.log(s_norm + SVD_EPSILON)).sum()
            eff_rank = torch.exp(entropy).item()

            weight_analysis[name] = {
                "effective_rank": eff_rank,
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

    # Register hooks on key modules
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

    # Remove hooks
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
        # Flatten all dimensions except the first (batch)
        flat_acts = acts.flatten(1)

        stats[name] = {
            "mean_magnitude": acts.abs().mean().item(),
            "sparsity": (acts.abs() < SPARSITY_THRESHOLD).float().mean().item(),
            "variance": acts.var().item(),
            "shape": list(acts.shape),
            "num_activations": acts.numel(),
        }

        # Compute effective rank of activation covariance if feasible
        if flat_acts.size(1) <= MEMORY_LIMIT:  # Avoid memory issues with very large matrices
            try:
                # Center the activations
                centered = flat_acts - flat_acts.mean(dim=0, keepdim=True)
                # Compute covariance matrix
                cov = torch.mm(centered.t(), centered) / (centered.size(0) - 1)
                # Effective rank of covariance
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

    # Set style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("Set2")

    weight_analysis = results["weight_analysis"]

    # Group by optimizer
    by_optimizer = defaultdict(list)
    for exp_name, analysis in weight_analysis.items():
        optimizer = get_optimizer_name(exp_name)
        by_optimizer[optimizer].append((exp_name, analysis))

    # Get common layer names across all models
    all_layer_names = set()
    for exp_name, analysis in weight_analysis.items():
        all_layer_names.update(analysis.keys())

    # Focus on key transformer layers
    key_layers = [name for name in all_layer_names if any(layer_type in name for layer_type in ["qkv", "proj", "fc1", "fc2", "head"])]

    # Create subplots for key layers
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, layer_name in enumerate(key_layers[:6]):  # Top 6 layers
        ax = axes[idx]

        for optimizer, models in by_optimizer.items():
            # Collect singular values for this layer across seeds
            all_sv = []
            for exp_name, analysis in models:
                if layer_name in analysis:
                    sv = np.array(analysis[layer_name]["singular_values"])
                    # Normalize and sort in descending order
                    sv_norm = sv / sv.sum()
                    sv_sorted = np.sort(sv_norm)[::-1]
                    all_sv.append(sv_sorted)

            if all_sv:
                # Average across seeds (pad shorter sequences with zeros)
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

    # Remove unused subplots
    for idx in range(len(key_layers), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(output_dir / "singular_value_spectra.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_effective_ranks(results, output_dir):
    """
    Plot effective ranks by layer and optimizer
    """

    weight_analysis = results["weight_analysis"]

    # Collect data for plotting
    data = []
    for exp_name, analysis in weight_analysis.items():
        optimizer = get_optimizer_name(exp_name)
        for layer_name, layer_data in analysis.items():
            data.append({"optimizer": optimizer, "layer": layer_name, "effective_rank": layer_data["effective_rank"], "exp_name": exp_name})

    if not data:
        print("No data for effective rank plots")
        return

    df = pd.DataFrame(data)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Box plot by optimizer
    sns.boxplot(data=df, x="optimizer", y="effective_rank", ax=ax1)
    ax1.set_title("Effective Rank Distribution by Optimizer")
    ax1.set_ylabel("Effective Rank")

    # Heatmap by layer and optimizer
    pivot_df = df.groupby(["layer", "optimizer"])["effective_rank"].mean().unstack()
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="viridis", ax=ax2)
    ax2.set_title("Mean Effective Rank by Layer and Optimizer")

    plt.tight_layout()
    plt.savefig(output_dir / "effective_ranks.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_activation_statistics(results, output_dir):
    """
    Plot activation statistics across optimizers
    """

    activation_analysis = results["activation_analysis"]

    # Collect data
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

    # Create subplots for different metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Mean magnitude
    sns.boxplot(data=df, x="optimizer", y="mean_magnitude", ax=axes[0])
    axes[0].set_title("Activation Mean Magnitude by Optimizer")
    axes[0].set_yscale("log")

    # Sparsity
    sns.boxplot(data=df, x="optimizer", y="sparsity", ax=axes[1])
    axes[1].set_title("Activation Sparsity by Optimizer")

    # Variance
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


def create_summary_table(results, output_dir):
    """
    Create a summary comparison table
    """

    # Aggregate results by optimizer
    summary = defaultdict(lambda: defaultdict(list))

    # Weight analysis summary
    for exp_name, analysis in results["weight_analysis"].items():
        optimizer = get_optimizer_name(exp_name)

        effective_ranks = [layer_data["effective_rank"] for layer_data in analysis.values()]
        summary[optimizer]["weight_effective_rank_mean"].append(np.mean(effective_ranks))
        summary[optimizer]["weight_effective_rank_std"].append(np.std(effective_ranks))

    # Activation analysis summary
    for exp_name, analysis in results["activation_analysis"].items():
        optimizer = get_optimizer_name(exp_name)

        magnitudes = [stats["mean_magnitude"] for stats in analysis.values()]
        sparsities = [stats["sparsity"] for stats in analysis.values()]
        variances = [stats["variance"] for stats in analysis.values()]

        summary[optimizer]["activation_magnitude_mean"].append(np.mean(magnitudes))
        summary[optimizer]["activation_sparsity_mean"].append(np.mean(sparsities))
        summary[optimizer]["activation_variance_mean"].append(np.mean(variances))

    # Create final summary
    final_summary = {}
    for optimizer, metrics in summary.items():
        final_summary[optimizer] = {}
        for metric, values in metrics.items():
            final_summary[optimizer][f"{metric}_across_seeds"] = {"mean": float(np.mean(values)), "std": float(np.std(values))}

    # Save summary
    with open(output_dir / "summary_table.json", "w") as f:
        json.dump(final_summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)

    for optimizer in sorted(final_summary.keys()):
        print(f"\n{optimizer.upper()}:")
        metrics = final_summary[optimizer]

        if "weight_effective_rank_mean_across_seeds" in metrics:
            eff_rank = metrics["weight_effective_rank_mean_across_seeds"]
            print(f"  Weight Effective Rank: {eff_rank['mean']:.3f} ± {eff_rank['std']:.3f}")

        if "activation_sparsity_mean_across_seeds" in metrics:
            sparsity = metrics["activation_sparsity_mean_across_seeds"]
            print(f"  Activation Sparsity: {sparsity['mean']:.3f} ± {sparsity['std']:.3f}")

        if "activation_variance_mean_across_seeds" in metrics:
            variance = metrics["activation_variance_mean_across_seeds"]
            print(f"  Activation Variance: {variance['mean']:.6f} ± {variance['std']:.6f}")


def main():
    """Main analysis function"""
    print("Starting model analysis...")

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path("analysis")
    output_dir.mkdir(exist_ok=True)

    # Discover trained models
    models = discover_trained_models()
    if not models:
        print("No trained models found!")
        return

    # Storage for all analysis results
    all_results = {"weight_analysis": {}, "activation_analysis": {}, "class_similarity": {}, "model_info": models}

    # Load test data for activation analysis (we'll use a consistent subset)
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

    # Analyze each model
    for model_info in tqdm(models, desc="Analyzing models"):
        exp_name = model_info["exp_name"]

        print(f"\nAnalyzing {exp_name}...")

        try:
            # Load model
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

    # Save results
    results_file = output_dir / "analysis_results.json"
    print(f"\nSaving results to {results_file}...")

    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nGenerating visualizations...")
    try:
        plot_singular_value_spectra(all_results, output_dir)
        plot_effective_ranks(all_results, output_dir)
        plot_activation_statistics(all_results, output_dir)
        plot_class_similarity_heatmaps(all_results, output_dir)
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
