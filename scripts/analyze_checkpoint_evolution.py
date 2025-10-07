#!/usr/bin/env python3
"""
Checkpoint Evolution Analysis Script

Analyzes model evolution across training checkpoints to understand:
1. How weight matrix effective ranks evolve during training
2. How activation statistics change across epochs
3. Learning dynamics and phase transitions
4. Layer-wise evolution patterns

Usage:
python scripts/analyze_checkpoint_evolution.py
"""

import os
import sys
import json
import re
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from models import ViTTiny
from data import get_cifar10_dataloaders

# Import analysis functions from existing script
sys.path.append(os.path.dirname(__file__))
from summarize_optimizer_differences import (
    analyze_model_weights,
    collect_activations,
    compute_activation_stats,
    compute_class_representation_similarity,
    DEFAULT_NUM_BATCHES,
)


def discover_checkpoint_experiments(logs_dir="multiseed_sweep_experiments/logs"):
    """
    Discover all experiments with checkpoints

    Returns:
        List of experiment info dicts with checkpoint paths
    """
    experiments = []
    logs_path = Path(logs_dir)

    if not logs_path.exists():
        print(f"Logs directory not found: {logs_dir}")
        return experiments

    for exp_dir in logs_path.glob("*"):
        if not exp_dir.is_dir():
            continue

        checkpoints_dir = exp_dir / "checkpoints"
        if not checkpoints_dir.exists():
            continue

        # Parse experiment name
        exp_name = exp_dir.name
        parts = exp_name.split("_")

        if len(parts) >= 4:  # Expected format: optimizer_lr{value}_seed{value}_{timestamp}
            optimizer = parts[0]
            lr_str = parts[1]  # lr{value}
            seed_str = parts[2]  # seed{value}

            try:
                lr = float(lr_str.replace("lr", ""))
                seed = int(seed_str.replace("seed", ""))

                # Find all checkpoint files
                checkpoint_files = list(checkpoints_dir.glob("checkpoint_epoch_*.pt"))
                checkpoint_epochs = []

                for cp_file in checkpoint_files:
                    # Extract epoch number from filename
                    match = re.search(r"checkpoint_epoch_(\d+)\.pt", cp_file.name)
                    if match:
                        epoch = int(match.group(1))
                        checkpoint_epochs.append((epoch, str(cp_file)))

                if checkpoint_epochs:
                    # Sort by epoch
                    checkpoint_epochs.sort(key=lambda x: x[0])

                    experiments.append(
                        {
                            "optimizer": optimizer,
                            "lr": lr,
                            "seed": seed,
                            "exp_name": exp_name,
                            "exp_dir": str(exp_dir),
                            "checkpoints": checkpoint_epochs,  # List of (epoch, path) tuples
                        }
                    )

            except ValueError:
                print(f"Could not parse experiment name: {exp_name}")

    print(f"Found {len(experiments)} experiments with checkpoints")
    for exp in experiments:
        print(f"  {exp['exp_name']}: {len(exp['checkpoints'])} checkpoints")

    return experiments


def load_checkpoint_model(checkpoint_path, device="cpu"):
    """
    Load model from checkpoint file

    Args:
        checkpoint_path: Path to checkpoint .pt file
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    model = ViTTiny(num_classes=10)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            # Checkpoint contains training metadata
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            # Alternative format
            state_dict = checkpoint["state_dict"]
        else:
            # Assume the checkpoint is just the state dict
            state_dict = checkpoint
    else:
        # Checkpoint is directly the state dict
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def analyze_checkpoint_evolution(exp_info, dataloader, device, num_batches=DEFAULT_NUM_BATCHES):
    """
    Analyze evolution of a single experiment across its checkpoints

    Returns:
        Dict with temporal analysis data
    """
    exp_name = exp_info["exp_name"]
    checkpoints = exp_info["checkpoints"]

    evolution_data = {
        "exp_name": exp_name,
        "optimizer": exp_info["optimizer"],
        "lr": exp_info["lr"],
        "seed": exp_info["seed"],
        "epochs": [],
        "weight_analysis": {},
        "activation_analysis": {},
        "class_similarity": {},
    }

    print(f"\nAnalyzing evolution for {exp_name}...")

    for epoch, checkpoint_path in tqdm(checkpoints, desc="Processing checkpoints"):
        try:
            # Load model from checkpoint
            model = load_checkpoint_model(checkpoint_path, device)

            # Analyze weights
            weight_analysis = analyze_model_weights(model)

            # Analyze activations
            activations = collect_activations(model, dataloader, device, num_batches)
            activation_stats = compute_activation_stats(activations)

            # Analyze class representations
            class_stats = compute_class_representation_similarity(model, dataloader, device, num_batches)

            # Store results
            evolution_data["epochs"].append(epoch)
            evolution_data["weight_analysis"][epoch] = weight_analysis
            evolution_data["activation_analysis"][epoch] = activation_stats
            evolution_data["class_similarity"][epoch] = class_stats

        except Exception as e:
            print(f"  Error analyzing epoch {epoch}: {e}")
            continue

    return evolution_data


def compute_temporal_metrics(evolution_data):
    """
    Compute aggregated temporal metrics from evolution data

    Returns:
        Dict with time series data for plotting
    """
    epochs = sorted(evolution_data["epochs"])

    metrics = {
        "epochs": epochs,
        "weight_effective_ranks": {"mean": [], "std": [], "by_layer": defaultdict(list)},
        "activation_stats": {"mean_magnitude": [], "sparsity": [], "variance": []},
        "class_similarity_stats": {"mean_cosine_sim": [], "std_cosine_sim": []},
    }

    for epoch in epochs:
        # Weight analysis metrics
        if epoch in evolution_data["weight_analysis"]:
            weight_analysis = evolution_data["weight_analysis"][epoch]
            effective_ranks = []

            for layer_name, layer_data in weight_analysis.items():
                eff_rank = layer_data["effective_rank"]
                effective_ranks.append(eff_rank)
                metrics["weight_effective_ranks"]["by_layer"][layer_name].append(eff_rank)

            if effective_ranks:
                metrics["weight_effective_ranks"]["mean"].append(np.mean(effective_ranks))
                metrics["weight_effective_ranks"]["std"].append(np.std(effective_ranks))
            else:
                metrics["weight_effective_ranks"]["mean"].append(np.nan)
                metrics["weight_effective_ranks"]["std"].append(np.nan)

        # Activation analysis metrics
        if epoch in evolution_data["activation_analysis"]:
            activation_analysis = evolution_data["activation_analysis"][epoch]

            magnitudes = [stats["mean_magnitude"] for stats in activation_analysis.values()]
            sparsities = [stats["sparsity"] for stats in activation_analysis.values()]
            variances = [stats["variance"] for stats in activation_analysis.values()]

            metrics["activation_stats"]["mean_magnitude"].append(np.mean(magnitudes) if magnitudes else np.nan)
            metrics["activation_stats"]["sparsity"].append(np.mean(sparsities) if sparsities else np.nan)
            metrics["activation_stats"]["variance"].append(np.mean(variances) if variances else np.nan)

        # Class similarity metrics
        if epoch in evolution_data["class_similarity"]:
            class_stats = evolution_data["class_similarity"][epoch]
            if "cosine_similarity" in class_stats:
                cosine_sim_matrix = np.array(class_stats["cosine_similarity"])
                # Get off-diagonal elements (excluding self-similarity)
                mask = ~np.eye(cosine_sim_matrix.shape[0], dtype=bool)
                off_diag_sims = cosine_sim_matrix[mask]

                metrics["class_similarity_stats"]["mean_cosine_sim"].append(np.mean(off_diag_sims))
                metrics["class_similarity_stats"]["std_cosine_sim"].append(np.std(off_diag_sims))
            else:
                metrics["class_similarity_stats"]["mean_cosine_sim"].append(np.nan)
                metrics["class_similarity_stats"]["std_cosine_sim"].append(np.nan)

    return metrics


def plot_evolution_line_plots(all_evolution_data, output_dir):
    """
    Create line plots showing metric evolution over training epochs
    """
    plt.style.use("seaborn-v0_8")
    sns.set_palette("Set2")

    # Organize data by optimizer
    by_optimizer = defaultdict(list)
    for evolution_data in all_evolution_data:
        optimizer = evolution_data["optimizer"]
        metrics = compute_temporal_metrics(evolution_data)
        by_optimizer[optimizer].append((evolution_data["exp_name"], metrics))

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Plot 1: Effective Rank Evolution
    ax = axes[0]
    for optimizer, exp_data in by_optimizer.items():
        all_epochs = []
        all_ranks = []

        for exp_name, metrics in exp_data:
            epochs = metrics["epochs"]
            ranks = metrics["weight_effective_ranks"]["mean"]

            # Filter out NaN values
            valid_data = [(e, r) for e, r in zip(epochs, ranks) if not np.isnan(r)]
            if valid_data:
                epochs_clean, ranks_clean = zip(*valid_data)
                ax.plot(epochs_clean, ranks_clean, alpha=0.3, linewidth=1, color=plt.cm.Set2(list(by_optimizer.keys()).index(optimizer)))
                all_epochs.extend(epochs_clean)
                all_ranks.extend(ranks_clean)

        # Plot average trend if we have data
        if all_epochs and all_ranks:
            # Group by epoch and compute mean
            epoch_groups = defaultdict(list)
            for e, r in zip(all_epochs, all_ranks):
                epoch_groups[e].append(r)

            avg_epochs = sorted(epoch_groups.keys())
            avg_ranks = [np.mean(epoch_groups[e]) for e in avg_epochs]

            ax.plot(
                avg_epochs,
                avg_ranks,
                linewidth=3,
                label=f"{optimizer.upper()}",
                color=plt.cm.Set2(list(by_optimizer.keys()).index(optimizer)),
            )

    ax.set_title("Weight Effective Rank Evolution")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Effective Rank")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Activation Sparsity Evolution
    ax = axes[1]
    for optimizer, exp_data in by_optimizer.items():
        all_epochs = []
        all_sparsity = []

        for exp_name, metrics in exp_data:
            epochs = metrics["epochs"]
            sparsity = metrics["activation_stats"]["sparsity"]

            valid_data = [(e, s) for e, s in zip(epochs, sparsity) if not np.isnan(s)]
            if valid_data:
                epochs_clean, sparsity_clean = zip(*valid_data)
                ax.plot(epochs_clean, sparsity_clean, alpha=0.3, linewidth=1, color=plt.cm.Set2(list(by_optimizer.keys()).index(optimizer)))
                all_epochs.extend(epochs_clean)
                all_sparsity.extend(sparsity_clean)

        if all_epochs and all_sparsity:
            epoch_groups = defaultdict(list)
            for e, s in zip(all_epochs, all_sparsity):
                epoch_groups[e].append(s)

            avg_epochs = sorted(epoch_groups.keys())
            avg_sparsity = [np.mean(epoch_groups[e]) for e in avg_epochs]

            ax.plot(
                avg_epochs,
                avg_sparsity,
                linewidth=3,
                label=f"{optimizer.upper()}",
                color=plt.cm.Set2(list(by_optimizer.keys()).index(optimizer)),
            )

    ax.set_title("Activation Sparsity Evolution")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Sparsity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Activation Magnitude Evolution
    ax = axes[2]
    for optimizer, exp_data in by_optimizer.items():
        all_epochs = []
        all_magnitude = []

        for exp_name, metrics in exp_data:
            epochs = metrics["epochs"]
            magnitude = metrics["activation_stats"]["mean_magnitude"]

            valid_data = [(e, m) for e, m in zip(epochs, magnitude) if not np.isnan(m)]
            if valid_data:
                epochs_clean, magnitude_clean = zip(*valid_data)
                ax.plot(
                    epochs_clean, magnitude_clean, alpha=0.3, linewidth=1, color=plt.cm.Set2(list(by_optimizer.keys()).index(optimizer))
                )
                all_epochs.extend(epochs_clean)
                all_magnitude.extend(magnitude_clean)

        if all_epochs and all_magnitude:
            epoch_groups = defaultdict(list)
            for e, m in zip(all_epochs, all_magnitude):
                epoch_groups[e].append(m)

            avg_epochs = sorted(epoch_groups.keys())
            avg_magnitude = [np.mean(epoch_groups[e]) for e in avg_epochs]

            ax.plot(
                avg_epochs,
                avg_magnitude,
                linewidth=3,
                label=f"{optimizer.upper()}",
                color=plt.cm.Set2(list(by_optimizer.keys()).index(optimizer)),
            )

    ax.set_title("Activation Magnitude Evolution")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Magnitude")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Class Similarity Evolution
    ax = axes[3]
    for optimizer, exp_data in by_optimizer.items():
        all_epochs = []
        all_similarity = []

        for exp_name, metrics in exp_data:
            epochs = metrics["epochs"]
            similarity = metrics["class_similarity_stats"]["mean_cosine_sim"]

            valid_data = [(e, s) for e, s in zip(epochs, similarity) if not np.isnan(s)]
            if valid_data:
                epochs_clean, similarity_clean = zip(*valid_data)
                ax.plot(
                    epochs_clean, similarity_clean, alpha=0.3, linewidth=1, color=plt.cm.Set2(list(by_optimizer.keys()).index(optimizer))
                )
                all_epochs.extend(epochs_clean)
                all_similarity.extend(similarity_clean)

        if all_epochs and all_similarity:
            epoch_groups = defaultdict(list)
            for e, s in zip(all_epochs, all_similarity):
                epoch_groups[e].append(s)

            avg_epochs = sorted(epoch_groups.keys())
            avg_similarity = [np.mean(epoch_groups[e]) for e in avg_epochs]

            ax.plot(
                avg_epochs,
                avg_similarity,
                linewidth=3,
                label=f"{optimizer.upper()}",
                color=plt.cm.Set2(list(by_optimizer.keys()).index(optimizer)),
            )

    ax.set_title("Inter-Class Similarity Evolution")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "checkpoint_evolution.png", dpi=300, bbox_inches="tight")
    plt.close()


def detect_learning_phases(metrics, smoothing_window=5):
    """
    Detect learning phases based on metric evolution

    Returns:
        Dict with phase information and transition points
    """
    epochs = metrics["epochs"]
    if len(epochs) < 2:
        return {"phases": [], "transitions": []}

    # Use effective rank as primary signal for phase detection
    effective_ranks = metrics["weight_effective_ranks"]["mean"]

    # Remove NaN values
    valid_data = [(e, r) for e, r in zip(epochs, effective_ranks) if not np.isnan(r)]
    if len(valid_data) < smoothing_window:
        return {"phases": [], "transitions": []}

    epochs_clean, ranks_clean = zip(*valid_data)
    ranks_array = np.array(ranks_clean)

    # Apply smoothing
    if len(ranks_array) >= smoothing_window:
        ranks_smooth = np.convolve(ranks_array, np.ones(smoothing_window) / smoothing_window, mode="valid")
        epochs_smooth = epochs_clean[smoothing_window // 2 : -smoothing_window // 2 + 1] if smoothing_window > 1 else epochs_clean
    else:
        ranks_smooth = ranks_array
        epochs_smooth = epochs_clean

    # Detect transitions based on derivative changes
    if len(ranks_smooth) < 3:
        return {"phases": [{"phase": "single", "start": epochs_clean[0], "end": epochs_clean[-1]}], "transitions": []}

    # Compute first and second derivatives
    first_deriv = np.gradient(ranks_smooth)
    second_deriv = np.gradient(first_deriv)

    # Find significant transitions (peaks in absolute second derivative)
    transition_threshold = np.std(second_deriv) * 1.5
    transitions = []

    for i in range(1, len(second_deriv) - 1):
        if abs(second_deriv[i]) > transition_threshold and (
            (second_deriv[i - 1] < second_deriv[i] > second_deriv[i + 1]) or (second_deriv[i - 1] > second_deriv[i] < second_deriv[i + 1])
        ):
            transitions.append(epochs_smooth[i])

    # Define phases based on transitions
    phases = []
    if not transitions:
        # No clear transitions found
        if first_deriv[-1] < -0.001:  # Still decreasing significantly
            phases.append({"phase": "active_learning", "start": epochs_clean[0], "end": epochs_clean[-1]})
        else:
            phases.append({"phase": "convergence", "start": epochs_clean[0], "end": epochs_clean[-1]})
    else:
        # Add initial phase
        if first_deriv[0] < -0.001:
            phases.append({"phase": "active_learning", "start": epochs_clean[0], "end": transitions[0]})
        else:
            phases.append({"phase": "initialization", "start": epochs_clean[0], "end": transitions[0]})

        # Add intermediate phases
        for i in range(len(transitions) - 1):
            avg_deriv = np.mean(first_deriv[(np.array(epochs_smooth) >= transitions[i]) & (np.array(epochs_smooth) <= transitions[i + 1])])
            if avg_deriv < -0.001:
                phase_name = "active_learning"
            elif avg_deriv > 0.001:
                phase_name = "rank_increase"
            else:
                phase_name = "plateau"
            phases.append({"phase": phase_name, "start": transitions[i], "end": transitions[i + 1]})

        # Add final phase
        final_deriv = np.mean(first_deriv[-max(1, len(first_deriv) // 4) :])
        if final_deriv < -0.001:
            phase_name = "active_learning"
        elif abs(final_deriv) <= 0.001:
            phase_name = "convergence"
        else:
            phase_name = "overfitting"
        phases.append({"phase": phase_name, "start": transitions[-1], "end": epochs_clean[-1]})

    return {"phases": phases, "transitions": transitions}


def analyze_layer_dynamics(evolution_data):
    """
    Analyze layer-wise learning dynamics across checkpoints

    Returns:
        Dict with layer-specific evolution metrics
    """
    layer_dynamics = {}
    epochs = sorted(evolution_data["epochs"])

    if len(epochs) < 2:
        return layer_dynamics

    # Track effective rank changes for each layer
    all_layer_names = set()
    for epoch in epochs:
        if epoch in evolution_data["weight_analysis"]:
            all_layer_names.update(evolution_data["weight_analysis"][epoch].keys())

    for layer_name in all_layer_names:
        layer_ranks = []
        layer_epochs = []

        for epoch in epochs:
            if epoch in evolution_data["weight_analysis"] and layer_name in evolution_data["weight_analysis"][epoch]:
                rank = evolution_data["weight_analysis"][epoch][layer_name]["effective_rank"]
                layer_ranks.append(rank)
                layer_epochs.append(epoch)

        if len(layer_ranks) >= 2:
            # Compute learning rate (rate of rank change)
            rank_changes = np.diff(layer_ranks)
            epoch_diffs = np.diff(layer_epochs)
            learning_rates = rank_changes / epoch_diffs

            # Compute stability metrics
            rank_variance = np.var(layer_ranks)
            final_to_initial_ratio = layer_ranks[-1] / layer_ranks[0] if layer_ranks[0] != 0 else 1.0

            layer_dynamics[layer_name] = {
                "initial_rank": layer_ranks[0],
                "final_rank": layer_ranks[-1],
                "rank_change": layer_ranks[-1] - layer_ranks[0],
                "relative_change": final_to_initial_ratio - 1.0,
                "mean_learning_rate": np.mean(learning_rates),
                "learning_rate_std": np.std(learning_rates),
                "rank_variance": rank_variance,
                "epochs_tracked": len(layer_ranks),
            }

    return layer_dynamics


def plot_phase_analysis(all_evolution_data, output_dir):
    """
    Create visualizations showing learning phase analysis
    """
    plt.style.use("seaborn-v0_8")

    fig, axes = plt.subplots(len(all_evolution_data), 1, figsize=(12, 4 * len(all_evolution_data)), squeeze=False)

    for idx, evolution_data in enumerate(all_evolution_data):
        ax = axes[idx, 0]

        metrics = compute_temporal_metrics(evolution_data)
        phases_info = detect_learning_phases(metrics)

        epochs = metrics["epochs"]
        effective_ranks = metrics["weight_effective_ranks"]["mean"]

        # Plot effective rank evolution
        valid_data = [(e, r) for e, r in zip(epochs, effective_ranks) if not np.isnan(r)]
        if valid_data:
            epochs_clean, ranks_clean = zip(*valid_data)
            ax.plot(epochs_clean, ranks_clean, linewidth=2, label="Effective Rank", color="blue")

            # Highlight phases with different background colors
            phase_colors = {
                "initialization": "lightcyan",
                "active_learning": "lightgreen",
                "plateau": "lightyellow",
                "convergence": "lightblue",
                "overfitting": "lightcoral",
                "rank_increase": "lightpink",
                "single": "lightgray",
            }

            for phase in phases_info["phases"]:
                ax.axvspan(
                    phase["start"],
                    phase["end"],
                    alpha=0.3,
                    color=phase_colors.get(phase["phase"], "lightgray"),
                    label=f"{phase['phase'].replace('_', ' ').title()}",
                )

            # Mark transitions
            for transition in phases_info["transitions"]:
                ax.axvline(x=transition, color="red", linestyle="--", alpha=0.7, linewidth=1)

        ax.set_title(f"Learning Phases: {evolution_data['exp_name']}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean Effective Rank")
        ax.grid(True, alpha=0.3)

        # Only show legend for unique labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    plt.tight_layout()
    plt.savefig(output_dir / "learning_phases.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_layer_dynamics_heatmap(all_evolution_data, output_dir):
    """
    Create heatmap showing layer-wise dynamics across experiments
    """
    plt.style.use("seaborn-v0_8")

    # Collect layer dynamics data
    all_layer_data = []

    for evolution_data in all_evolution_data:
        layer_dynamics = analyze_layer_dynamics(evolution_data)

        for layer_name, dynamics in layer_dynamics.items():
            all_layer_data.append(
                {
                    "exp_name": evolution_data["exp_name"],
                    "optimizer": evolution_data["optimizer"],
                    "layer": layer_name,
                    "rank_change": dynamics["rank_change"],
                    "relative_change": dynamics["relative_change"],
                    "learning_rate": dynamics["mean_learning_rate"],
                    "stability": -dynamics["rank_variance"],  # Negative so higher is more stable
                }
            )

    if not all_layer_data:
        print("No layer dynamics data to plot")
        return

    df = pd.DataFrame(all_layer_data)

    # Create subplots for different metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics = [
        ("rank_change", "Absolute Rank Change"),
        ("relative_change", "Relative Rank Change"),
        ("learning_rate", "Mean Learning Rate"),
        ("stability", "Rank Stability (-variance)"),
    ]

    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        # Create pivot table for heatmap
        pivot_df = df.pivot_table(values=metric, index="layer", columns="optimizer", aggfunc="mean")

        if not pivot_df.empty:
            sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="RdBu_r", center=0, ax=ax)
            ax.set_title(title)
            ax.set_xlabel("Optimizer")
            ax.set_ylabel("Layer")

    plt.tight_layout()
    plt.savefig(output_dir / "layer_dynamics_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Main analysis function"""
    print("Starting checkpoint evolution analysis...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path("checkpoint_analysis")
    output_dir.mkdir(exist_ok=True)

    # Discover experiments
    experiments = discover_checkpoint_experiments()
    if not experiments:
        print("No experiments with checkpoints found!")
        return

    # Load test dataset
    print("Loading test dataset...")
    config = {
        "data_dir": "data",
        "batch_size": 32,
        "num_workers": 0,
        "train_val_split": 0.8,
        "seed": 42,
        "log_dir": "temp",
    }
    _, _, test_loader = get_cifar10_dataloaders(config)

    # Analyze evolution for each experiment
    all_evolution_data = []

    for exp_info in experiments:
        try:
            evolution_data = analyze_checkpoint_evolution(exp_info, test_loader, device)
            all_evolution_data.append(evolution_data)
        except Exception as e:
            print(f"Error analyzing {exp_info['exp_name']}: {e}")
            continue

    # Save results
    results_file = output_dir / "evolution_results.json"
    print(f"\nSaving results to {results_file}...")

    # Convert to JSON-serializable format
    json_data = []
    for evolution_data in all_evolution_data:
        json_data.append(
            {
                "exp_name": evolution_data["exp_name"],
                "optimizer": evolution_data["optimizer"],
                "lr": evolution_data["lr"],
                "seed": evolution_data["seed"],
                "epochs": evolution_data["epochs"],
                # Note: Skipping detailed analysis data to avoid huge JSON files
                # Could add option to save full data if needed
            }
        )

    with open(results_file, "w") as f:
        json.dump(json_data, f, indent=2)

    # Generate visualizations
    print("Generating visualizations...")
    try:
        plot_evolution_line_plots(all_evolution_data, output_dir)
        plot_phase_analysis(all_evolution_data, output_dir)
        plot_layer_dynamics_heatmap(all_evolution_data, output_dir)
        print("✓ Visualizations complete!")
    except Exception as e:
        print(f"Warning: Could not generate visualizations: {e}")

    print("✓ Checkpoint evolution analysis complete!")
    print(f"Results saved to: {output_dir}")

    return all_evolution_data


if __name__ == "__main__":
    results = main()
