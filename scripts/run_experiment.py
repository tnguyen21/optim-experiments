#!/usr/bin/env python3
"""
Main entry point for running ResNet-8 CIFAR-10 experiments

Usage:
python scripts/run_experiment.py --config config/base_config.yaml --optimizer sgd
python scripts/run_experiment.py --config config/base_config.yaml --optimizer adamw
"""

import argparse
import os
import sys
import yaml
import trackio
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from models import ViTTiny
from data import get_cifar10_dataloaders
from train import train
from utils import set_seed, get_optimizer, setup_logging, save_final_model


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run ResNet-8 CIFAR-10 experiment")

    parser.add_argument("--config", type=str, default="config/base_config.yaml", help="Path to config file")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adamw", "muon"], help="Optimizer type (overrides config)")
    parser.add_argument("--lr", type=float, help="Learning rate (overrides config)")
    parser.add_argument("--seed", type=int, help="Random seed (overrides config)")
    parser.add_argument("--experiment-name", type=str, help="Experiment name (overrides config)")

    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def override_config(config, args):
    """Override config values with command line arguments"""
    if args.optimizer:
        config["optimizer"]["type"] = args.optimizer

        # Set optimizer-specific defaults
        if args.optimizer == "adamw":
            config["optimizer"]["lr"] = args.lr if args.lr else 0.001
            config["optimizer"]["betas"] = [0.9, 0.999]
            config["optimizer"]["weight_decay"] = 0.0001
            # Remove SGD/Muon-specific params
            if "momentum" in config["optimizer"]:
                del config["optimizer"]["momentum"]
            if "nesterov" in config["optimizer"]:
                del config["optimizer"]["nesterov"]
            if "adjust_lr_fn" in config["optimizer"]:
                del config["optimizer"]["adjust_lr_fn"]
        elif args.optimizer == "sgd":
            config["optimizer"]["lr"] = args.lr if args.lr else 0.1
            config["optimizer"]["momentum"] = 0.9
            config["optimizer"]["weight_decay"] = 0.0001
            # Remove AdamW/Muon-specific params
            if "betas" in config["optimizer"]:
                del config["optimizer"]["betas"]
            if "nesterov" in config["optimizer"]:
                del config["optimizer"]["nesterov"]
            if "adjust_lr_fn" in config["optimizer"]:
                del config["optimizer"]["adjust_lr_fn"]
        elif args.optimizer == "muon":
            config["optimizer"]["lr"] = args.lr if args.lr else 0.02  # Higher LR for Muon
            config["optimizer"]["momentum"] = 0.95
            config["optimizer"]["weight_decay"] = 0.1  # Higher weight decay for Muon
            config["optimizer"]["nesterov"] = True
            config["optimizer"]["adjust_lr_fn"] = "original"
            # Remove AdamW-specific params
            if "betas" in config["optimizer"]:
                del config["optimizer"]["betas"]

    if args.lr:
        config["optimizer"]["lr"] = args.lr

    if args.seed:
        config["seed"] = args.seed

    if args.experiment_name:
        config["experiment_name"] = args.experiment_name

    return config


def create_experiment_name(config):
    """Create a unique experiment name with timestamp"""
    optimizer = config["optimizer"]["type"]
    seed = config["seed"]
    lr = config["optimizer"]["lr"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"{optimizer}_lr{lr}_seed{seed}_{timestamp}"


def main():
    args = parse_args()

    config = load_config(args.config)
    config = override_config(config, args)

    experiment_name = create_experiment_name(config)

    config["log_dir"] = os.path.join(config["log_dir"], experiment_name)

    print(f"Starting experiment: {experiment_name}")
    print(f"Config: {config}")

    set_seed(config["seed"])

    print("Loading CIFAR-10 dataset...")
    train_loader, val_loader, test_loader = get_cifar10_dataloaders(config)

    print("Creating ViT-Tiny model...")
    model = ViTTiny(num_classes=config["num_classes"])

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    optimizer = get_optimizer(model, config)
    print(f"Using optimizer: {optimizer}")

    results_dir = setup_logging(config)

    print("Starting training...")
    final_metrics = train(model, train_loader, val_loader, test_loader, optimizer, config)

    results_path = os.path.join(results_dir, experiment_name)
    save_final_model(model, results_path, final_metrics)

    trackio.finish()

    print("\nExperiment completed successfully!")
    print(f"Results saved to: {results_path}")
    print(f"Final metrics: {final_metrics}")

    return final_metrics


if __name__ == "__main__":
    main()
