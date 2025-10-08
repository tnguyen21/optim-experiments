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

# Import scale functionality with graceful fallback
try:
    sys.path.append(os.path.dirname(__file__))
    from model_scale_sweep import ViTScaled, design_model_scales
    SCALE_SUPPORT = True
except ImportError:
    SCALE_SUPPORT = False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run ViT CIFAR-10 experiment with optional model scaling")

    parser.add_argument("--config", type=str, default="config/base_config.yaml", help="Path to config file")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adamw", "muon"], help="Optimizer type (overrides config)")
    parser.add_argument("--lr", type=float, help="Learning rate (overrides config)")
    parser.add_argument("--seed", type=int, help="Random seed (overrides config)")
    parser.add_argument("--experiment-name", type=str, help="Experiment name (overrides config)")
    
    # Add scale support
    if SCALE_SUPPORT:
        scale_choices = list(design_model_scales().keys())
        parser.add_argument("--scale", type=str, choices=scale_choices, 
                          help="Model scale (tiny/small/medium/large/xl). Auto-detected from config if not specified.")
        parser.add_argument("--scales", type=str, nargs="+", choices=scale_choices,
                          help="Run experiments for multiple scales sequentially (e.g., --scales tiny small medium)")
    else:
        parser.add_argument("--scale", type=str, help="Model scale (scale support not available)")
        parser.add_argument("--scales", type=str, nargs="+", help="Multiple scales (scale support not available)")

    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def detect_scale_from_config(config_path, config):
    """
    Auto-detect model scale from config file name or content
    
    Returns:
        Scale name (str) or None if not detected
    """
    if not SCALE_SUPPORT:
        return "tiny"
    
    # Try to detect from filename
    config_filename = os.path.basename(config_path).lower()
    scale_keywords = {
        "tiny": ["tiny"],
        "small": ["small"], 
        "medium": ["medium"],
        "large": ["large"],
        "xl": ["xl"]
    }
    
    for scale, keywords in scale_keywords.items():
        if any(keyword in config_filename for keyword in keywords):
            return scale
    
    # Try to detect from config content
    if "model_scale" in config:
        return config["model_scale"]
    
    # Check if specific scale configs are used
    if "optimal" in config_filename:
        # These are our optimizer-specific configs, default to tiny
        return "tiny"
    
    # Default fallback
    return "tiny"


def create_model_by_scale(scale_name):
    """Create model for given scale configuration"""
    if not SCALE_SUPPORT:
        from models import ViTTiny
        return ViTTiny(num_classes=10)
    
    configs = design_model_scales()
    if scale_name not in configs:
        raise ValueError(f"Unknown scale: {scale_name}. Available: {list(configs.keys())}")
    
    config = configs[scale_name]
    return ViTScaled(
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_ratio"]
    )


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
    
    # Handle scale override
    if hasattr(args, 'scale') and args.scale:
        config["model_scale"] = args.scale

    return config


def create_experiment_name(config):
    """Create a unique experiment name with timestamp and optional scale prefix"""
    optimizer = config["optimizer"]["type"]
    seed = config["seed"]
    lr = config["optimizer"]["lr"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Include scale if specified and not tiny (for backward compatibility)
    scale = config.get("model_scale", "tiny")
    if scale != "tiny" and SCALE_SUPPORT:
        return f"{scale}_{optimizer}_lr{lr}_seed{seed}_{timestamp}"
    else:
        return f"{optimizer}_lr{lr}_seed{seed}_{timestamp}"


def run_single_experiment(config, scale=None):
    """Run a single experiment with given config and optional scale override"""
    
    # Override scale if specified
    if scale:
        config["model_scale"] = scale
    
    experiment_name = create_experiment_name(config)
    config["log_dir"] = os.path.join(config["log_dir"], experiment_name)

    print(f"Starting experiment: {experiment_name}")
    print(f"Config: {config}")

    set_seed(config["seed"])

    print("Loading CIFAR-10 dataset...")
    train_loader, val_loader, test_loader = get_cifar10_dataloaders(config)

    # Create model based on scale
    scale = config.get("model_scale", "tiny")
    if SCALE_SUPPORT and scale != "tiny":
        print(f"Creating ViT-{scale.upper()} model...")
        model = create_model_by_scale(scale)
    else:
        print("Creating ViT-Tiny model...")
        model = ViTTiny(num_classes=config["num_classes"])

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total ({total_params/1e6:.1f}M), {trainable_params:,} trainable")
    print(f"Model scale: {scale}")

    optimizer = get_optimizer(model, config)
    print(f"Using optimizer: {optimizer}")

    results_dir = setup_logging(config)

    checkpoint_dir = os.path.join(config["log_dir"], "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("Starting training...")
    final_metrics = train(model, train_loader, val_loader, test_loader, optimizer, config, checkpoint_dir)

    results_path = os.path.join(results_dir, experiment_name)
    save_final_model(model, results_path, final_metrics)

    trackio.finish()

    print("\nExperiment completed successfully!")
    print(f"Results saved to: {results_path}")
    print(f"Final metrics: {final_metrics}")

    return final_metrics


def main():
    args = parse_args()

    config = load_config(args.config)
    
    # Check if multi-scale sweep is requested
    if hasattr(args, 'scales') and args.scales:
        print(f"Running multi-scale sweep for scales: {args.scales}")
        all_results = []
        
        for scale in args.scales:
            print(f"\n{'='*60}")
            print(f"Running experiment with scale: {scale}")
            print(f"{'='*60}")
            
            # Create a copy of config to avoid modifying original
            scale_config = config.copy()
            scale_config = override_config(scale_config, args)
            
            try:
                result = run_single_experiment(scale_config, scale)
                all_results.append({
                    "scale": scale,
                    "metrics": result,
                    "success": True
                })
            except Exception as e:
                print(f"Error running experiment with scale {scale}: {e}")
                all_results.append({
                    "scale": scale,
                    "error": str(e),
                    "success": False
                })
        
        print(f"\n{'='*60}")
        print("MULTI-SCALE SWEEP SUMMARY")
        print(f"{'='*60}")
        for result in all_results:
            if result["success"]:
                metrics = result["metrics"]
                print(f"{result['scale']:<10}: Final Test Acc: {metrics.get('final_test_acc', 'N/A'):.2f}%")
            else:
                print(f"{result['scale']:<10}: FAILED - {result['error']}")
        
        return all_results
    
    # Single experiment (original behavior)
    # Detect scale if not specified via CLI
    if not (hasattr(args, 'scale') and args.scale):
        detected_scale = detect_scale_from_config(args.config, config)
        config["model_scale"] = detected_scale
        print(f"Auto-detected model scale: {detected_scale}")
    
    config = override_config(config, args)
    
    return run_single_experiment(config)


if __name__ == "__main__":
    main()
