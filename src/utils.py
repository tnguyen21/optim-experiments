import os
import random
import json
import torch
import numpy as np
import trackio


def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimizer(model, config):
    """Create optimizer based on config"""
    optimizer_config = config["optimizer"]
    optimizer_type = optimizer_config["type"].lower()

    if optimizer_type == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=optimizer_config["lr"],
            momentum=optimizer_config.get("momentum", 0.9),
            weight_decay=optimizer_config.get("weight_decay", 0.0001),
        )
    elif optimizer_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_config["lr"],
            betas=optimizer_config.get("betas", [0.9, 0.999]),
            weight_decay=optimizer_config.get("weight_decay", 0.0001),
        )
    elif optimizer_type == "muon":
        # For CNNs, Muon works best with 2D linear layers
        # We'll use a hybrid approach: Muon for 2D params, AdamW for others
        muon_params = []
        adamw_params = []

        for name, param in model.named_parameters():
            if param.dim() == 2:  # Exactly 2D parameters (linear layers)
                muon_params.append(param)
            else:  # Conv layers (4D), biases (1D), batch norm (1D)
                adamw_params.append(param)

        # Print parameter distribution for transparency
        muon_param_count = sum(p.numel() for p in muon_params)
        adamw_param_count = sum(p.numel() for p in adamw_params)
        total_params = muon_param_count + adamw_param_count

        print("Muon optimizer distribution:")
        print(f"  Muon (2D params): {muon_param_count:,} parameters ({100 * muon_param_count / total_params:.1f}%)")
        print(f"  AdamW (other params): {adamw_param_count:,} parameters ({100 * adamw_param_count / total_params:.1f}%)")

        if not muon_params:
            # Fallback to AdamW if no 2D parameters
            print("Warning: No 2D parameters found for Muon, using AdamW for all parameters")
            return torch.optim.AdamW(
                model.parameters(),
                lr=optimizer_config["lr"],
                weight_decay=optimizer_config.get("weight_decay", 0.1),
            )

        # Use a wrapper to handle mixed optimizers
        from torch.optim import Muon, AdamW

        optimizers = []
        if muon_params:
            muon_opt = Muon(
                muon_params,
                lr=optimizer_config["lr"],
                weight_decay=optimizer_config.get("weight_decay", 0.1),
                momentum=optimizer_config.get("momentum", 0.95),
                nesterov=optimizer_config.get("nesterov", True),
                adjust_lr_fn=optimizer_config.get("adjust_lr_fn", "original"),
            )
            optimizers.append(("muon", muon_opt))

        if adamw_params:
            adamw_opt = AdamW(
                adamw_params,
                lr=optimizer_config["lr"],
                weight_decay=optimizer_config.get("weight_decay", 0.1),
            )
            optimizers.append(("adamw", adamw_opt))

        # Create a simple wrapper class for mixed optimizers
        class MixedOptimizer:
            def __init__(self, optimizers):
                self.optimizers = optimizers

            def zero_grad(self):
                for _, opt in self.optimizers:
                    opt.zero_grad()

            def step(self):
                for _, opt in self.optimizers:
                    opt.step()

            def __repr__(self):
                return f"MixedOptimizer({dict(self.optimizers)})"

        return MixedOptimizer(optimizers)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def setup_logging(config):
    """Setup Trackio logging and create directories"""
    # Create log and results directories
    log_dir = config["log_dir"]
    results_dir = log_dir.replace("logs", "results")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Initialize trackio with project name and config
    project_name = f"vit-tiny-cifar10-{config['optimizer']['type']}"
    trackio.init(
        project=project_name,
        config={
            "optimizer": config["optimizer"]["type"],
            "learning_rate": config["optimizer"]["lr"],
            "batch_size": config["batch_size"],
            "num_epochs": config["num_epochs"],
            "seed": config["seed"],
            "model": config["model"],
        },
    )

    return results_dir


def save_final_model(model, filepath, metrics):
    """Save final model weights and summary metrics"""
    # Create directory if it doesn't exist
    os.makedirs(filepath, exist_ok=True)

    # Save model weights
    model_path = os.path.join(filepath, "final_model.pt")
    torch.save(model.state_dict(), model_path)

    # Save metrics
    metrics_path = os.path.join(filepath, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
