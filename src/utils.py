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
    project_name = f"resnet8-cifar10-{config['optimizer']['type']}"
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
