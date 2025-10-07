import time
import trackio
import torch
import os


def get_learning_rate(optimizer):
    """Return the learning rate for the first optimizer param group."""
    if hasattr(optimizer, "param_groups"):
        if optimizer.param_groups:
            return optimizer.param_groups[0].get("lr")
        return None

    if hasattr(optimizer, "optimizers"):
        # Mixed optimizer case; return the first available lr
        for _, opt in optimizer.optimizers:
            lr = get_learning_rate(opt)
            if lr is not None:
                return lr

    return None


def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """
    Train for one epoch

    Returns: avg_loss, avg_acc
    """
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / len(train_loader)
    avg_acc = 100.0 * correct / total

    return avg_loss, avg_acc


def validate(model, val_loader, criterion, device):
    """
    Validate the model

    Returns: avg_loss, avg_acc
    """
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / len(val_loader)
    avg_acc = 100.0 * correct / total

    return avg_loss, avg_acc


def train(model, train_loader, val_loader, test_loader, optimizer, config, checkpoint_dir):
    """
    Main training loop

    Returns: final_metrics dict
    """
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    num_epochs = config["num_epochs"]
    eval_every = config["eval_every"]

    print(f"Training on device: {device}")
    print(f"Training for {num_epochs} epochs, evaluating every {eval_every} epochs")

    best_val_acc = 0.0
    final_metrics = {}

    for epoch in range(num_epochs):
        epoch_start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
        )
        epoch_time_ms = (time.perf_counter() - epoch_start) * 1_000

        trackio.log({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc})

        current_lr = get_learning_rate(optimizer)
        lr_display = f"{current_lr:.6e}" if current_lr is not None else "n/a"
        print(f"epoch {epoch:3} | {train_loss=:.4f} | {train_acc=:.2f}% | lr {lr_display} | {epoch_time_ms:0.2f}ms")

        if (epoch + 1) % eval_every == 0 or epoch == num_epochs - 1:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            trackio.log({"epoch": epoch + 1, "val_loss": val_loss, "val_acc": val_acc})
            print(f"eval | {val_loss=:.4f} | {val_acc=:.2f}%")

            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved: {checkpoint_path}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc

    print("\nEvaluating on test set...")
    test_loss, test_acc = validate(model, test_loader, criterion, device)

    trackio.log({"epoch": num_epochs, "test_loss": test_loss, "test_acc": test_acc})

    print(f"Final Test Results: Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

    final_metrics = {
        "final_train_loss": train_loss,
        "final_train_acc": train_acc,
        "best_val_acc": best_val_acc,
        "final_test_loss": test_loss,
        "final_test_acc": test_acc,
        "total_epochs": num_epochs,
        "optimizer": config["optimizer"]["type"],
        "learning_rate": config["optimizer"]["lr"],
        "seed": config["seed"],
    }

    return final_metrics
