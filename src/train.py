import torch
import torch.nn as nn
from tqdm import tqdm
import trackio


def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """
    Train for one epoch

    Returns: avg_loss, avg_acc
    """
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]")

    for batch_idx, (data, targets) in enumerate(pbar):
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

        pbar.set_postfix({"Loss": f"{running_loss / (batch_idx + 1):.4f}", "Acc": f"{100.0 * correct / total:.2f}%"})

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
        pbar = tqdm(val_loader, desc="Validation")

        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(device), targets.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({"Loss": f"{running_loss / (batch_idx + 1):.4f}", "Acc": f"{100.0 * correct / total:.2f}%"})

    avg_loss = running_loss / len(val_loader)
    avg_acc = 100.0 * correct / total

    return avg_loss, avg_acc


def train(model, train_loader, val_loader, test_loader, optimizer, config):
    """
    Main training loop

    Returns: final_metrics dict
    """
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    num_epochs = config["num_epochs"]
    eval_every = config["eval_every"]

    print(f"Training on device: {device}")
    print(f"Training for {num_epochs} epochs, evaluating every {eval_every} epochs")

    best_val_acc = 0.0
    final_metrics = {}

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)

        trackio.log({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc})

        if (epoch + 1) % eval_every == 0 or epoch == num_epochs - 1:
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            trackio.log({"epoch": epoch + 1, "val_loss": val_loss, "val_acc": val_acc})

            print(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
        else:
            print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

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
