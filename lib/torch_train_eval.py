import torch
from torch import nn
import numpy as np
from tqdm.auto import tqdm

import time
import pickle
import os


class EpochResults:
    def __init__(self, train_loss, train_acc, val_loss, val_acc) -> None:
        self.train_loss = train_loss
        self.train_acc = train_acc
        self.val_loss = val_loss
        self.val_acc = val_acc


def train_model(
    model: nn.Module,
    criterion,
    optimizer,
    scheduler,
    device: str,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    output_dir: str,
    num_epochs: int = 25,
    patience: int = 1,
    warmup_period: int = 10,
    previous_history: dict[str, list[float]] = None,
    gradient_accumulation: int = 1,
    train_stats_period: int = -1,
) -> tuple[nn.Module, dict[str, np.ndarray]]:
    dataloaders = {"train": train_dataloader, "val": val_dataloader}

    output_model_path = os.path.join(output_dir, "model.pt")
    output_history_path = os.path.join(output_dir, "history.pickle")

    if previous_history is None:
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
    else:
        history = previous_history

    since = time.time()
    best_acc = 0.0
    # early stopping counter
    epochs_no_progress = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)
        res = run_epoch(
            model,
            optimizer,
            criterion,
            scheduler,
            dataloaders,
            device,
            gradient_accumulation,
            train_stats_period,
        )
        print(
            f"Train Loss: {res.train_loss:.4f} Train Acc: {res.train_acc:.4f}\n"
            f"Val Loss: {res.val_loss:.4f} Val Acc: {res.val_acc:.4f}"
        )

        history = update_save_history(history, res, output_history_path)

        # deep copy the model
        if res.val_acc > best_acc:
            best_acc = res.val_acc
            torch.save(model.state_dict(), output_model_path)
            epochs_no_progress = 0
        else:
            if warmup_period <= epoch:
                epochs_no_progress += 1

        # early stopping mechanism
        if warmup_period <= epoch and epochs_no_progress >= patience:
            break

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(torch.load(output_model_path))
    return model, history


def update_save_history(
    history: dict, res: EpochResults, hist_output_path: str
) -> dict:
    history["train_loss"].append(res.train_acc)
    history["train_acc"].append(res.train_acc)
    history["val_loss"].append(res.val_loss)
    history["val_acc"].append(res.val_acc)

    try:
        with open(hist_output_path, "wb") as handle:
            pickle.dump(history, handle)
    except Exception as e:
        print("WARNING: Error while saving training history: ", e)

    return history


def run_epoch(
    model: nn.Module,
    optimizer,
    criterion,
    scheduler,
    dataloaders,
    device: str,
    gradient_accumulation: int = 1,
    train_stats_period: int = -1,
) -> EpochResults:
    train_loss, train_acc = train_epoch(
        model,
        optimizer,
        criterion,
        scheduler,
        dataloaders["train"],
        device,
        gradient_accumulation,
        train_stats_period,
    )
    val_loss, val_acc = val_epoch(model, criterion, dataloaders["val"], device)
    return EpochResults(
        train_loss=train_loss,
        train_acc=train_acc,
        val_loss=val_loss,
        val_acc=val_acc,
    )


def train_epoch(
    model: nn.Module,
    optimizer,
    criterion,
    scheduler,
    dataloader,
    device: str,
    gradient_accumulation: int = 1,
    train_stats_period: int = -1,
) -> tuple[float, float]:
    # Each epoch has a training and validation phase

    model.train()  # Set model to training mode

    running_loss = 0.0
    running_corrects = 0

    iteration = 0
    samples = 0
    # Iterate over data.
    for inputs, labels in tqdm(dataloader):
        samples += len(labels)
        iteration += 1

        inputs = inputs.to(device)
        labels = labels.to(device)
     
        # forward pass with gradient accumulation
        if iteration % gradient_accumulation == 0:
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            scheduler.step()

            # statistics
            running_loss += loss.detach().item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).double().cpu()

            # release GPU VRAM https://discuss.pytorch.org/t/gpu-memory-consumption-increases-while-training/2770/4
            del loss, outputs, preds

        if train_stats_period > 0 and iteration % train_stats_period == 0:
            print(
                f"Loss: {running_loss / samples:.6f} Accuracy: {running_corrects / samples :.5f}"
            )

    epoch_loss = running_loss / samples
    epoch_acc = running_corrects / samples

    train_loss = epoch_loss
    train_acc = epoch_acc

    return train_loss, train_acc


def val_epoch(
    model: nn.Module, criterion, dataloader, device: str
) -> tuple[float, float]:
    model.eval()  # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    samples = 0
    # Iterate over data.
    for inputs, labels in tqdm(dataloader):
        samples += len(labels)
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).double().cpu()

        # release GPU VRAM https://discuss.pytorch.org/t/gpu-memory-consumption-increases-while-training/2770/4
        del loss, outputs, preds

    epoch_loss = running_loss / samples
    epoch_acc = running_corrects / samples

    return epoch_loss, epoch_acc


def test(model, test_dataloader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    actual = []
    preds = []

    # Iterate over batches
    for inputs, labels in tqdm(test_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(inputs)

        # Get and store predictions
        _, predicted = torch.max(outputs, 1)

        for label, pred in zip(labels, predicted):
            actual.append(label.cpu())
            preds.append(pred.cpu())

    return np.array(actual), np.array(preds)
