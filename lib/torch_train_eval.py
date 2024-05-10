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
) -> EpochResults:
    train_loss, train_acc = train_epoch(
        model,
        optimizer,
        criterion,
        scheduler,
        dataloaders["train"],
        device,
    )
    val_loss, val_acc = val_epoch(
        model, criterion, dataloaders["val"], device
    )
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
) -> tuple[float, float]:
    # Each epoch has a training and validation phase

    model.train()  # Set model to training mode

    running_loss = 0.0
    running_corrects = 0

    samples = 0
    # Iterate over data.
    for inputs, labels in tqdm(dataloader):
        samples += len(labels)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        scheduler.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        #print(running_loss / samples)

    epoch_loss = running_loss / samples
    epoch_acc = running_corrects.double().cpu() / samples

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
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / samples
    epoch_acc = running_corrects.double().cpu() / samples

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
