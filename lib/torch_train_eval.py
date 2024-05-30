import torch
from torch import nn
import numpy as np
from tqdm.auto import tqdm

import time
import pickle
import os


class EpochResults:
    """
    Stores results for a single epoch.

    :param train_loss: Training loss for the epoch
    :type train_loss: float
    :param train_acc: Training accuracy for the epoch
    :type train_acc: float
    :param val_loss: Validation loss for the epoch
    :type val_loss: float
    :param val_acc: Validation accuracy for the epoch
    :type val_acc: float
    """

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
    train_stats_period: int = 100000,
    verbose: bool = True
) -> tuple[nn.Module, dict[str, np.ndarray]]:
    """
    Trains the model using the given data loaders, criterion, optimizer, and scheduler.

    :param model: The model to be trained
    :type model: nn.Module
    :param criterion: Loss function
    :param optimizer: Optimizer
    :param scheduler: Learning rate scheduler
    :param device: Device to use for training (e.g., 'cpu' or 'cuda')
    :type device: str
    :param train_dataloader: DataLoader for training data
    :type train_dataloader: torch.utils.data.DataLoader
    :param val_dataloader: DataLoader for validation data
    :type val_dataloader: torch.utils.data.DataLoader
    :param output_dir: Directory to save the model and training history
    :type output_dir: str
    :param num_epochs: Number of epochs to train, defaults to 25
    :type num_epochs: int, optional
    :param patience: Number of epochs with no improvement after warmup to stop training, defaults to 1
    :type patience: int, optional
    :param warmup_period: Number of initial epochs to ignore for early stopping, defaults to 10
    :type warmup_period: int, optional
    :param previous_history: Dictionary to continue training from previous history, defaults to None
    :type previous_history: dict[str, list[float]], optional
    :param gradient_accumulation: Number of steps to accumulate gradients before performing an optimizer step, defaults to 1
    :type gradient_accumulation: int, optional
    :param train_stats_period: Frequency (in iterations) of printing training statistics, defaults to 100000
    :type train_stats_period: int, optional
    :param verbose: Whether to print detailed training progress, defaults to True
    :type verbose: bool, optional
    :return: Trained model and training history
    :rtype: tuple[nn.Module, dict[str, np.ndarray]]
    """
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

    for epoch in range(len(history["train_loss"]), num_epochs):
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
            verbose=verbose
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
    """
   Updates and saves the training history with results from the current epoch.

   :param history: Training history
   :type history: dict
   :param res: Results of the current epoch
   :type res: EpochResults
   :param hist_output_path: Path to save the updated history
   :type hist_output_path: str
   :return: Updated history
   :rtype: dict
   """
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
    train_stats_period: int = 100000,
    verbose: bool = True
) -> EpochResults:
    """
    Runs a single epoch of training and validation.

    :param model: The model to be trained
    :type model: nn.Module
    :param optimizer: Optimizer
    :param criterion: Loss function
    :param scheduler: Learning rate scheduler
    :param dataloaders: Dictionary containing training and validation data loaders
    :param device: Device to use for training (e.g., 'cpu' or 'cuda')
    :type device: str
    :param gradient_accumulation: Number of steps to accumulate gradients before performing an optimizer step, defaults to 1
    :type gradient_accumulation: int, optional
    :param train_stats_period: Frequency (in iterations) of printing training statistics, defaults to 100000
    :type train_stats_period: int, optional
    :param verbose: Whether to print detailed training progress, defaults to True
    :type verbose: bool, optional
    :return: Results of the epoch
    :rtype: EpochResults
    """
    train_loss, train_acc = train_epoch(
        model,
        optimizer,
        criterion,
        scheduler,
        dataloaders["train"],
        device,
        gradient_accumulation,
        train_stats_period,
        verbose=verbose
    )
    val_loss, val_acc = val_epoch(model, criterion, dataloaders["val"], device, verbose=verbose)
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
    train_stats_period: int = 100000,
    verbose: bool = True
) -> tuple[float, float]:
    """
    Runs a single training epoch.

    :param model: The model to be trained
    :type model: nn.Module
    :param optimizer: Optimizer
    :param criterion: Loss function
    :param scheduler: Learning rate scheduler
    :param dataloader: DataLoader for training data
    :param device: Device to use for training (e.g., 'cpu' or 'cuda')
    :type device: str
    :param gradient_accumulation: Number of steps to accumulate gradients before performing an optimizer step, defaults to 1
    :type gradient_accumulation: int, optional
    :param train_stats_period: Frequency (in iterations) of printing training statistics, defaults to 100000
    :type train_stats_period: int, optional
    :param verbose: Whether to print detailed training progress, defaults to True
    :type verbose: bool, optional
    :return: Training loss and accuracy for the epoch
    :rtype: tuple[float, float]
    """
    # Each epoch has a training and validation phase

    model.train()  # Set model to training mode

    running_loss = 0.0
    running_corrects = 0

    iteration = 0
    samples = 0

    iterable = tqdm(dataloader) if verbose else dataloader
    # Iterate over data.
    for inputs, labels in iterable:
        samples += len(labels)
        iteration += 1

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        loss = criterion(outputs, labels) / gradient_accumulation

        with torch.set_grad_enabled(True):
            loss.backward()

        loss_float = loss.detach().item()
        # statistics
        running_loss += loss_float * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).double().cpu()
        
        # release GPU VRAM before next invocation 
        # https://discuss.pytorch.org/t/gpu-memory-consumption-increases-while-training/2770/4
        del inputs, outputs, loss
     
        # forward pass with gradient accumulation
        if iteration % gradient_accumulation == 0:
            with torch.set_grad_enabled(True):
                optimizer.step()
                optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        if iteration % train_stats_period == 0:
            print(
                f"Loss: {running_loss / samples:.6f} Accuracy: {running_corrects / samples :.5f}"
            )

    epoch_loss = running_loss / samples
    epoch_acc = running_corrects / samples

    train_loss = epoch_loss
    train_acc = epoch_acc

    return train_loss, train_acc


def val_epoch(
    model: nn.Module, criterion, dataloader, device: str, verbose: bool = True
) -> tuple[float, float]:
    """
    Runs a single validation epoch.

    :param model: The model to be evaluated
    :type model: nn.Module
    :param criterion: Loss function
    :param dataloader: DataLoader for validation data
    :param device: Device to use for validation (e.g., 'cpu' or 'cuda')
    :type device: str
    :param verbose: Whether to print detailed validation progress, defaults to True
    :type verbose: bool, optional
    :return: Validation loss and accuracy for the epoch
    :rtype: tuple[float, float]
    """
    model.eval()  # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    samples = 0
    iterable = tqdm(dataloader) if verbose else dataloader
    # Iterate over data.
    for inputs, labels in iterable:
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
    """
    Evaluates the model on the test data.

    :param model: The trained model
    :type model: nn.Module
    :param test_dataloader: DataLoader for test data
    :param device: Device to use for testing (e.g., 'cpu' or 'cuda')
    :type device: str
    :return: Actual and predicted labels
    :rtype: tuple[np.ndarray, np.ndarray]
    """
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
