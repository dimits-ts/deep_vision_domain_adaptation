import lib.torch_train_eval

import torch
from torch import nn
from tqdm.auto import tqdm
import numpy as np

import os
import time
import itertools


def coral_loss(source, target):
    """
    Computes the CORAL loss between source and target features.
    :param source: Source domain features
    :param target: Target domain features
    :return: CORAL loss
    """
    d = source.size(1)
    source_coral = torch.matmul(source.t(), source) / source.size(0)
    target_coral = torch.matmul(target.t(), target) / target.size(0)
    loss = torch.mean(torch.square(source_coral - target_coral)) / (4 * d * d)
    return loss


def coral_train_model(
    model: nn.Module,
    criterion,
    optimizer,
    scheduler,
    device: str,
    source_train_dataloader: torch.utils.data.DataLoader,
    target_train_dataloader: torch.utils.data.DataLoader,
    source_val_dataloader: torch.utils.data.DataLoader,
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
    :param source_train_dataloader: DataLoader for training data
    :type source_train_dataloader: torch.utils.data.DataLoader
    :param source_val_dataloader: DataLoader for validation data
    :type source_val_dataloader: torch.utils.data.DataLoader
    :param target_train_dataloader:
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
        res = coral_run_epoch(
            model,
            optimizer,
            criterion,
            scheduler,
            source_train_dataloader=source_train_dataloader,
            target_train_dataloader=target_train_dataloader,
            source_val_dataloader=source_val_dataloader,
            device=device,
            gradient_accumulation=gradient_accumulation,
            train_stats_period=train_stats_period,
            verbose=verbose
        )
        print(
            f"Train Loss: {res.train_loss:.4f} Train Acc: {res.train_acc:.4f}\n"
            f"Val Loss: {res.val_loss:.4f} Val Acc: {res.val_acc:.4f}"
        )

        history = lib.torch_train_eval.update_save_history(history, res, output_history_path)

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


def coral_run_epoch(
    model: nn.Module,
    optimizer,
    criterion,
    scheduler,
    source_train_dataloader: torch.utils.data.DataLoader,
    target_train_dataloader: torch.utils.data.DataLoader,
    source_val_dataloader: torch.utils.data.DataLoader,
    device: str,
    gradient_accumulation: int = 1,
    train_stats_period: int = 100000,
    verbose: bool = True
) -> lib.torch_train_eval.EpochResults:
    """
    Runs a single epoch of training and validation.
    :param model: The model to be trained
    :type model: nn.Module
    :param optimizer: Optimizer
    :param criterion: Loss function
    :param scheduler: Learning rate scheduler
    :param source_train_dataloader:
    :param source_val_dataloader:
    :param target_train_dataloader:
    :param device: Device to use for training (e.g., 'cpu' or 'cuda')
    :type device: str
    :param gradient_accumulation: Number of steps to accumulate gradients before performing an optimizer step, defaults
    to 1
    :type gradient_accumulation: int, optional
    :param train_stats_period: Frequency (in iterations) of printing training statistics, defaults to 100000
    :type train_stats_period: int, optional
    :param verbose: Whether to print detailed training progress, defaults to True
    :type verbose: bool, optional
    :return: Results of the epoch
    :rtype: EpochResults
    """
    train_loss, train_acc = coral_train_epoch(
        model,
        optimizer,
        criterion,
        scheduler,
        dataloader_source=source_train_dataloader,
        dataloader_target=target_train_dataloader,
        device=device,
        gradient_accumulation=gradient_accumulation,
        train_stats_period=train_stats_period,
        verbose=verbose
    )
    val_loss, val_acc = lib.torch_train_eval.val_epoch(model,
                                                       criterion,
                                                       source_val_dataloader,
                                                       device,
                                                       verbose=verbose)
    return lib.torch_train_eval.EpochResults(
        train_loss=train_loss,
        train_acc=train_acc,
        val_loss=val_loss,
        val_acc=val_acc,
    )


def coral_train_epoch(
        model: nn.Module,
        optimizer,
        criterion,
        scheduler,
        dataloader_source,
        dataloader_target,
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
    :param dataloader_source: DataLoader for source domain data
    :param dataloader_target: DataLoader for target domain data
    :param device: Device to use for training (e.g., 'cpu' or 'cuda')
    :type device: str
    :param gradient_accumulation: Number of steps to accumulate gradients before performing an optimizer step, defaults
     to 1
    :type gradient_accumulation: int, optional
    :param train_stats_period: Frequency (in iterations) of printing training statistics, defaults to 100000
    :type train_stats_period: int, optional
    :param verbose: Whether to print detailed training progress, defaults to True
    :type verbose: bool, optional
    :return: Training loss and accuracy for the epoch
    :rtype: tuple[float, float]
    """
    # Ensure the model is in training mode
    model.train()

    running_loss = 0.0
    running_corrects = 0

    iteration = 0
    samples = 0

    iterable_source = tqdm(dataloader_source) if verbose else dataloader_source
    iterable_target = dataloader_target
    # Zip the source and target dataloaders, cycling through the smaller dataloader
    for (inputs_source, labels_source), (inputs_target, _) in zip(iterable_source,
                                                                  itertools.cycle(iterable_target)):
        samples += len(labels_source)
        iteration += 1

        inputs_source = inputs_source.to(device)
        labels_source = labels_source.to(device)
        inputs_target = inputs_target.to(device)

        outputs_source = model(inputs_source)
        outputs_target = model(inputs_target)

        _, preds = torch.max(outputs_source, 1)

        classification_loss = criterion(outputs_source, labels_source) / gradient_accumulation
        domain_loss = coral_loss(outputs_source, outputs_target) / gradient_accumulation
        loss = classification_loss + domain_loss

        with torch.set_grad_enabled(True):
            loss.backward()

        loss_float = loss.detach().item()
        # Update running loss and correct counts
        running_loss += loss_float * inputs_source.size(0)
        running_corrects += torch.sum(preds == labels_source.data).double().cpu()

        # Release GPU memory
        del inputs_source, outputs_source, loss
        del inputs_target, outputs_target

        # Perform optimizer step with gradient accumulation
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
