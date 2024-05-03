from typing import Callable

import torch
from torch import nn
import numpy as np
from tqdm.auto import tqdm

import time
import os

import tasks.data
import tasks.torch_train_eval


# from https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1410.pdf section 4.1
def adaptive_threshold(classification_accuracy: float, rho: float = 3) -> float:
    return 1 / (1 + np.exp(-rho * classification_accuracy))


def select_samples(
        model: nn.Module, dataset, threshold: float, device: str
) -> tuple[list[str], list[int]]:
    selected_samples_ls = []
    predicted_labels_ls = []

    # Iterate over batches
    for inputs, file_path in tqdm(dataset):
        inputs = inputs.to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted_labels = torch.max(outputs, 1)

        # Store results
        for logit, pred in zip(outputs, predicted_labels):
            confidence = torch.max(nn.Softmax()(logit))

            if confidence > threshold:
                selected_samples_ls.append(file_path[0])
                predicted_labels_ls.append(pred.item())

    return selected_samples_ls, predicted_labels_ls


def train_adaptive_model(
        model: nn.Module,
        criterion,
        optimizer,
        scheduler,
        device: str,
        source_train_dataset: tasks.data.ImageDataset,
        source_val_dataset: tasks.data.ImageDataset,
        labeled_dataloader_initializer: Callable[[tasks.data.ImageDataset], torch.utils.data.DataLoader],
        unlabeled_dataloader_initializer: Callable[[tasks.data.UnlabeledImageDataset], torch.utils.data.DataLoader],
        target_train_dataset: tasks.data.UnlabeledImageDataset,
        target_val_dataset: tasks.data.UnlabeledImageDataset,
        output_dir: str,
        num_epochs: int = 25,
        previous_source_history: dict[str, list[float]] = None,
        previous_target_history: dict[str, list[float]] = None
) -> tuple[nn.Module, dict[str, np.ndarray], dict[str, np.ndarray]]:
    original_target_samples_num = len(target_train_dataset)  # for printing

    dataloaders = {"train": labeled_dataloader_initializer(source_train_dataset),
                   "val": labeled_dataloader_initializer(source_val_dataset)}

    output_model_path = os.path.join(output_dir, "model.pt")
    output_history_path_source = os.path.join(output_dir, "source_history.pickle")
    output_history_path_target = os.path.join(output_dir, "target_history.pickle")

    if previous_source_history is None:
        source_history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
    else:
        source_history = previous_source_history

    if previous_target_history is None:
        target_history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
    else:
        target_history = previous_source_history

    since = time.time()
    torch.save(model.state_dict(), output_model_path)
    best_acc = 0.0

    for epoch in range(num_epochs):
        # ========= Normal forward and backward pass =========
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        source_res = tasks.torch_train_eval.run_epoch(
            model,
            optimizer,
            criterion,
            scheduler,
            dataloaders,
            device,
        )
        source_history = tasks.torch_train_eval.update_save_history(source_history,
                                                                    source_res,
                                                                    output_history_path_source)

        # ========= Add statistics for target dataset =========
        target_val_loss, target_val_acc = tasks.torch_train_eval.val_epoch(
            model, criterion, labeled_dataloader_initializer(target_val_dataset), device
        )
        # I will ignore training statistics for now (since the training data is supposed to be unlabeled)
        target_res = tasks.torch_train_eval.EpochResults(train_loss=0,
                                                         train_acc=0,
                                                         val_loss=target_val_loss,
                                                         val_acc=target_val_acc)
        target_history = tasks.torch_train_eval.update_save_history(target_history, target_res,
                                                                    output_history_path_target)

        # ========= Pseudo-labeling task =========
        #threshold = adaptive_threshold(classification_accuracy=source_res.val_acc)
        threshold = adaptive_threshold(classification_accuracy=0.8)
        samples = select_samples(model, unlabeled_dataloader_initializer(target_train_dataset), threshold, device)

        print(f"Selected {len(samples[0])}/{original_target_samples_num} images to be included in next epoch")
        for image_path, class_id in zip(samples[0], samples[1]):
            # update datasets and recreate dataloaders
            target_train_dataset.remove(image_path)

            source_train_dataset.add(image_path, class_id)
            source_train_loader = labeled_dataloader_initializer(source_train_dataset)
            dataloaders["train"] = source_train_loader

        # deep copy the model
        if target_val_acc > best_acc:
            best_acc = target_val_acc
            torch.save(model.state_dict(), output_model_path)

        # ========= Print & checkpoint =========
        print(
            f"source dataset Train Loss: {source_res.train_loss:.4f} Train Acc: {source_res.train_acc:.4f}\n"
            f"Source dataset Val Loss: {source_res.val_loss:.4f} Val Acc: {source_res.val_acc:.4f}\n"
            f"Target dataset Val Loss: {target_res.val_loss:.4f} Val Acc: {target_res.val_acc:.4f}"
        )

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(torch.load(output_model_path))

    return model, source_history, target_history
