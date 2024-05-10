import copy
from typing import Callable

import torch
from torch import nn
import numpy as np
from tqdm.auto import tqdm

import time
import os

import lib.data
import lib.torch_train_eval


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
        source_train_dataset: lib.data.ImageDataset,
        source_val_dataset: lib.data.ImageDataset,
        labeled_dataloader_initializer: Callable[
            [lib.data.ImageDataset], torch.utils.data.DataLoader
        ],
        unlabeled_dataloader_initializer: Callable[
            [lib.data.UnlabeledImageDataset], torch.utils.data.DataLoader
        ],
        unlabeled_target_train_dataset: lib.data.UnlabeledImageDataset,
        target_val_dataset: lib.data.ImageDataset,
        output_dir: str,
        source_train_period: int,
        joint_training_period: int,
        max_epoch: int,
        rho: float = 3,
        alpha_f: float = 3,
        previous_source_history: dict[str, list[float]] = None,
        previous_target_history: dict[str, list[float]] = None,
) -> tuple[nn.Module, dict[str, np.ndarray], dict[str, np.ndarray]]:
    unlabeled_target_train_dataset = copy.deepcopy(unlabeled_target_train_dataset)
    # this is where we will separately store the pseudo-labeled data at each epoch
    pseudolabeled_target_train_dataset = lib.data.ImageDataset(
        parser_func=unlabeled_target_train_dataset.parser_func,
        preprocessing_func=unlabeled_target_train_dataset.preprocessing_func)

    source_dataloaders = {
        "train": labeled_dataloader_initializer(source_train_dataset),
        "val": labeled_dataloader_initializer(source_val_dataset)
    }

    output_model_path = os.path.join(output_dir, "model.pt")
    output_history_path_source = os.path.join(output_dir, "source_history.pickle")
    output_history_path_target = os.path.join(output_dir, "target_history.pickle")

    if previous_source_history is None:
        source_history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
    else:
        source_history = previous_source_history

    if previous_target_history is None:
        target_history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "target_samples_num": [],
        }
    else:
        target_history = previous_target_history

    since = time.time()
    torch.save(model.state_dict(), output_model_path)
    best_acc = 0.0

    # Keep in memory which files have been pseudo-labeled for debug and stats purposes
    seen_samples = set()
    for sample in pseudolabeled_target_train_dataset.samples:
        seen_samples.add(sample)

    for epoch in range(max_epoch):
        print(f"Epoch {epoch}/{max_epoch - 1}")
        print("-" * 10)

        if epoch < source_train_period:
            alpha_t = 0
        elif epoch < joint_training_period:
            alpha_t = (epoch - source_train_period) / (joint_training_period - epoch) * alpha_f
        else:
            alpha_t = alpha_f

        # ========= Source domain forward and backward pass =========

        # we subsample the source data since the classifier is already trained on them
        #source_num_samples = len(pseudolabeled_target_train_dataset)

        #source_dataloaders = recreate_dataloaders(
        #    source_train_dataset,
        #    source_num_samples,
        #    labeled_dataloader_initializer,
        #    source_val_loader,
        #)
        source_res = lib.torch_train_eval.run_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            dataloaders=source_dataloaders,
            device=device,
        )
        source_history = lib.torch_train_eval.update_save_history(
            source_history, source_res, epoch, output_history_path_source
        )
        last_val_acc = source_res.val_acc
        print(f"Source dataset Train Loss: {source_res.train_loss:.4f} Train Acc: {source_res.train_acc:.4f}\n"
              f"Source dataset Val Loss: {source_res.val_loss:.4f} Val Acc: {source_res.val_acc:.4f}")

        # ========= Pseudo-labeling task =========
        if alpha_t != 0:
            target_res, target_history = target_task(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                labeled_dataloader_initializer=labeled_dataloader_initializer,
                labeled_target_train_dataset=pseudolabeled_target_train_dataset,
                target_val_dataset=target_val_dataset,
                unlabeled_dataloader_initializer=unlabeled_dataloader_initializer,
                unlabeled_target_train_dataset=unlabeled_target_train_dataset,
                last_val_acc=last_val_acc,
                rho=rho,
                backprop_weight=alpha_t,
                device=device,
                seen_samples=seen_samples,
                history=target_history,
                output_history_path=output_history_path_target,
                epoch=epoch
            )
            print(
                f"Target dataset Val Loss: {target_res.val_loss:.4f} Val Acc: {target_res.val_acc:.4f}"
            )

        # ========= Print & checkpoint =========

        # deep copy the model
        if source_res.val_acc > best_acc:
            best_acc = source_res.val_acc
            torch.save(model.state_dict(), output_model_path)

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(torch.load(output_model_path))

    return model, source_history, target_history, pseudolabeled_target_train_dataset


def target_task(
        model,
        optimizer,
        criterion,
        scheduler,
        labeled_dataloader_initializer,
        labeled_target_train_dataset,
        target_val_dataset,
        unlabeled_dataloader_initializer,
        unlabeled_target_train_dataset,
        last_val_acc,
        rho,
        backprop_weight,
        device,
        seen_samples,
        history,
        output_history_path,
        epoch: int
):
    pseudo_label(
        model=model,
        unlabeled_dataloader_initializer=unlabeled_dataloader_initializer,
        unlabeled_target_train_dataset=unlabeled_target_train_dataset,
        labeled_target_train_dataset=labeled_target_train_dataset,
        accuracy=last_val_acc,
        rho=rho,
        device=device,
    )

    new_samples = [
        sample
        for sample in labeled_target_train_dataset.samples
        if sample not in seen_samples
    ]
    for sample in labeled_target_train_dataset.samples:
        seen_samples.add(sample)

    # debug
    print(new_samples)

    # ========= Target domain forward and backward pass =========
    target_dataloaders = {
        "train": labeled_dataloader_initializer(labeled_target_train_dataset),
        "val": labeled_dataloader_initializer(target_val_dataset),
    }
    target_res = lib.torch_train_eval.run_epoch(
        model,
        optimizer,
        criterion,
        scheduler,
        target_dataloaders,
        device,
        backprop_weight
    )
    # this is where we use the new_samples we kept in memory above
    history["target_samples_num"].append(len(new_samples))
    target_history = lib.torch_train_eval.update_save_history(
        history, target_res, epoch, output_history_path
    )

    return target_res, target_history


def pseudo_label(
        model,
        unlabeled_dataloader_initializer,
        unlabeled_target_train_dataset,
        labeled_target_train_dataset,
        accuracy,
        rho,
        device,
):
    threshold = adaptive_threshold(classification_accuracy=accuracy, rho=rho)
    print(f"Selected threshold {threshold} on val_acc {accuracy}")
    samples = select_samples(
        model,
        unlabeled_dataloader_initializer(unlabeled_target_train_dataset),
        threshold,
        device,
    )

    print(
        f"Selected {len(samples[0])}/{len(unlabeled_target_train_dataset)} remaining images to be included in "
        f"next epoch"
    )
    for image_path, class_id in zip(samples[0], samples[1]):
        # update datasets and recreate dataloaders
        unlabeled_target_train_dataset.remove(image_path)
        labeled_target_train_dataset.add(image_path, class_id)


def recreate_dataloaders(
        source_train_dataset,
        source_num_samples,
        labeled_dataloader_initializer,
        source_val_loader,
):
    indices = torch.randperm(len(source_train_dataset)).tolist()[:source_num_samples]
    subset_sampler = torch.utils.data.SubsetRandomSampler(indices)

    return {
        "train": labeled_dataloader_initializer(
            source_train_dataset, sampler=subset_sampler
        ),
        "val": source_val_loader,
    }
