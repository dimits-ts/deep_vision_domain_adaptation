import copy
from typing import Callable

import torch
from torch import nn
import numpy as np
from tqdm.auto import tqdm

import time
import os
import pickle

import lib.data
import lib.torch_train_eval


# from https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1410.pdf section 4.1
def adaptive_threshold(classification_accuracy: float, rho: float = 3) -> float:
    """
    Calculate an adaptive threshold based on classification accuracy
    as defined in https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1410.pdf section 4.1.

    :param classification_accuracy: Accuracy of the classification model.
    :type classification_accuracy: float
    :param rho: Parameter that controls the steepness of the sigmoid function, defaults to 3.
    :type rho: float, optional
    :return: Calculated adaptive threshold.
    :rtype: float
    """
    return 1 / (1 + np.exp(-rho * classification_accuracy))


def select_samples(
        model: nn.Module, dataset, threshold: float, device: str, verbose: bool=True
) -> tuple[list[str], list[int]]:
    """
    Select samples from the dataset with confidence higher than the given threshold.

    :param model: The model used for prediction.
    :type model: nn.Module
    :param dataset: The dataset to select samples from.
    :type dataset: Dataset
    :param threshold: Confidence threshold for selecting samples.
    :type threshold: float
    :param device: Device to perform computations on (e.g., 'cpu', 'cuda').
    :type device: str
    :param verbose: Whether to display a progress bar, defaults to True.
    :type verbose: bool, optional
    :return: Lists of selected sample file paths and their predicted labels.
    :rtype: tuple[list[str], list[int]]
    """
    model.eval()
    
    selected_samples_ls = []
    predicted_labels_ls = []
    iterable = tqdm(dataset) if verbose else dataset

    for inputs, file_path in iterable:
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
        num_epochs: int = 25,
        gradient_accumulation: int = 1,
        pseudo_sample_period: int = 1,
        rho=3,
        previous_source_history: dict[str, list[float]] = None,
        previous_target_history: dict[str, list[float]] = None,
        verbose: bool = True
) -> tuple[
    nn.Module,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    list[list[tuple[str, int]]],
]:
    """
    Train an adaptive model with pseudo-labeling on target domain data.

    :param model: The model to be trained.
    :type model: nn.Module
    :param criterion: Loss function.
    :param optimizer: Optimizer for model parameters.
    :param scheduler: Learning rate scheduler.
    :param device: Device to perform computations on (e.g., 'cpu', 'cuda').
    :type device: str
    :param source_train_dataset: Labeled training dataset from the source domain.
    :type source_train_dataset: lib.data.ImageDataset
    :param source_val_dataset: Labeled validation dataset from the source domain.
    :type source_val_dataset: lib.data.ImageDataset
    :param labeled_dataloader_initializer: Function to initialize a DataLoader for labeled data.
    :type labeled_dataloader_initializer: Callable[[lib.data.ImageDataset], torch.utils.data.DataLoader]
    :param unlabeled_dataloader_initializer: Function to initialize a DataLoader for unlabeled data.
    :type unlabeled_dataloader_initializer: Callable[[lib.data.UnlabeledImageDataset], torch.utils.data.DataLoader]
    :param unlabeled_target_train_dataset: Unlabeled training dataset from the target domain.
    :type unlabeled_target_train_dataset: lib.data.UnlabeledImageDataset
    :param target_val_dataset: Labeled validation dataset from the target domain.
    :type target_val_dataset: lib.data.ImageDataset
    :param output_dir: Directory to save model checkpoints and history.
    :type output_dir: str
    :param num_epochs: Number of training epochs, defaults to 25.
    :type num_epochs: int, optional
    :param gradient_accumulation: Number of steps to accumulate gradients before updating model parameters, defaults to 1.
    :type gradient_accumulation: int, optional
    :param pseudo_sample_period: Frequency (in epochs) to perform pseudo-labeling, defaults to 1.
    :type pseudo_sample_period: int, optional
    :param rho: Parameter for adaptive threshold calculation, defaults to 3.
    :type rho: int, optional
    :param previous_source_history: Training history for the source domain, defaults to None.
    :type previous_source_history: dict[str, list[float]], optional
    :param previous_target_history: Training history for the target domain, defaults to None.
    :type previous_target_history: dict[str, list[float]], optional
    :param verbose: Whether to display detailed training progress, defaults to True.
    :type verbose: bool, optional
    :return: Tuple containing the trained model, source domain history, target domain history, and pseudo-label history.
    :rtype: tuple[nn.Module, dict[str, np.ndarray], dict[str, np.ndarray], list[list[tuple[str, int]]]]
    """
    # a list containing the selected pseudo labeled samples for each epoch
    pseudo_label_history = []
    unlabeled_target_train_dataset = copy.deepcopy(unlabeled_target_train_dataset)

    # this is where we will separately store the pseudo-labeled data at each epoch
    pseudolabeled_target_train_dataset = lib.data.ImageDataset(
        parser_func=unlabeled_target_train_dataset.parser_func,
        preprocessing_func=unlabeled_target_train_dataset.preprocessing_func,
    )

    source_dataloaders = {
        "train": labeled_dataloader_initializer(source_train_dataset),
        "val": labeled_dataloader_initializer(source_val_dataset),
    }

    output_model_path = os.path.join(output_dir, "model.pt")
    output_history_path_source = os.path.join(output_dir, "source_history.pickle")
    output_history_path_target = os.path.join(output_dir, "target_history.pickle")
    output_label_history_path = os.path.join(output_dir, "label_history.pickle")

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
        target_history = previous_target_history

    since = time.time()
    torch.save(model.state_dict(), output_model_path)
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # ========= Source domain forward and backward pass =========

        # we subsample the source data since the classifier is already trained on them
        source_num_samples = len(pseudolabeled_target_train_dataset)

        # for all iterations except first
        if source_num_samples != 0:
            indices = torch.randperm(len(source_train_dataset)).tolist()[
                      :source_num_samples
                      ]
            subset_sampler = torch.utils.data.SubsetRandomSampler(indices)
            source_dataloaders = {
                "train": labeled_dataloader_initializer(
                    source_train_dataset, sampler=subset_sampler
                ),
                "val": labeled_dataloader_initializer(source_val_dataset),
            }

            source_res = lib.torch_train_eval.run_epoch(
                model,
                optimizer,
                criterion,
                scheduler,
                source_dataloaders,
                device,
                gradient_accumulation=gradient_accumulation,
                verbose=verbose,
                train_stats_period=int(10e6)
            )

            # set new validation accuracy
            last_val_acc = source_res.val_acc

            source_history = lib.torch_train_eval.update_save_history(
                source_history, source_res, output_history_path_source
            )
            

            print(f"Source dataset Train Loss: {source_res.train_loss:.4f} Train Acc: {source_res.train_acc:.4f}\n"
                  f"Source dataset Val Loss: {source_res.val_loss:.4f} Val Acc: {source_res.val_acc:.4f}\n")
        else:
            _, last_val_acc = lib.torch_train_eval.val_epoch(
                model,
                criterion=criterion,
                dataloader=source_dataloaders["val"],
                device=device,
                verbose=verbose
            )

        if epoch % pseudo_sample_period == 0:
            # ========= Pseudo-labeling task =========
            threshold = adaptive_threshold(classification_accuracy=last_val_acc, rho=rho)
            samples = select_samples(
                model,
                unlabeled_dataloader_initializer(unlabeled_target_train_dataset),
                threshold,
                device,
                verbose=verbose
            )

            print(
                f"Selected {len(samples[0])}/{len(unlabeled_target_train_dataset)} images on threshold {threshold}"
            )

            pseudo_label_history.append([])
            for image_path, class_id in zip(samples[0], samples[1]):
                # update datasets and recreate dataloaders
                unlabeled_target_train_dataset.remove(image_path)
                pseudolabeled_target_train_dataset.add(image_path, class_id)

                # update pseudo label history
                pseudo_label_history[-1].append((image_path, class_id))
            print(pseudo_label_history[-1])

            # save label history
            try:
                with open(output_label_history_path, "wb") as handle:
                    pickle.dump(pseudo_label_history, handle)
            except Exception as e:
                print("WARNING: Error while saving label history: ", e)

        # ========= Target domain forward and backward pass =========
        target_dataloaders = {
            "train": labeled_dataloader_initializer(
                pseudolabeled_target_train_dataset
            ),
            "val": labeled_dataloader_initializer(target_val_dataset),
        }
        target_res = lib.torch_train_eval.run_epoch(
            model,
            optimizer,
            criterion,
            scheduler,
            target_dataloaders,
            device,
            verbose=verbose
        )
        
        target_history = lib.torch_train_eval.update_save_history(
            target_history, target_res, output_history_path_target
        )
        print(f"Target dataset Val Loss: {target_res.val_loss:.4f} Val Acc: {target_res.val_acc:.4f}")

        # ========= Checkpoint =========

        # deep copy the model
        if last_val_acc > best_acc:
            best_acc = last_val_acc
            torch.save(model.state_dict(), output_model_path)

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(torch.load(output_model_path))

    return model, source_history, target_history, pseudo_label_history
