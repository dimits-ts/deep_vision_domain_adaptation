import torch
import imageio.v2 as imageio
from torchvision.transforms import v2
import numpy as np

import lib.data


def create_padded_dataloader(
        dataset: lib.data.ImageDataset,
        shuffle: bool = True,
        sampler=None,
        batch_size=1,
        num_workers=3
):
    """
    Create a DataLoader with padding for batches.

    :param dataset: The dataset to load data from.
    :type dataset: lib.data.ImageDataset
    :param shuffle: Whether to shuffle the data, defaults to True.
    :type shuffle: bool, optional
    :param sampler: Sampler for data loading, mutually exclusive with shuffle, defaults to None.
    :type sampler: Sampler, optional
    :param batch_size: Number of samples per batch, defaults to 1.
    :type batch_size: int, optional
    :param num_workers: Number of subprocesses to use for data loading, defaults to 3.
    :type num_workers: int, optional
    :return: DataLoader with padded batches.
    :rtype: torch.utils.data.DataLoader
    """
    pin_memory = batch_size == 1
    # sampler and shuffle are mutually exclusive
    if sampler is None:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lib.data.collate_pad,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    else:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=lib.data.collate_pad,
            num_workers=num_workers,
            pin_memory=pin_memory
        )


def single_batch_loader(dataset, shuffle=True, sampler=None, n_workers: int = 5) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader that loads a single batch at a time.

    :param dataset: The dataset to load data from.
    :type dataset: lib.data.ImageDataset
    :param shuffle: Whether to shuffle the data, defaults to True.
    :type shuffle: bool, optional
    :param sampler: Sampler for data loading, mutually exclusive with shuffle, defaults to None.
    :type sampler: Sampler, optional
    :param n_workers: Number of subprocesses to use for data loading, defaults to 5.
    :type n_workers: int, optional
    :return: DataLoader that loads a single batch at a time.
    :rtype: torch.utils.data.DataLoader
    """
    return torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=1,
            shuffle=shuffle,
            num_workers=n_workers,
            pin_memory=True
        )


def resnet_preprocessor(image: np.ndarray) -> torch.Tensor:
    """
    Preprocess an image for a ResNet model.

    :param image: The input image as a NumPy array.
    :type image: np.ndarray
    :return: Preprocessed image as a Torch tensor.
    :rtype: torch.Tensor
    """
    preprocess = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    processed_image = preprocess(image)

    return processed_image


def image_read_func(image_path: str) -> np.ndarray:
    """
    Read an image from a file path.

    :param image_path: Path to the image file.
    :type image_path: str
    :return: The read image as a NumPy array.
    :rtype: np.ndarray
    """
    return imageio.imread(image_path, pilmode='RGB')
