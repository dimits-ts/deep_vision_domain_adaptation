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


def single_batch_loader(dataset, shuffle=True, sampler=None, n_workers: int=5):
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
    Preprocesses an image for ResNet model.

    :param numpy.ndarray image: The input image.
    :return: Preprocessed image.
    :rtype: numpy.ndarray
    """
    preprocess = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    processed_image = preprocess(image)

    return processed_image


def image_read_func(image_path):
    return imageio.imread(image_path, pilmode='RGB')
