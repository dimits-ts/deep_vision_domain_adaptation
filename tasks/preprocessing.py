import torch
import torchvision
import imageio.v2 as imageio
from torchvision.transforms import v2
import numpy as np
import PIL

import lib.data

def create_padded_dataloader(
    dataset: lib.data.ImageDataset,
    shuffle: bool = True,
    sampler = None,
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


def crop_and_resize(image: np.ndarray, max_vram_size: int = 2000, cropped_size: int = 512) -> np.ndarray:
    """
    Crops and resizes an image to ensure it fits within GPU's VRAM.

    :param cropped_size: A where AxA are the new dimensions of the image
    :param numpy.ndarray image: The input image.
    :param int max_vram_size: Maximum allowed size in MB for GPU's VRAM.
    :return: Cropped and resized image.
    :rtype: numpy.ndarray
    """
    # Convert numpy array to PIL Image
    image_pil = PIL.Image.fromarray(image)

    # Initial size estimation
    original_size = np.prod(image.shape) / (1024 * 1024)  # Convert to MB

    if original_size <= max_vram_size:
        # No need for cropping or resizing
        return image_pil
    else:
        print("Resized image of size ", original_size)
        resize_transform = v2.Compose([
            v2.Resize((cropped_size, cropped_size)),  # Resize to a smaller size
        ])
        image_resized = resize_transform(image_pil)
        return image_resized


def resnet_preprocessor(image: np.ndarray, max_vram_size: int = 2000, cropped_size: int = 512) -> PIL.Image:
    """
    Preprocesses an image for ResNet model.

    :param cropped_size:
    :param max_vram_size: Maximum allowed size in MB for GPU's VRAM.
    :param numpy.ndarray image: The input image.
    :return: Preprocessed image.
    :rtype: numpy.ndarray
    """
    preprocess = v2.Compose([
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Crop and resize image if necessary
    cropped_image = crop_and_resize(image, max_vram_size=max_vram_size, cropped_size=cropped_size)

    # Apply preprocessing transformations
    processed_image = preprocess(cropped_image)

    return processed_image


def image_read_func(image_path):
    return imageio.imread(image_path, pilmode='RGB')

