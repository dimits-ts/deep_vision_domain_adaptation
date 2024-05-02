from torch.utils.data.sampler import SubsetRandomSampler
import torch
import numpy as np
from tqdm.auto import tqdm
import sklearn.preprocessing

from typing import Callable
import os


class ImageDataset(torch.utils.data.Dataset):
    """
    Lazily loads images from a root directory.
    Directory is assumed to be of shape "<root>/<class_name>/<instance_file>".
    Allows custom functions for reading, preprocessing each image and setting the label encodings.
    """

    def __init__(
            self,
            parser_func: Callable,
            preprocessing_func: Callable[[np.ndarray], np.ndarray],
            label_encoder=None,
    ):
        """
        Initializes the ImageDataset.

        :param parser_func: Function to parse images.
        :type parser_func: Callable, optional
        :param preprocessing_func: Function to preprocess images.
        :type preprocessing_func: Callable[[numpy.ndarray], numpy.ndarray], optional
        :param label_encoder: Encoder for label encoding.
        :type label_encoder: sklearn.preprocessing.LabelEncoder or None, optional
        """
        self.parser_func = parser_func
        self.preprocessing_func = preprocessing_func
        self.label_encoder = label_encoder
        self.samples = []
        # to keep track of registered files quickly
        self.paths = set()

    def load_from_directory(self, data_dir: str):
        """
        Load all image samples from a directory.
        :param str data_dir: Root directory containing the dataset.
        :return:
        """
        self._load_dataset_paths(data_dir)

    def add(self, image_path: str, encoded_label: int) -> None:
        self._insert_sample(image_path, encoded_label)

    def remove(self, image_path: str):
        if image_path in self.paths:
            self.paths.remove(image_path)
            self.samples = [sample for sample in self.samples if sample[0] != image_path]
        else:
            print("Warning: Removal failed: could not find ", image_path, " in the dataset.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = self.parser_func(image_path)
        image = self.preprocessing_func(image)

        if not torch.is_tensor(image):
            image = torch.tensor(image)

        return image, label

    def _insert_sample(self, image_path: str, encoded_label: int):
        if image_path not in self.paths:
            self.paths.add(image_path)
            self.samples.append((image_path, encoded_label))

    def _load_dataset_paths(self, data_dir: str) -> None:
        """
        Loads paths of images in the dataset.

        :param str data_dir: Root directory containing the dataset.
        :return: List of tuples containing image paths and their corresponding labels.
        :rtype: List[Tuple[str, int]]
        """
        class_names = os.listdir(data_dir)

        if self.label_encoder is None:
            self.label_encoder = sklearn.preprocessing.LabelEncoder()
            self.label_encoder.fit(class_names)

        for class_name in tqdm(class_names):
            class_data_dir = os.path.join(data_dir, class_name)

            for file_name in os.listdir(class_data_dir):
                self._insert_sample(os.path.join(class_data_dir, file_name),
                                    self.label_encoder.transform([class_name])[0])


class UnlabeledImageDataset(ImageDataset):

    def __init__(
            self,
            parser_func: Callable,
            preprocessing_func: Callable[[np.ndarray], np.ndarray],
    ):
        super().__init__(parser_func=parser_func, preprocessing_func=preprocessing_func, label_encoder=None)

    def load_from_image_dataset(self, dataset: ImageDataset):
        self.samples = dataset.samples

    def __getitem__(self, idx):
        # hide label
        image_path, _ = self.samples[idx]
        image = self.parser_func(image_path)
        image = self.preprocessing_func(image)

        if not torch.is_tensor(image):
            image = torch.tensor(image)

        # instead of label return path
        return image, image_path


def collate_pad(batch):
    # Sort the batch by image height in descending order
    batch = sorted(batch, key=lambda x: x[0].shape[1], reverse=True)

    # Get the maximum height and width among all images in the batch
    max_height = max(img.shape[1] for img, _ in batch)
    max_width = max(img.shape[2] for img, _ in batch)

    # Pad each image to match the maximum height and width
    padded_batch = []
    for img, label in batch:
        # Calculate padding sizes
        pad_height = max_height - img.shape[1]
        pad_width = max_width - img.shape[2]

        # Pad the image
        padded_img = torch.nn.functional.pad(img, (0, pad_width, 0, pad_height))

        padded_batch.append((padded_img, label))

    # Stack images and labels into tensors
    images = torch.stack([img for img, _ in padded_batch])
    labels = torch.tensor([label for _, label in padded_batch])

    return images, labels
