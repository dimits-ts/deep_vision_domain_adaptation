import torch

import pickle
import os
from typing import Any


def get_model(device: str, 
              replace_fc_layer: bool=False, 
              num_classes: int=0, 
              use_default_weights: bool=True) -> torch.nn.Module:
    """
    Get a ResNet-18 model from PyTorch's model hub.

    :param device: Device to load the model onto (e.g., 'cpu' or 'cuda').
    :type device: str
    :param replace_fc_layer: Whether to replace the final fully connected layer, defaults to False.
    :type replace_fc_layer: bool, optional
    :param num_classes: Number of output classes for the new fully connected layer, required if replace_fc_layer is True.
    :type num_classes: int, optional
    :param use_default_weights: Whether to use the default pretrained weights, defaults to True.
    :type use_default_weights: bool, optional
    :return: The ResNet-18 model.
    :rtype: torch.nn.Module
    """
    weights = "DEFAULT" if use_default_weights else None
    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", weights=weights).to(device)
    if replace_fc_layer:
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes).to(device)
    return model


def try_load_weights(model: torch.nn.Module, weights_path: str) -> torch.nn.Module:
    """
    Try to load weights into the model from a given path.

    :param model: The model to load weights into.
    :type model: torch.nn.Module
    :param weights_path: Path to the weights file.
    :type weights_path: str
    :return: The model with loaded weights.
    :rtype: torch.nn.Module
    """
    try:
        model.load_state_dict(torch.load(weights_path))
    except Exception as e:
        print("No weights found in path ", weights_path, "\n", e)
    return model


def try_load_history(history_path: str) -> Any:
    """
    Try to load training history from a given path.

    :param history_path: Path to the history file.
    :type history_path: str
    :return: The loaded history, or None if loading failed.
    :rtype: Any
    """
    try:
        with open(history_path, 'rb') as handle:
            history = pickle.load(handle)
    except Exception:
        print("No history found in path ", history_path)
        history = None

    return history


def load_trained_model(bare_model: torch.nn.Module, training_dir: str) -> dict[str, Any]:
    """
    Load a trained model and its associated training histories from a directory.

    :param bare_model: The untrained base model.
    :type bare_model: torch.nn.Module
    :param training_dir: Directory containing the model weights and history files.
    :type training_dir: str
    :return: Dictionary containing the model and its training histories.
    :rtype: dict[str, Any]
    """
    model = try_load_weights(
        bare_model, os.path.join(training_dir, "model.pt")
    )
    source_history = try_load_history(
        os.path.join(training_dir, "source_history.pickle")
    )
    target_history = try_load_history(
        os.path.join(training_dir, "target_history.pickle")
    )
    label_history = try_load_history(
        os.path.join(training_dir, "label_history.pickle")
    )

    return {
        "model": model,
        "source_history": source_history,
        "target_history": target_history,
        "label_history": label_history,
    }
