import torch

import pickle
import os
from typing import Any


def try_load_weights(model, weights_path: str):
    try:
        model.load_state_dict(torch.load(weights_path))
    except Exception as e:
        print("No weights found in path ", weights_path, "\n", e)
    return model


def try_load_history(history_path):
    try:
        with open(history_path, 'rb') as handle:
            history = pickle.load(handle)
    except:
        print("No history found in path ", history_path)
        history = None

    return history


def load_trained_model(bare_model: torch.nn.Module, training_dir: str) -> dict[str, Any]:
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
