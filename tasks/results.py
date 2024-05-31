import matplotlib.pyplot as plt
import sklearn.metrics
import numpy as np
import seaborn as sns
import torch

import tasks.utils
import lib.torch_train_eval

import os


def count_misclassifications(label_history, encodings):
    misclassifications = []

    for history in label_history:
        misclassified_count = 0

        for file_path, class_id in history:
            # Extract class name from file path
            class_name = file_path.split("/")[-2]

            # Map class name to class ID using encodings dictionary
            predicted_class_id = [
                k for k, v in encodings.items() if v == class_name
            ][0]

            # Check if class IDs are different
            if predicted_class_id != class_id:
                misclassified_count += 1

        misclassifications.append(misclassified_count)

    return misclassifications


def plot_label_history(label_history, encodings, ax=None):
    misclassifications = count_misclassifications(label_history, encodings)
    label_counts = [len(x) for x in label_history]

    data_label = {
        "Sampling period": list(range(len(label_counts))) * 2,
        "Count": label_counts + misclassifications,
        "Type": ["Total samples"] * len(label_counts)
        + ["Misclassified samples"] * len(misclassifications),
    }

    if ax is None:
        ax = plt.gca()

    sns.lineplot(
        x="Sampling period",
        y="Count",
        hue="Type",
        data=data_label,
        marker="o",
        ax=ax,
    )

    # y-ticks integers only
    ax.yaxis.get_major_locator().set_params(integer=True)

    ax.set_xlabel("Sampling period")
    ax.set_ylabel("Pseudolabeled samples selected")
    ax.set_title("Label History")
    ax.legend(title="Type")

    ax.grid(True)


def learning_curves_loss(history) -> None:
    epochs = np.array(range(len(history["train_loss"])))

    data_loss = {
        "Epoch": np.concatenate([epochs, epochs]),
        "Loss": np.concatenate([history["train_loss"], history["val_loss"]]),
        "Type": ["Training"] * len(history["train_loss"])
        + ["Validation"] * len(history["val_loss"]),
    }

    sns.lineplot(x="Epoch", y="Loss", hue="Type", data=data_loss, marker="o")

    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Learning Curves - Loss")
    plt.legend(title="Type")

    # Adding gridlines
    plt.grid(True)

    plt.show()


def learning_curves_accuracy(history) -> None:
    epochs = np.array(range(len(history["train_acc"])))

    data_acc = {
        "Epoch": np.concatenate([epochs, epochs]),
        "Accuracy": np.concatenate([history["train_acc"], history["val_acc"]]),
        "Type": ["Training"] * len(history["train_acc"])
        + ["Validation"] * len(history["val_acc"]),
    }

    sns.lineplot(x="Epoch", y="Accuracy", hue="Type", data=data_acc, marker="o")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Learning Curves - Accuracy")
    plt.legend(title="Type")

    # Adding gridlines
    plt.grid(True)

    plt.show()


def classification_results(model, dataloader, class_names: list[str], device: str):
    actual, predicted = lib.torch_train_eval.test(model, dataloader, device)

    print(
        sklearn.metrics.classification_report(
            actual,
            predicted,
            zero_division=0,
            target_names=class_names,
            labels=np.arange(0, len(class_names), 1),
        )
    )

    cf_matrix = sklearn.metrics.confusion_matrix(
        actual, predicted, labels=np.arange(0, len(class_names), 1)
    )
    display = sklearn.metrics.ConfusionMatrixDisplay(
        confusion_matrix=cf_matrix, display_labels=class_names
    )
    display.plot()
    plt.xticks(rotation=90)
    plt.show()


def plot_classification_matrices(
    model: torch.nn.Module,
    model_dirs: list[tuple[str, str]],
    target_test_loader: torch.utils.data.DataLoader,
    device: str,
    rows: int,
    cols: int,
    figsize: tuple[float, float] = (12, 12),
    save_path: str = None,
):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.ravel()

    for idx, (model_dir, title) in enumerate(model_dirs):
        model_path = os.path.join(model_dir, "model.pt")
        tasks.utils.try_load_weights(model, model_path)
        actual, predicted = lib.torch_train_eval.test(
            model, target_test_loader, device
        )
        cm = sklearn.metrics.confusion_matrix(actual, predicted)

        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt="d", ax=axes[idx], cmap="Blues")
        axes[idx].set_title(title)
        axes[idx].set_xlabel("Predicted labels")
        axes[idx].set_ylabel("True labels")

    fig.suptitle("Model performance on target domain dataset")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Figured saved to " + save_path)

    plt.show()


def plot_label_history_grid(
    model_dirs,
    encodings,
    rows,
    cols,
    figsize: tuple[float, float] = (12, 12),
    save_path=None,
):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.ravel()

    for idx, (model_dir, title) in enumerate(model_dirs):
        res = tasks.utils.load_trained_model(torch.nn.Module(), model_dir)
        label_history = res["label_history"]
        plot_label_history(label_history, encodings, ax=axes[idx])
        axes[idx].set_title(title)

    fig.suptitle("Label History of Models")
    plt.tight_layout()

    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()
