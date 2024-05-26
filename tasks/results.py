import matplotlib.pyplot as plt
import sklearn.metrics
import numpy as np
import seaborn as sns

import lib.torch_train_eval


def count_misclassifications(label_history, encodings):
    misclassifications = []

    for history in label_history:
        misclassified_count = 0

        for file_path, class_id in history:
            # Extract class name from file path
            class_name = file_path.split('/')[-2]

            # Map class name to class ID using encodings dictionary
            predicted_class_id = [k for k, v in encodings.items() if v == class_name][0]

            # Check if class IDs are different
            if predicted_class_id != class_id:
                misclassified_count += 1

        misclassifications.append(misclassified_count)

    return misclassifications


def plot_label_history(label_history, encodings):
    misclassifications = count_misclassifications(label_history, encodings)
    label_counts = [len(x) for x in label_history]

    data_label = {
        'Sampling period': list(range(len(label_counts))) * 2,
        'Count': label_counts + misclassifications,
        'Type': ['Total samples'] * len(label_counts) + ['Misclassified samples'] * len(misclassifications)
    }

    sns.lineplot(x='Sampling period', y='Count', hue='Type', data=data_label, marker='o')

    # y-ticks integers only
    plt.gca().yaxis.get_major_locator().set_params(integer=True)

    plt.xlabel("Sampling period")
    plt.ylabel("Pseudolabeled samples selected")
    plt.title("Label History")
    plt.legend(title='Type')

    plt.grid(True)
    plt.show()


def learning_curves_loss(history) -> None:
    epochs = np.array(range(len(history["train_loss"])))

    data_loss = {
        'Epoch': np.concatenate([epochs, epochs]),
        'Loss': np.concatenate([history["train_loss"], history["val_loss"]]),
        'Type': ['Training'] * len(history["train_loss"]) + ['Validation'] * len(history["val_loss"])
    }

    sns.lineplot(x='Epoch', y='Loss', hue='Type', data=data_loss, marker='o')

    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Learning Curves - Loss")
    plt.legend(title='Type')

    # Adding gridlines
    plt.grid(True)

    plt.show()


def learning_curves_accuracy(history) -> None:
    epochs = np.array(range(len(history["train_acc"])))

    data_acc = {
        'Epoch': np.concatenate([epochs, epochs]),
        'Accuracy': np.concatenate([history["train_acc"], history["val_acc"]]),
        'Type': ['Training'] * len(history["train_acc"]) + ['Validation'] * len(history["val_acc"])
    }

    sns.lineplot(x='Epoch', y='Accuracy', hue='Type', data=data_acc, marker='o')

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Learning Curves - Accuracy")
    plt.legend(title='Type')

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
