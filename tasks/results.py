import matplotlib.pyplot as plt
import sklearn.metrics
import numpy as np

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
    plt.plot(label_counts)
    plt.plot(misclassifications)
    plt.show()


def learning_curves_loss(history) -> None:
    plt.plot(np.array(range(len(history["train_loss"]))), history["train_loss"], label="Training")
    plt.plot(np.array(range(len(history["val_loss"]))), history["val_loss"], label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.legend()
    plt.show()


def learning_curves_accuracy(history) -> None:
    plt.plot(np.array(range(len(history["train_acc"]))), history["train_acc"], label="Training")
    plt.plot(np.array(range(len(history["val_acc"]))), history["val_acc"], label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
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
