import numpy as np
import itertools
import matplotlib.pyplot as plt


def show():
    plt.show()


def plot_cm_with_labels(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(7, 7))
    scaled_cm = cm / np.sum(cm, axis=-1, keepdims=True)
    plt.imshow(scaled_cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix
    labels = np.around(
        cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100., decimals=2)

    # Use white text if squares are dark; otherwise black
    threshold = scaled_cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if scaled_cm[i, j] > threshold else "black"
        plt.text(
            j, i, str(labels[i, j]) + " %", horizontalalignment="center",
            color=color)
        plt.text(
            j, i+0.1, cm[i, j], horizontalalignment="center",
            color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
