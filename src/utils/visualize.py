import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize


# Dummy history
class DummyHistory:
    def __init__(self):
        self.history = {
            "loss": [0.9, 0.7, 0.5, 0.4, 0.3],
            "val_loss": [1.0, 0.8, 0.6, 0.65, 0.7],
            "accuracy": [0.6, 0.7, 0.8, 0.85, 0.9],
            "val_accuracy": [0.55, 0.65, 0.75, 0.73, 0.72],
        }


history = DummyHistory()

# Dummy labels (10 lớp: 0–9)
y_true = np.random.randint(0, 10, 100)
y_pred = np.random.randint(0, 10, 100)

# Dummy probabilities cho ROC
y_score = np.random.rand(100, 10)
y_score = y_score / y_score.sum(axis=1, keepdims=True)


# loss curve plot
def plot_loss(history):
    plt.figure()
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.show()


# roc curve plot
def plot_roc(y_true, y_score):
    n_classes = y_score.shape[1]
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    plt.figure()

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        plt.plot(
            fpr, tpr, label=f"Class {i} (AUC={auc(fpr, tpr):.2f})"
        )

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Multi-class)")
    plt.legend()
    plt.show()


# ma trận nhầm lẫn
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


# bar graph plot
def plot_bar(y_true):
    unique, counts = np.unique(y_true, return_counts=True)

    plt.figure()
    plt.bar(unique, counts)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Distribution")
    plt.show()


plot_loss(history)
plot_roc(y_true, y_score)
plot_confusion_matrix(y_true, y_pred)
plot_bar(y_true)
