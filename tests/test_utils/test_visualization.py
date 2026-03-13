"""
Tests for visualization utilities.

Tests plotting and visualization functions.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

matplotlib.use("Agg")  # Non-interactive backend for testing


class TestLossPlotting:
    """Tests for loss curve plotting."""

    def test_plot_training_loss(self, tmp_path):
        """Test plotting training loss curve."""
        from src.utils.visualization import plot_loss_curve

        losses = [1.0, 0.8, 0.6, 0.5, 0.4]

        fig = plot_loss_curve(losses, title="Training Loss")

        assert fig is not None
        plt.close(fig)

    def test_plot_train_val_loss(self, tmp_path):
        """Test plotting train and validation loss."""
        from src.utils.visualization import plot_train_val_loss

        train_losses = [1.0, 0.8, 0.6, 0.5, 0.4]
        val_losses = [1.1, 0.9, 0.7, 0.6, 0.55]

        fig = plot_train_val_loss(train_losses, val_losses)

        assert fig is not None
        plt.close(fig)

    def test_save_loss_plot(self, tmp_path):
        """Test saving loss plot to file."""
        from src.utils.visualization import plot_loss_curve, save_figure

        losses = [1.0, 0.8, 0.6, 0.5, 0.4]
        fig = plot_loss_curve(losses)

        output_path = tmp_path / "loss_curve.png"
        save_figure(fig, output_path)

        assert output_path.exists()
        plt.close(fig)


class TestSamplePlotting:
    """Tests for sample image plotting."""

    def test_plot_single_sample(self, sample_numpy_image):
        """Test plotting a single sample."""
        from src.utils.visualization import plot_sample

        fig = plot_sample(sample_numpy_image, label="A")

        assert fig is not None
        plt.close(fig)

    def test_plot_sample_grid(self, sample_batch_numpy):
        """Test plotting a grid of samples."""
        from src.utils.visualization import plot_sample_grid

        labels = ["0", "1", "2", "3", "4", "5", "6", "7"]

        fig = plot_sample_grid(sample_batch_numpy, labels, grid_size=(2, 4))

        assert fig is not None
        plt.close(fig)

    def test_plot_augmentation_comparison(self, sample_numpy_image):
        """Test plotting original vs augmented."""
        from src.utils.visualization import plot_augmentation_comparison

        original = sample_numpy_image
        augmented = sample_numpy_image + np.random.randn(*sample_numpy_image.shape) * 10

        fig = plot_augmentation_comparison(original, augmented)

        assert fig is not None
        plt.close(fig)


class TestConfusionMatrixPlotting:
    """Tests for confusion matrix visualization."""

    def test_plot_confusion_matrix(self):
        """Test plotting confusion matrix."""
        from src.utils.visualization import plot_confusion_matrix

        cm = np.random.randint(0, 100, (10, 10))
        classes = [str(i) for i in range(10)]

        fig = plot_confusion_matrix(cm, classes)

        assert fig is not None
        plt.close(fig)

    def test_plot_normalized_confusion_matrix(self):
        """Test plotting normalized confusion matrix."""
        from src.utils.visualization import plot_confusion_matrix

        cm = np.random.randint(0, 100, (10, 10))
        classes = [str(i) for i in range(10)]

        fig = plot_confusion_matrix(cm, classes, normalize=True)

        assert fig is not None
        plt.close(fig)


class TestEmbeddingVisualization:
    """Tests for embedding visualization."""

    def test_plot_embeddings_tsne(self, sample_embeddings, sample_labels):
        """Test t-SNE visualization of embeddings."""
        from src.utils.visualization import plot_embeddings_tsne

        # Need at least as many samples as perplexity for t-SNE
        embeddings = torch.randn(50, 128)
        labels = torch.randint(0, 10, (50,))

        fig = plot_embeddings_tsne(embeddings.numpy(), labels.numpy())

        assert fig is not None
        plt.close(fig)

    def test_plot_embeddings_pca(self, sample_embeddings, sample_labels):
        """Test PCA visualization of embeddings."""
        from src.utils.visualization import plot_embeddings_pca

        fig = plot_embeddings_pca(sample_embeddings.numpy(), sample_labels.numpy())

        assert fig is not None
        plt.close(fig)


class TestMetricsPlotting:
    """Tests for metrics visualization."""

    def test_plot_accuracy_curve(self):
        """Test plotting accuracy over epochs."""
        from src.utils.visualization import plot_accuracy_curve

        train_acc = [0.5, 0.6, 0.7, 0.8, 0.85]
        val_acc = [0.45, 0.55, 0.65, 0.75, 0.80]

        fig = plot_accuracy_curve(train_acc, val_acc)

        assert fig is not None
        plt.close(fig)

    def test_plot_learning_rate_schedule(self):
        """Test plotting learning rate schedule."""
        from src.utils.visualization import plot_lr_schedule

        lrs = [0.001, 0.001, 0.0005, 0.0005, 0.00025]

        fig = plot_lr_schedule(lrs)

        assert fig is not None
        plt.close(fig)


class TestPredictionVisualization:
    """Tests for prediction visualization."""

    def test_plot_prediction(self, sample_numpy_image):
        """Test plotting image with prediction."""
        from src.utils.visualization import plot_prediction

        fig = plot_prediction(sample_numpy_image, predicted="A", confidence=0.95, actual="A")

        assert fig is not None
        plt.close(fig)

    def test_plot_top_k_predictions(self, sample_numpy_image):
        """Test plotting top-k predictions."""
        from src.utils.visualization import plot_top_k_predictions

        predictions = [
            {"character": "A", "confidence": 0.8},
            {"character": "4", "confidence": 0.1},
            {"character": "H", "confidence": 0.05},
        ]

        fig = plot_top_k_predictions(sample_numpy_image, predictions)

        assert fig is not None
        plt.close(fig)
