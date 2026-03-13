"""
Tests for training metrics.

Tests accuracy, confusion matrix, and similarity metrics.
"""

import numpy as np
import pytest
import torch


class TestAccuracyMetric:
    """Tests for accuracy computation."""

    def test_perfect_accuracy(self):
        """Test 100% accuracy case."""
        from src.training.metrics import accuracy

        predictions = torch.tensor([0, 1, 2, 3])
        labels = torch.tensor([0, 1, 2, 3])

        acc = accuracy(predictions, labels)
        assert acc == pytest.approx(1.0)

    def test_zero_accuracy(self):
        """Test 0% accuracy case."""
        from src.training.metrics import accuracy

        predictions = torch.tensor([1, 2, 3, 0])
        labels = torch.tensor([0, 1, 2, 3])

        acc = accuracy(predictions, labels)
        assert acc == pytest.approx(0.0)

    def test_partial_accuracy(self):
        """Test partial accuracy case."""
        from src.training.metrics import accuracy

        predictions = torch.tensor([0, 1, 0, 0])
        labels = torch.tensor([0, 1, 2, 3])

        acc = accuracy(predictions, labels)
        assert acc == pytest.approx(0.5)

    def test_accuracy_from_logits(self):
        """Test accuracy computation from logits."""
        from src.training.metrics import accuracy_from_logits

        logits = torch.zeros(4, 10)
        logits[0, 0] = 10.0
        logits[1, 1] = 10.0
        logits[2, 2] = 10.0
        logits[3, 3] = 10.0
        labels = torch.tensor([0, 1, 2, 3])

        acc = accuracy_from_logits(logits, labels)
        assert acc == pytest.approx(1.0)


class TestTopKAccuracy:
    """Tests for top-k accuracy."""

    def test_top1_accuracy(self):
        """Test top-1 accuracy."""
        from src.training.metrics import top_k_accuracy

        logits = torch.tensor(
            [
                [10.0, 1.0, 1.0],
                [1.0, 10.0, 1.0],
            ]
        )
        labels = torch.tensor([0, 1])

        acc = top_k_accuracy(logits, labels, k=1)
        assert acc == pytest.approx(1.0)

    def test_top3_accuracy(self):
        """Test top-3 accuracy."""
        from src.training.metrics import top_k_accuracy

        logits = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],  # Top-3: 4, 3, 2
                [5.0, 4.0, 3.0, 2.0, 1.0],  # Top-3: 0, 1, 2
            ]
        )
        labels = torch.tensor([3, 2])  # Both in top-3

        acc = top_k_accuracy(logits, labels, k=3)
        assert acc == pytest.approx(1.0)


class TestConfusionMatrix:
    """Tests for confusion matrix computation."""

    def test_confusion_matrix_shape(self):
        """Test confusion matrix has correct shape."""
        from src.training.metrics import confusion_matrix

        predictions = torch.randint(0, 10, (100,))
        labels = torch.randint(0, 10, (100,))

        cm = confusion_matrix(predictions, labels, num_classes=10)
        assert cm.shape == (10, 10)

    def test_confusion_matrix_diagonal(self):
        """Test diagonal for perfect predictions."""
        from src.training.metrics import confusion_matrix

        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        predictions = labels.clone()

        cm = confusion_matrix(predictions, labels, num_classes=3)

        # Diagonal should be 2, off-diagonal should be 0
        assert cm[0, 0] == 2
        assert cm[1, 1] == 2
        assert cm[2, 2] == 2
        assert cm.sum() == 6

    def test_confusion_matrix_off_diagonal(self):
        """Test off-diagonal for misclassifications."""
        from src.training.metrics import confusion_matrix

        labels = torch.tensor([0, 0, 0])
        predictions = torch.tensor([1, 1, 1])  # All misclassified as class 1

        cm = confusion_matrix(predictions, labels, num_classes=3)

        assert cm[0, 1] == 3  # 3 samples of class 0 predicted as class 1
        assert cm[0, 0] == 0


class TestCosineSimilarityMetric:
    """Tests for cosine similarity metric."""

    def test_identical_embeddings(self):
        """Test similarity of identical embeddings."""
        from src.training.metrics import cosine_similarity_metric

        emb = torch.randn(4, 128)

        similarity = cosine_similarity_metric(emb, emb)
        np.testing.assert_array_almost_equal(similarity.numpy(), np.ones(4), decimal=5)

    def test_orthogonal_embeddings(self):
        """Test similarity of orthogonal embeddings."""
        from src.training.metrics import cosine_similarity_metric

        emb1 = torch.tensor([[1.0, 0.0, 0.0]])
        emb2 = torch.tensor([[0.0, 1.0, 0.0]])

        similarity = cosine_similarity_metric(emb1, emb2)
        assert similarity.item() == pytest.approx(0.0, abs=0.001)

    def test_opposite_embeddings(self):
        """Test similarity of opposite embeddings."""
        from src.training.metrics import cosine_similarity_metric

        emb1 = torch.tensor([[1.0, 0.0]])
        emb2 = torch.tensor([[-1.0, 0.0]])

        similarity = cosine_similarity_metric(emb1, emb2)
        assert similarity.item() == pytest.approx(-1.0, abs=0.001)


class TestPrecisionRecall:
    """Tests for precision and recall metrics."""

    def test_precision(self):
        """Test precision computation."""
        from src.training.metrics import precision_per_class

        predictions = torch.tensor([0, 0, 0, 1, 1])
        labels = torch.tensor([0, 0, 1, 1, 1])

        precision = precision_per_class(predictions, labels, num_classes=2)

        # Class 0: 2 correct out of 3 predicted = 2/3
        # Class 1: 2 correct out of 2 predicted = 1.0
        assert precision[0] == pytest.approx(2 / 3, abs=0.01)
        assert precision[1] == pytest.approx(1.0, abs=0.01)

    def test_recall(self):
        """Test recall computation."""
        from src.training.metrics import recall_per_class

        predictions = torch.tensor([0, 0, 0, 1, 1])
        labels = torch.tensor([0, 0, 1, 1, 1])

        recall = recall_per_class(predictions, labels, num_classes=2)

        # Class 0: 2 correct out of 2 actual = 1.0
        # Class 1: 2 correct out of 3 actual = 2/3
        assert recall[0] == pytest.approx(1.0, abs=0.01)
        assert recall[1] == pytest.approx(2 / 3, abs=0.01)
