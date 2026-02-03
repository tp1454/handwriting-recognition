"""
Tests for loss functions.

Tests cross-entropy, contrastive, and triplet losses.
"""
import pytest
import torch
import torch.nn.functional as F
import numpy as np


class TestCrossEntropyLoss:
    """Tests for cross-entropy loss."""

    def test_loss_positive(self, sample_batch, sample_labels):
        """Test that loss is positive."""
        from src.models.cnn_classifier import CNNClassifier
        
        model = CNNClassifier(num_classes=62)
        output = model(sample_batch)
        loss = F.cross_entropy(output, sample_labels)
        
        assert loss.item() > 0

    def test_perfect_prediction_low_loss(self):
        """Test that perfect predictions have low loss."""
        # Create perfect logits (very high for correct class)
        logits = torch.zeros(4, 10)
        labels = torch.tensor([0, 1, 2, 3])
        for i in range(4):
            logits[i, labels[i]] = 100.0
        
        loss = F.cross_entropy(logits, labels)
        assert loss.item() < 0.01

    def test_random_prediction_higher_loss(self):
        """Test that random predictions have higher loss."""
        logits = torch.zeros(4, 10)  # Uniform predictions
        labels = torch.tensor([0, 1, 2, 3])
        
        loss = F.cross_entropy(logits, labels)
        # -log(1/10) ≈ 2.3
        assert loss.item() > 2.0


class TestContrastiveLoss:
    """Tests for contrastive loss."""

    def test_same_class_close_low_loss(self):
        """Test that same class with close embeddings has low loss."""
        from src.training.losses import ContrastiveLoss
        
        loss_fn = ContrastiveLoss(margin=1.0)
        emb1 = torch.tensor([[1.0, 0.0]])
        emb2 = torch.tensor([[1.0, 0.1]])
        label = torch.tensor([1])  # Same class
        
        loss = loss_fn(emb1, emb2, label)
        assert loss.item() < 0.1

    def test_different_class_close_high_loss(self):
        """Test that different class with close embeddings has high loss."""
        from src.training.losses import ContrastiveLoss
        
        loss_fn = ContrastiveLoss(margin=1.0)
        emb1 = torch.tensor([[1.0, 0.0]])
        emb2 = torch.tensor([[1.0, 0.1]])
        label = torch.tensor([0])  # Different class
        
        loss = loss_fn(emb1, emb2, label)
        assert loss.item() > 0.5

    def test_different_class_far_low_loss(self):
        """Test that different class with far embeddings has low loss."""
        from src.training.losses import ContrastiveLoss
        
        loss_fn = ContrastiveLoss(margin=1.0)
        emb1 = torch.tensor([[0.0, 0.0]])
        emb2 = torch.tensor([[2.0, 0.0]])
        label = torch.tensor([0])  # Different class
        
        loss = loss_fn(emb1, emb2, label)
        # Distance > margin, so loss should be ~0
        assert loss.item() < 0.1

    def test_margin_effect(self):
        """Test that margin affects the loss."""
        from src.training.losses import ContrastiveLoss
        
        emb1 = torch.tensor([[0.0, 0.0]])
        emb2 = torch.tensor([[0.5, 0.0]])
        label = torch.tensor([0])  # Different class
        
        loss_small_margin = ContrastiveLoss(margin=0.3)(emb1, emb2, label)
        loss_large_margin = ContrastiveLoss(margin=2.0)(emb1, emb2, label)
        
        # Larger margin should give higher loss
        assert loss_large_margin.item() > loss_small_margin.item()

    def test_gradient_flow(self):
        """Test that gradients flow through contrastive loss."""
        from src.training.losses import ContrastiveLoss
        
        loss_fn = ContrastiveLoss(margin=1.0)
        emb1 = torch.tensor([[1.0, 0.0]], requires_grad=True)
        emb2 = torch.tensor([[1.0, 0.5]], requires_grad=True)
        label = torch.tensor([1])
        
        loss = loss_fn(emb1, emb2, label)
        loss.backward()
        
        assert emb1.grad is not None
        assert emb2.grad is not None


class TestTripletLoss:
    """Tests for triplet loss."""

    def test_easy_triplet_low_loss(self):
        """Test that easy triplets have low loss."""
        from src.training.losses import TripletLoss
        
        loss_fn = TripletLoss(margin=1.0)
        anchor = torch.tensor([[0.0, 0.0]])
        positive = torch.tensor([[0.1, 0.0]])  # Close
        negative = torch.tensor([[5.0, 0.0]])  # Far
        
        loss = loss_fn(anchor, positive, negative)
        assert loss.item() < 0.01

    def test_hard_triplet_high_loss(self):
        """Test that hard triplets have high loss."""
        from src.training.losses import TripletLoss
        
        loss_fn = TripletLoss(margin=1.0)
        anchor = torch.tensor([[0.0, 0.0]])
        positive = torch.tensor([[2.0, 0.0]])  # Far
        negative = torch.tensor([[0.1, 0.0]])  # Close
        
        loss = loss_fn(anchor, positive, negative)
        assert loss.item() > 0.5

    def test_triplet_margin_effect(self):
        """Test margin effect on triplet loss."""
        from src.training.losses import TripletLoss
        
        anchor = torch.tensor([[0.0, 0.0]])
        positive = torch.tensor([[0.5, 0.0]])
        negative = torch.tensor([[1.0, 0.0]])
        
        loss_small = TripletLoss(margin=0.1)(anchor, positive, negative)
        loss_large = TripletLoss(margin=2.0)(anchor, positive, negative)
        
        assert loss_large.item() > loss_small.item()

    def test_batch_triplet_loss(self):
        """Test triplet loss with batch inputs."""
        from src.training.losses import TripletLoss
        
        loss_fn = TripletLoss(margin=1.0)
        anchor = torch.randn(8, 128)
        positive = torch.randn(8, 128)
        negative = torch.randn(8, 128)
        
        loss = loss_fn(anchor, positive, negative)
        assert loss.shape == ()  # Scalar output
        assert loss.item() >= 0  # Loss should be non-negative


class TestLossReduction:
    """Tests for loss reduction modes."""

    def test_contrastive_mean_reduction(self):
        """Test mean reduction for contrastive loss."""
        from src.training.losses import ContrastiveLoss
        
        loss_fn = ContrastiveLoss(margin=1.0, reduction='mean')
        emb1 = torch.randn(4, 128)
        emb2 = torch.randn(4, 128)
        labels = torch.randint(0, 2, (4,))
        
        loss = loss_fn(emb1, emb2, labels)
        assert loss.shape == ()

    def test_contrastive_sum_reduction(self):
        """Test sum reduction for contrastive loss."""
        from src.training.losses import ContrastiveLoss
        
        loss_fn = ContrastiveLoss(margin=1.0, reduction='sum')
        emb1 = torch.randn(4, 128)
        emb2 = torch.randn(4, 128)
        labels = torch.randint(0, 2, (4,))
        
        loss = loss_fn(emb1, emb2, labels)
        assert loss.shape == ()

    def test_contrastive_none_reduction(self):
        """Test no reduction for contrastive loss."""
        from src.training.losses import ContrastiveLoss
        
        loss_fn = ContrastiveLoss(margin=1.0, reduction='none')
        emb1 = torch.randn(4, 128)
        emb2 = torch.randn(4, 128)
        labels = torch.randint(0, 2, (4,))
        
        loss = loss_fn(emb1, emb2, labels)
        assert loss.shape == (4,)
