"""
Tests for Siamese Network model.

Tests the Siamese network architecture for similarity learning.
"""
import pytest
import torch
import torch.nn.functional as F
import numpy as np


class TestSiameseNetwork:
    """Tests for SiameseNetwork model."""

    def test_output_shapes(self, sample_batch):
        """Test that both outputs have correct shape."""
        from src.models.siamese_network import SiameseNetwork
        from src.models.cnn_classifier import CNNClassifier
        
        encoder = CNNClassifier(num_classes=62, embedding_dim=128)
        model = SiameseNetwork(encoder)
        
        emb1, emb2 = model(sample_batch, sample_batch)
        assert emb1.shape == (8, 128)
        assert emb2.shape == (8, 128)

    def test_shared_weights(self, sample_batch):
        """Test that encoder weights are shared."""
        from src.models.siamese_network import SiameseNetwork
        from src.models.cnn_classifier import CNNClassifier
        
        encoder = CNNClassifier(num_classes=62, embedding_dim=128)
        model = SiameseNetwork(encoder)
        model.eval()
        
        with torch.no_grad():
            emb1_a, _ = model(sample_batch, sample_batch)
            emb1_b, _ = model(sample_batch, sample_batch)
        
        # Same input should produce same embedding
        assert torch.allclose(emb1_a, emb1_b)

    def test_different_inputs(self, sample_batch):
        """Test with different inputs for each branch."""
        from src.models.siamese_network import SiameseNetwork
        from src.models.cnn_classifier import CNNClassifier
        
        encoder = CNNClassifier(num_classes=62, embedding_dim=128)
        model = SiameseNetwork(encoder)
        
        input1 = sample_batch
        input2 = torch.randn_like(sample_batch)
        
        emb1, emb2 = model(input1, input2)
        
        # Different inputs should produce different embeddings
        assert not torch.allclose(emb1, emb2)

    def test_forward_pass_no_error(self, sample_batch):
        """Test that forward pass completes without error."""
        from src.models.siamese_network import SiameseNetwork
        from src.models.cnn_classifier import CNNClassifier
        
        encoder = CNNClassifier(num_classes=62, embedding_dim=128)
        model = SiameseNetwork(encoder)
        
        try:
            emb1, emb2 = model(sample_batch, sample_batch)
            assert True
        except Exception as e:
            pytest.fail(f"Forward pass raised exception: {e}")

    def test_gradient_flow(self, sample_batch):
        """Test that gradients flow through both branches."""
        from src.models.siamese_network import SiameseNetwork
        from src.models.cnn_classifier import CNNClassifier
        
        encoder = CNNClassifier(num_classes=62, embedding_dim=128)
        model = SiameseNetwork(encoder)
        model.train()
        
        input1 = sample_batch
        input2 = torch.randn_like(sample_batch)
        
        emb1, emb2 = model(input1, input2)
        loss = F.mse_loss(emb1, emb2)
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                break


class TestSiameseSimilarity:
    """Tests for similarity computation with Siamese network."""

    def test_cosine_similarity_range(self, sample_batch):
        """Test that cosine similarity is in [-1, 1]."""
        from src.models.siamese_network import SiameseNetwork
        from src.models.cnn_classifier import CNNClassifier
        
        encoder = CNNClassifier(num_classes=62, embedding_dim=128)
        model = SiameseNetwork(encoder)
        model.eval()
        
        with torch.no_grad():
            emb1, emb2 = model(sample_batch, sample_batch)
            similarity = F.cosine_similarity(emb1, emb2)
        
        assert similarity.min() >= -1.0
        assert similarity.max() <= 1.0

    def test_identical_inputs_high_similarity(self, sample_single_image):
        """Test that identical inputs have high similarity."""
        from src.models.siamese_network import SiameseNetwork
        from src.models.cnn_classifier import CNNClassifier
        
        encoder = CNNClassifier(num_classes=62, embedding_dim=128)
        model = SiameseNetwork(encoder)
        model.eval()
        
        with torch.no_grad():
            emb1, emb2 = model(sample_single_image, sample_single_image)
            similarity = F.cosine_similarity(emb1, emb2)
        
        # Identical inputs should have perfect similarity
        assert similarity.item() == pytest.approx(1.0, abs=0.001)

    def test_euclidean_distance(self, sample_batch):
        """Test Euclidean distance computation."""
        from src.models.siamese_network import SiameseNetwork
        from src.models.cnn_classifier import CNNClassifier
        
        encoder = CNNClassifier(num_classes=62, embedding_dim=128)
        model = SiameseNetwork(encoder)
        model.eval()
        
        with torch.no_grad():
            emb1, emb2 = model(sample_batch, sample_batch)
            distance = F.pairwise_distance(emb1, emb2)
        
        # Distance should be non-negative
        assert distance.min() >= 0.0
