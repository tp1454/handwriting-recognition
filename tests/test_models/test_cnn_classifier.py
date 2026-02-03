"""
Tests for CNN Classifier model.

Tests the CNN architecture for 62-class character classification.
"""
import pytest
import torch
import torch.nn.functional as F
import numpy as np


class TestCNNClassifier:
    """Tests for CNNClassifier model."""

    def test_output_shape(self, sample_batch):
        """Test that output has correct shape."""
        from src.models.cnn_classifier import CNNClassifier
        
        model = CNNClassifier(num_classes=62)
        output = model(sample_batch)
        assert output.shape == (8, 62)

    def test_single_image_output(self, sample_single_image):
        """Test output for single image."""
        from src.models.cnn_classifier import CNNClassifier
        
        model = CNNClassifier(num_classes=62)
        output = model(sample_single_image)
        assert output.shape == (1, 62)

    def test_softmax_sums_to_one(self, sample_batch):
        """Test that softmax of logits sums to 1."""
        from src.models.cnn_classifier import CNNClassifier
        
        model = CNNClassifier(num_classes=62)
        logits = model(sample_batch)
        probs = F.softmax(logits, dim=1)
        np.testing.assert_array_almost_equal(
            probs.sum(dim=1).detach().numpy(),
            np.ones(8),
            decimal=5
        )

    def test_different_num_classes(self, sample_batch):
        """Test model with different number of classes."""
        from src.models.cnn_classifier import CNNClassifier
        
        model = CNNClassifier(num_classes=10)
        output = model(sample_batch)
        assert output.shape == (8, 10)

    def test_forward_pass_no_error(self, sample_batch):
        """Test that forward pass completes without error."""
        from src.models.cnn_classifier import CNNClassifier
        
        model = CNNClassifier(num_classes=62)
        try:
            output = model(sample_batch)
            assert True
        except Exception as e:
            pytest.fail(f"Forward pass raised exception: {e}")

    def test_model_eval_mode(self, sample_batch):
        """Test model in evaluation mode."""
        from src.models.cnn_classifier import CNNClassifier
        
        model = CNNClassifier(num_classes=62)
        model.eval()
        
        with torch.no_grad():
            output = model(sample_batch)
        
        assert output.shape == (8, 62)

    def test_gradient_flow(self, sample_batch, sample_labels):
        """Test that gradients flow through the model."""
        from src.models.cnn_classifier import CNNClassifier
        
        model = CNNClassifier(num_classes=62)
        model.train()
        
        output = model(sample_batch)
        loss = F.cross_entropy(output, sample_labels)
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.all(param.grad == 0)
                break

    def test_model_parameters_count(self):
        """Test that model has expected number of parameters."""
        from src.models.cnn_classifier import CNNClassifier
        
        model = CNNClassifier(num_classes=62)
        total_params = sum(p.numel() for p in model.parameters())
        
        # Model should have a reasonable number of parameters
        assert total_params > 1000  # At least some parameters
        assert total_params < 10_000_000  # Not too many


class TestCNNClassifierEmbedding:
    """Tests for embedding extraction from CNN."""

    def test_get_embedding_shape(self, sample_batch):
        """Test embedding output shape."""
        from src.models.cnn_classifier import CNNClassifier
        
        model = CNNClassifier(num_classes=62, embedding_dim=128)
        embedding = model.get_embedding(sample_batch)
        assert embedding.shape == (8, 128)

    def test_embedding_different_dim(self, sample_batch):
        """Test embedding with different dimensions."""
        from src.models.cnn_classifier import CNNClassifier
        
        model = CNNClassifier(num_classes=62, embedding_dim=256)
        embedding = model.get_embedding(sample_batch)
        assert embedding.shape == (8, 256)

    def test_embedding_normalized(self, sample_batch):
        """Test that embeddings can be normalized."""
        from src.models.cnn_classifier import CNNClassifier
        
        model = CNNClassifier(num_classes=62, embedding_dim=128)
        embedding = model.get_embedding(sample_batch)
        normalized = F.normalize(embedding, p=2, dim=1)
        
        # Check unit norm
        norms = torch.norm(normalized, dim=1)
        np.testing.assert_array_almost_equal(
            norms.detach().numpy(),
            np.ones(8),
            decimal=5
        )

    def test_embedding_consistency(self, sample_single_image):
        """Test that same input produces same embedding."""
        from src.models.cnn_classifier import CNNClassifier
        
        model = CNNClassifier(num_classes=62, embedding_dim=128)
        model.eval()
        
        with torch.no_grad():
            emb1 = model.get_embedding(sample_single_image)
            emb2 = model.get_embedding(sample_single_image)
        
        assert torch.allclose(emb1, emb2)
