"""
Tests for embedding extraction module.

Tests embedding extraction from backbone networks.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F


class TestEmbeddingExtractor:
    """Tests for embedding extraction."""

    def test_embedding_shape(self, sample_batch):
        """Test embedding output shape."""
        from src.models.cnn_classifier import CNNClassifier
        from src.models.embeddings import EmbeddingExtractor

        backbone = CNNClassifier(num_classes=62, embedding_dim=128)
        extractor = EmbeddingExtractor(backbone)

        embeddings = extractor(sample_batch)
        assert embeddings.shape == (8, 128)

    def test_embedding_normalization(self, sample_batch):
        """Test that embeddings can be L2 normalized."""
        from src.models.cnn_classifier import CNNClassifier
        from src.models.embeddings import EmbeddingExtractor

        backbone = CNNClassifier(num_classes=62, embedding_dim=128)
        extractor = EmbeddingExtractor(backbone, normalize=True)

        embeddings = extractor(sample_batch)
        norms = torch.norm(embeddings, dim=1)

        np.testing.assert_array_almost_equal(norms.detach().numpy(), np.ones(8), decimal=5)

    def test_embedding_batch_independence(self, sample_single_image):
        """Test that embeddings are computed independently per sample."""
        from src.models.cnn_classifier import CNNClassifier
        from src.models.embeddings import EmbeddingExtractor

        backbone = CNNClassifier(num_classes=62, embedding_dim=128)
        extractor = EmbeddingExtractor(backbone)
        extractor.eval()

        # Single image
        with torch.no_grad():
            single_emb = extractor(sample_single_image)

        # Same image in batch
        batch = sample_single_image.repeat(4, 1, 1, 1)
        with torch.no_grad():
            batch_emb = extractor(batch)

        # All embeddings in batch should match single
        for i in range(4):
            assert torch.allclose(batch_emb[i], single_emb[0], atol=1e-5)


class TestEmbeddingDistance:
    """Tests for embedding distance computations."""

    def test_cosine_distance(self, sample_embedding_pair):
        """Test cosine distance computation."""
        from src.models.embeddings import cosine_distance

        emb1, emb2 = sample_embedding_pair
        distance = cosine_distance(emb1, emb2)

        # Distance should be in [0, 2] for cosine
        assert distance.min() >= 0.0
        assert distance.max() <= 2.0

    def test_euclidean_distance(self, sample_embedding_pair):
        """Test Euclidean distance computation."""
        from src.models.embeddings import euclidean_distance

        emb1, emb2 = sample_embedding_pair
        distance = euclidean_distance(emb1, emb2)

        # Distance should be non-negative
        assert distance.min() >= 0.0

    def test_same_embedding_zero_distance(self, sample_embeddings):
        """Test that same embeddings have zero distance."""
        from src.models.embeddings import euclidean_distance

        distance = euclidean_distance(sample_embeddings, sample_embeddings)

        np.testing.assert_array_almost_equal(distance.detach().numpy(), np.zeros(8), decimal=5)
