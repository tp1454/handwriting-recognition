"""
Tests for similarity scoring.

Tests similarity computation between handwriting samples.
"""
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch


class TestSimilarityScoring:
    """Tests for similarity scoring functionality."""

    def test_similarity_returns_score(self, sample_single_image):
        """Test that similarity returns a score."""
        from src.inference.similarity import compute_similarity
        
        with patch('src.inference.similarity.get_encoder') as mock_get_encoder:
            mock_encoder = Mock()
            mock_encoder.return_value = torch.randn(1, 128)
            mock_get_encoder.return_value = mock_encoder
            
            score = compute_similarity(sample_single_image, sample_single_image)
            
            assert isinstance(score, float)

    def test_similarity_range(self, sample_single_image):
        """Test that similarity is in valid range [0, 100]."""
        from src.inference.similarity import compute_similarity
        
        with patch('src.inference.similarity.get_encoder') as mock_get_encoder:
            mock_encoder = Mock()
            mock_encoder.return_value = torch.randn(1, 128)
            mock_get_encoder.return_value = mock_encoder
            
            score = compute_similarity(sample_single_image, sample_single_image)
            
            assert 0.0 <= score <= 100.0

    def test_identical_images_high_similarity(self, sample_single_image):
        """Test that identical images have high similarity."""
        from src.inference.similarity import compute_similarity
        
        with patch('src.inference.similarity.get_encoder') as mock_get_encoder:
            mock_encoder = Mock()
            # Return same embedding for both
            embedding = torch.randn(1, 128)
            mock_encoder.return_value = embedding
            mock_get_encoder.return_value = mock_encoder
            
            score = compute_similarity(sample_single_image, sample_single_image)
            
            assert score > 90.0

    def test_different_images_lower_similarity(self, sample_single_image):
        """Test that different images have lower similarity."""
        from src.inference.similarity import compute_similarity
        
        with patch('src.inference.similarity.get_encoder') as mock_get_encoder:
            mock_encoder = Mock()
            # Return different embeddings
            call_count = [0]
            def side_effect(x):
                call_count[0] += 1
                if call_count[0] == 1:
                    return torch.tensor([[1.0, 0.0, 0.0]])
                else:
                    return torch.tensor([[0.0, 1.0, 0.0]])
            
            mock_encoder.side_effect = side_effect
            mock_get_encoder.return_value = mock_encoder
            
            other_image = torch.randn_like(sample_single_image)
            score = compute_similarity(sample_single_image, other_image)
            
            assert score < 50.0


class TestReferenceComparison:
    """Tests for comparison against reference samples."""

    def test_compare_to_reference(self, sample_single_image):
        """Test comparison against a reference image."""
        from src.inference.similarity import compare_to_reference
        
        with patch('src.inference.similarity.get_encoder') as mock_get_encoder:
            with patch('src.inference.similarity.load_reference') as mock_load_ref:
                mock_encoder = Mock()
                mock_encoder.return_value = torch.randn(1, 128)
                mock_get_encoder.return_value = mock_encoder
                mock_load_ref.return_value = torch.randn(1, 1, 28, 28)
                
                result = compare_to_reference(
                    sample_single_image,
                    reference_id='A_standard'
                )
                
                assert 'similarity' in result
                assert 'reference_id' in result

    def test_compare_to_multiple_references(self, sample_single_image):
        """Test comparison against multiple references."""
        from src.inference.similarity import compare_to_references
        
        with patch('src.inference.similarity.get_encoder') as mock_get_encoder:
            with patch('src.inference.similarity.load_references') as mock_load_refs:
                mock_encoder = Mock()
                mock_encoder.return_value = torch.randn(1, 128)
                mock_get_encoder.return_value = mock_encoder
                mock_load_refs.return_value = {
                    'A_standard': torch.randn(1, 1, 28, 28),
                    'A_cursive': torch.randn(1, 1, 28, 28),
                }
                
                results = compare_to_references(
                    sample_single_image,
                    character='A'
                )
                
                assert len(results) >= 1

    def test_best_match_returned(self, sample_single_image):
        """Test that best matching reference is identified."""
        from src.inference.similarity import find_best_match
        
        with patch('src.inference.similarity.get_encoder') as mock_get_encoder:
            with patch('src.inference.similarity.load_references') as mock_load_refs:
                mock_encoder = Mock()
                mock_encoder.return_value = torch.randn(1, 128)
                mock_get_encoder.return_value = mock_encoder
                mock_load_refs.return_value = {
                    'A_1': torch.randn(1, 1, 28, 28),
                    'A_2': torch.randn(1, 1, 28, 28),
                }
                
                best_match = find_best_match(sample_single_image, character='A')
                
                assert 'reference_id' in best_match
                assert 'similarity' in best_match


class TestEmbeddingExtraction:
    """Tests for embedding extraction in similarity."""

    def test_embedding_shape(self, sample_single_image):
        """Test that embeddings have correct shape."""
        from src.inference.similarity import extract_embedding
        
        with patch('src.inference.similarity.get_encoder') as mock_get_encoder:
            mock_encoder = Mock()
            mock_encoder.return_value = torch.randn(1, 128)
            mock_get_encoder.return_value = mock_encoder
            
            embedding = extract_embedding(sample_single_image)
            
            assert embedding.shape == (1, 128)

    def test_embedding_normalized(self, sample_single_image):
        """Test that embeddings are L2 normalized."""
        from src.inference.similarity import extract_embedding
        
        with patch('src.inference.similarity.get_encoder') as mock_get_encoder:
            mock_encoder = Mock()
            # Return normalized embedding
            emb = torch.randn(1, 128)
            emb = emb / torch.norm(emb)
            mock_encoder.return_value = emb
            mock_get_encoder.return_value = mock_encoder
            
            embedding = extract_embedding(sample_single_image, normalize=True)
            norm = torch.norm(embedding).item()
            
            assert norm == pytest.approx(1.0, abs=0.01)


class TestSimilarityMetrics:
    """Tests for different similarity metrics."""

    def test_cosine_similarity(self):
        """Test cosine similarity computation."""
        from src.inference.similarity import cosine_similarity
        
        emb1 = torch.tensor([[1.0, 0.0]])
        emb2 = torch.tensor([[1.0, 0.0]])
        
        sim = cosine_similarity(emb1, emb2)
        assert sim == pytest.approx(1.0)

    def test_euclidean_similarity(self):
        """Test Euclidean distance to similarity conversion."""
        from src.inference.similarity import euclidean_to_similarity
        
        emb1 = torch.tensor([[0.0, 0.0]])
        emb2 = torch.tensor([[0.0, 0.0]])
        
        sim = euclidean_to_similarity(emb1, emb2)
        assert sim == pytest.approx(100.0)  # Zero distance = 100% similarity
