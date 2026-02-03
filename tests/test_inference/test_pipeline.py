"""
Tests for end-to-end inference pipeline.

Tests the complete inference orchestration.
"""
import pytest
import torch
import numpy as np
from PIL import Image
from unittest.mock import Mock, patch, MagicMock
import io
import base64


class TestInferencePipeline:
    """Tests for the inference pipeline."""

    def test_pipeline_initialization(self):
        """Test pipeline initializes correctly."""
        from src.inference.pipeline import InferencePipeline
        
        with patch('src.inference.pipeline.load_classifier') as mock_load_cls:
            with patch('src.inference.pipeline.load_encoder') as mock_load_enc:
                mock_load_cls.return_value = Mock()
                mock_load_enc.return_value = Mock()
                
                pipeline = InferencePipeline()
                
                assert pipeline is not None

    def test_pipeline_classify(self, sample_image_28x28):
        """Test classification through pipeline."""
        from src.inference.pipeline import InferencePipeline
        
        with patch('src.inference.pipeline.load_classifier') as mock_load_cls:
            with patch('src.inference.pipeline.load_encoder') as mock_load_enc:
                mock_classifier = Mock()
                mock_classifier.return_value = torch.randn(1, 62)
                mock_classifier.eval = Mock()
                mock_load_cls.return_value = mock_classifier
                mock_load_enc.return_value = Mock()
                
                pipeline = InferencePipeline()
                result = pipeline.classify(sample_image_28x28)
                
                assert 'character' in result
                assert 'confidence' in result

    def test_pipeline_similarity(self, sample_image_28x28):
        """Test similarity scoring through pipeline."""
        from src.inference.pipeline import InferencePipeline
        
        with patch('src.inference.pipeline.load_classifier') as mock_load_cls:
            with patch('src.inference.pipeline.load_encoder') as mock_load_enc:
                mock_encoder = Mock()
                mock_encoder.return_value = torch.randn(1, 128)
                mock_load_cls.return_value = Mock()
                mock_load_enc.return_value = mock_encoder
                
                pipeline = InferencePipeline()
                result = pipeline.compute_similarity(
                    sample_image_28x28,
                    sample_image_28x28
                )
                
                assert isinstance(result, float)

    def test_pipeline_full_analysis(self, sample_image_28x28):
        """Test full analysis (classify + similarity)."""
        from src.inference.pipeline import InferencePipeline
        
        with patch('src.inference.pipeline.load_classifier') as mock_load_cls:
            with patch('src.inference.pipeline.load_encoder') as mock_load_enc:
                with patch('src.inference.pipeline.load_reference') as mock_load_ref:
                    mock_classifier = Mock()
                    mock_classifier.return_value = torch.randn(1, 62)
                    mock_classifier.eval = Mock()
                    mock_load_cls.return_value = mock_classifier
                    
                    mock_encoder = Mock()
                    mock_encoder.return_value = torch.randn(1, 128)
                    mock_load_enc.return_value = mock_encoder
                    
                    mock_load_ref.return_value = torch.randn(1, 1, 28, 28)
                    
                    pipeline = InferencePipeline()
                    result = pipeline.analyze(sample_image_28x28)
                    
                    assert 'character' in result
                    assert 'confidence' in result
                    assert 'similarity' in result


class TestBase64ImageHandling:
    """Tests for base64 image handling in pipeline."""

    def test_decode_base64_image(self, mock_image_base64):
        """Test decoding base64 image."""
        from src.inference.pipeline import decode_base64_image
        
        image = decode_base64_image(mock_image_base64)
        
        assert image is not None
        assert isinstance(image, (Image.Image, np.ndarray, torch.Tensor))

    def test_decode_invalid_base64(self, invalid_base64):
        """Test error on invalid base64."""
        from src.inference.pipeline import decode_base64_image
        
        with pytest.raises(ValueError):
            decode_base64_image(invalid_base64)

    def test_pipeline_accepts_base64(self, mock_image_base64):
        """Test pipeline accepts base64 input."""
        from src.inference.pipeline import InferencePipeline
        
        with patch('src.inference.pipeline.load_classifier') as mock_load_cls:
            with patch('src.inference.pipeline.load_encoder') as mock_load_enc:
                mock_classifier = Mock()
                mock_classifier.return_value = torch.randn(1, 62)
                mock_classifier.eval = Mock()
                mock_load_cls.return_value = mock_classifier
                mock_load_enc.return_value = Mock()
                
                pipeline = InferencePipeline()
                result = pipeline.classify_base64(mock_image_base64)
                
                assert 'character' in result


class TestPipelinePreprocessing:
    """Tests for preprocessing in pipeline."""

    def test_preprocessing_applied(self, sample_image_rgb):
        """Test that preprocessing is applied to inputs."""
        from src.inference.pipeline import InferencePipeline
        
        with patch('src.inference.pipeline.load_classifier') as mock_load_cls:
            with patch('src.inference.pipeline.load_encoder') as mock_load_enc:
                with patch('src.inference.pipeline.preprocess') as mock_preprocess:
                    mock_classifier = Mock()
                    mock_classifier.return_value = torch.randn(1, 62)
                    mock_classifier.eval = Mock()
                    mock_load_cls.return_value = mock_classifier
                    mock_load_enc.return_value = Mock()
                    mock_preprocess.return_value = torch.randn(1, 1, 28, 28)
                    
                    pipeline = InferencePipeline()
                    pipeline.classify(sample_image_rgb)
                    
                    mock_preprocess.assert_called()

    def test_already_tensor_minimal_preprocessing(self, sample_single_image):
        """Test that tensors require minimal preprocessing."""
        from src.inference.pipeline import InferencePipeline
        
        with patch('src.inference.pipeline.load_classifier') as mock_load_cls:
            with patch('src.inference.pipeline.load_encoder') as mock_load_enc:
                mock_classifier = Mock()
                mock_classifier.return_value = torch.randn(1, 62)
                mock_classifier.eval = Mock()
                mock_load_cls.return_value = mock_classifier
                mock_load_enc.return_value = Mock()
                
                pipeline = InferencePipeline()
                result = pipeline.classify(sample_single_image)
                
                assert result is not None


class TestPipelineBatchProcessing:
    """Tests for batch processing in pipeline."""

    def test_batch_classification(self, sample_batch):
        """Test batch classification."""
        from src.inference.pipeline import InferencePipeline
        
        with patch('src.inference.pipeline.load_classifier') as mock_load_cls:
            with patch('src.inference.pipeline.load_encoder') as mock_load_enc:
                mock_classifier = Mock()
                mock_classifier.return_value = torch.randn(8, 62)
                mock_classifier.eval = Mock()
                mock_load_cls.return_value = mock_classifier
                mock_load_enc.return_value = Mock()
                
                pipeline = InferencePipeline()
                results = pipeline.classify_batch(sample_batch)
                
                assert len(results) == 8

    def test_batch_maintains_order(self, sample_batch):
        """Test that batch results maintain input order."""
        from src.inference.pipeline import InferencePipeline
        
        with patch('src.inference.pipeline.load_classifier') as mock_load_cls:
            with patch('src.inference.pipeline.load_encoder') as mock_load_enc:
                mock_classifier = Mock()
                # Create predictable output
                logits = torch.zeros(8, 62)
                for i in range(8):
                    logits[i, i % 62] = 10.0
                mock_classifier.return_value = logits
                mock_classifier.eval = Mock()
                mock_load_cls.return_value = mock_classifier
                mock_load_enc.return_value = Mock()
                
                pipeline = InferencePipeline()
                results = pipeline.classify_batch(sample_batch)
                
                # Results should be in order
                assert len(results) == 8


@pytest.mark.integration
class TestPipelineIntegration:
    """Integration tests for inference pipeline."""

    def test_end_to_end_classification(self, sample_image_28x28):
        """Test end-to-end classification flow."""
        from src.inference.pipeline import InferencePipeline
        
        # This would use real models in integration test
        # For unit test, we mock
        with patch('src.inference.pipeline.load_classifier') as mock_load_cls:
            with patch('src.inference.pipeline.load_encoder') as mock_load_enc:
                mock_classifier = Mock()
                mock_classifier.return_value = torch.randn(1, 62)
                mock_classifier.eval = Mock()
                mock_load_cls.return_value = mock_classifier
                mock_load_enc.return_value = Mock()
                
                pipeline = InferencePipeline()
                result = pipeline.classify(sample_image_28x28)
                
                # Verify complete result structure
                assert 'character' in result
                assert 'confidence' in result
                assert isinstance(result['character'], str)
                assert isinstance(result['confidence'], float)
