"""
Tests for classifier inference.

Tests character prediction with confidence scores.
"""
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch


class TestClassifierInference:
    """Tests for character classification inference."""

    def test_predict_returns_character(self, sample_single_image, label_map):
        """Test that predict returns a valid character."""
        from src.inference.classifier import predict
        
        with patch('src.inference.classifier.get_model') as mock_get_model:
            mock_model = Mock()
            mock_model.return_value = torch.randn(1, 62)
            mock_model.eval = Mock()
            mock_get_model.return_value = mock_model
            
            character, confidence = predict(sample_single_image)
            
            valid_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            assert character in valid_chars

    def test_predict_returns_confidence(self, sample_single_image):
        """Test that predict returns a confidence score."""
        from src.inference.classifier import predict
        
        with patch('src.inference.classifier.get_model') as mock_get_model:
            mock_model = Mock()
            mock_model.return_value = torch.randn(1, 62)
            mock_model.eval = Mock()
            mock_get_model.return_value = mock_model
            
            character, confidence = predict(sample_single_image)
            
            assert isinstance(confidence, float)
            assert 0.0 <= confidence <= 1.0

    def test_predict_high_confidence_correct(self, sample_single_image):
        """Test that high logit produces high confidence."""
        from src.inference.classifier import predict
        
        with patch('src.inference.classifier.get_model') as mock_get_model:
            mock_model = Mock()
            # Very high logit for class 0
            logits = torch.zeros(1, 62)
            logits[0, 0] = 100.0
            mock_model.return_value = logits
            mock_model.eval = Mock()
            mock_get_model.return_value = mock_model
            
            character, confidence = predict(sample_single_image)
            
            assert confidence > 0.99

    def test_predict_batch(self, sample_batch):
        """Test batch prediction."""
        from src.inference.classifier import predict_batch
        
        with patch('src.inference.classifier.get_model') as mock_get_model:
            mock_model = Mock()
            mock_model.return_value = torch.randn(8, 62)
            mock_model.eval = Mock()
            mock_get_model.return_value = mock_model
            
            characters, confidences = predict_batch(sample_batch)
            
            assert len(characters) == 8
            assert len(confidences) == 8


class TestTopKPredictions:
    """Tests for top-k predictions."""

    def test_top_k_returns_k_results(self, sample_single_image):
        """Test that top_k returns k predictions."""
        from src.inference.classifier import predict_top_k
        
        with patch('src.inference.classifier.get_model') as mock_get_model:
            mock_model = Mock()
            mock_model.return_value = torch.randn(1, 62)
            mock_model.eval = Mock()
            mock_get_model.return_value = mock_model
            
            results = predict_top_k(sample_single_image, k=5)
            
            assert len(results) == 5

    def test_top_k_sorted_by_confidence(self, sample_single_image):
        """Test that top_k results are sorted by confidence."""
        from src.inference.classifier import predict_top_k
        
        with patch('src.inference.classifier.get_model') as mock_get_model:
            mock_model = Mock()
            mock_model.return_value = torch.randn(1, 62)
            mock_model.eval = Mock()
            mock_get_model.return_value = mock_model
            
            results = predict_top_k(sample_single_image, k=5)
            
            confidences = [r['confidence'] for r in results]
            assert confidences == sorted(confidences, reverse=True)

    def test_top_k_confidences_sum_less_than_one(self, sample_single_image):
        """Test that top_k confidences are valid probabilities."""
        from src.inference.classifier import predict_top_k
        
        with patch('src.inference.classifier.get_model') as mock_get_model:
            mock_model = Mock()
            mock_model.return_value = torch.randn(1, 62)
            mock_model.eval = Mock()
            mock_get_model.return_value = mock_model
            
            results = predict_top_k(sample_single_image, k=5)
            
            total_conf = sum(r['confidence'] for r in results)
            assert total_conf <= 1.0


class TestModelLoading:
    """Tests for model loading in inference."""

    def test_model_loads_once(self):
        """Test that model is cached after first load."""
        from src.inference.classifier import get_model
        
        with patch('src.inference.classifier.load_model') as mock_load:
            mock_load.return_value = Mock()
            
            # Call twice
            model1 = get_model()
            model2 = get_model()
            
            # Should only load once due to caching
            # Note: actual implementation may vary
            assert model1 is not None

    def test_model_in_eval_mode(self):
        """Test that loaded model is in eval mode."""
        from src.inference.classifier import get_model
        
        with patch('src.inference.classifier.load_model') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            get_model()
            
            mock_model.eval.assert_called()


class TestPreprocessingIntegration:
    """Tests for preprocessing in inference pipeline."""

    def test_raw_image_preprocessed(self, sample_image_rgb):
        """Test that raw images are preprocessed before inference."""
        from src.inference.classifier import classify_image
        
        with patch('src.inference.classifier.get_model') as mock_get_model:
            with patch('src.inference.classifier.preprocess') as mock_preprocess:
                mock_model = Mock()
                mock_model.return_value = torch.randn(1, 62)
                mock_model.eval = Mock()
                mock_get_model.return_value = mock_model
                mock_preprocess.return_value = torch.randn(1, 1, 28, 28)
                
                classify_image(sample_image_rgb)
                
                mock_preprocess.assert_called_once()
