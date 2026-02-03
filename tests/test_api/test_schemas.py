"""
Tests for Pydantic schemas.

Tests request/response validation models.
"""
import pytest
from pydantic import ValidationError
import base64


class TestClassifyRequest:
    """Tests for ClassifyRequest schema."""

    def test_valid_base64_image(self, mock_image_base64):
        """Test valid base64 image is accepted."""
        from api.schemas import ClassifyRequest
        
        request = ClassifyRequest(image=mock_image_base64)
        assert request.image == mock_image_base64

    def test_invalid_base64_rejected(self):
        """Test invalid base64 is rejected."""
        from api.schemas import ClassifyRequest
        
        with pytest.raises(ValidationError):
            ClassifyRequest(image="not_valid_base64!@#$%")

    def test_empty_image_rejected(self):
        """Test empty image is rejected."""
        from api.schemas import ClassifyRequest
        
        with pytest.raises(ValidationError):
            ClassifyRequest(image="")

    def test_missing_image_rejected(self):
        """Test missing image field is rejected."""
        from api.schemas import ClassifyRequest
        
        with pytest.raises(ValidationError):
            ClassifyRequest()


class TestClassifyResponse:
    """Tests for ClassifyResponse schema."""

    def test_valid_response(self):
        """Test valid response is created."""
        from api.schemas import ClassifyResponse
        
        response = ClassifyResponse(character='A', confidence=0.95)
        assert response.character == 'A'
        assert response.confidence == 0.95

    def test_confidence_range(self):
        """Test confidence must be in valid range."""
        from api.schemas import ClassifyResponse
        
        # Valid range
        response = ClassifyResponse(character='A', confidence=0.0)
        assert response.confidence == 0.0
        
        response = ClassifyResponse(character='A', confidence=1.0)
        assert response.confidence == 1.0

    def test_character_single_char(self):
        """Test character is a single character."""
        from api.schemas import ClassifyResponse
        
        response = ClassifyResponse(character='A', confidence=0.95)
        assert len(response.character) == 1


class TestSimilarityRequest:
    """Tests for SimilarityRequest schema."""

    def test_valid_request(self, mock_image_base64):
        """Test valid similarity request."""
        from api.schemas import SimilarityRequest
        
        request = SimilarityRequest(
            image1=mock_image_base64,
            image2=mock_image_base64
        )
        assert request.image1 == mock_image_base64
        assert request.image2 == mock_image_base64

    def test_missing_image1_rejected(self, mock_image_base64):
        """Test missing image1 is rejected."""
        from api.schemas import SimilarityRequest
        
        with pytest.raises(ValidationError):
            SimilarityRequest(image2=mock_image_base64)

    def test_missing_image2_rejected(self, mock_image_base64):
        """Test missing image2 is rejected."""
        from api.schemas import SimilarityRequest
        
        with pytest.raises(ValidationError):
            SimilarityRequest(image1=mock_image_base64)


class TestSimilarityResponse:
    """Tests for SimilarityResponse schema."""

    def test_valid_response(self):
        """Test valid similarity response."""
        from api.schemas import SimilarityResponse
        
        response = SimilarityResponse(
            character='A',
            confidence=0.95,
            similarity=85.5,
            reference_id='A_standard'
        )
        assert response.similarity == 85.5

    def test_similarity_range(self):
        """Test similarity is in valid range [0, 100]."""
        from api.schemas import SimilarityResponse
        
        # Valid range
        response = SimilarityResponse(
            character='A',
            confidence=0.95,
            similarity=0.0,
            reference_id='A_standard'
        )
        assert response.similarity == 0.0
        
        response = SimilarityResponse(
            character='A',
            confidence=0.95,
            similarity=100.0,
            reference_id='A_standard'
        )
        assert response.similarity == 100.0


class TestAnalyzeRequest:
    """Tests for AnalyzeRequest schema."""

    def test_valid_request(self, mock_image_base64):
        """Test valid analyze request."""
        from api.schemas import AnalyzeRequest
        
        request = AnalyzeRequest(image=mock_image_base64)
        assert request.image == mock_image_base64

    def test_optional_reference_id(self, mock_image_base64):
        """Test optional reference_id field."""
        from api.schemas import AnalyzeRequest
        
        # Without reference_id
        request = AnalyzeRequest(image=mock_image_base64)
        assert request.reference_id is None
        
        # With reference_id
        request = AnalyzeRequest(
            image=mock_image_base64,
            reference_id='A_standard'
        )
        assert request.reference_id == 'A_standard'


class TestTopKResponse:
    """Tests for TopKResponse schema."""

    def test_valid_response(self):
        """Test valid top-k response."""
        from api.schemas import TopKResponse, PredictionItem
        
        items = [
            PredictionItem(character='A', confidence=0.8),
            PredictionItem(character='4', confidence=0.1),
            PredictionItem(character='H', confidence=0.05),
        ]
        response = TopKResponse(predictions=items)
        
        assert len(response.predictions) == 3

    def test_predictions_sorted(self):
        """Test that predictions should be sorted by confidence."""
        from api.schemas import TopKResponse, PredictionItem
        
        items = [
            PredictionItem(character='A', confidence=0.8),
            PredictionItem(character='4', confidence=0.1),
            PredictionItem(character='H', confidence=0.05),
        ]
        response = TopKResponse(predictions=items)
        
        confidences = [p.confidence for p in response.predictions]
        assert confidences == sorted(confidences, reverse=True)
