"""
Tests for preprocessing functions.

Tests image preprocessing including:
- Resizing to 28x28
- Normalization to [0, 1] range
- Grayscale conversion
- Centering
"""
import pytest
import numpy as np
from PIL import Image


class TestResize:
    """Tests for image resize functionality."""

    def test_resize_to_28x28(self, sample_image_rgb):
        """Test that RGB images are resized to 28x28."""
        from src.data.preprocessing import resize
        
        resized = resize(sample_image_rgb, target_size=28)
        assert resized.shape == (28, 28)

    def test_resize_large_image(self, sample_image_large):
        """Test resizing larger images."""
        from src.data.preprocessing import resize
        
        resized = resize(sample_image_large, target_size=28)
        assert resized.shape == (28, 28)

    def test_resize_already_28x28(self, sample_image_28x28):
        """Test that 28x28 images remain unchanged in size."""
        from src.data.preprocessing import resize
        
        resized = resize(sample_image_28x28, target_size=28)
        assert resized.shape == (28, 28)

    @pytest.mark.parametrize("input_size,expected", [
        ((64, 64), (28, 28)),
        ((100, 100), (28, 28)),
        ((28, 28), (28, 28)),
        ((50, 30), (28, 28)),
    ])
    def test_resize_various_sizes(self, input_size, expected):
        """Test resizing from various input sizes."""
        from src.data.preprocessing import resize
        
        image = np.random.randint(0, 256, input_size, dtype=np.uint8)
        result = resize(image, target_size=28)
        assert result.shape == expected


class TestNormalization:
    """Tests for image normalization."""

    def test_normalization_range(self, sample_image_28x28):
        """Test that normalization produces values in [0, 1]."""
        from src.data.preprocessing import normalize
        
        normalized = normalize(sample_image_28x28)
        assert 0.0 <= normalized.min()
        assert normalized.max() <= 1.0

    def test_normalization_dtype(self, sample_image_28x28):
        """Test that normalized output is float."""
        from src.data.preprocessing import normalize
        
        normalized = normalize(sample_image_28x28)
        assert normalized.dtype == np.float32 or normalized.dtype == np.float64

    def test_normalization_preserves_shape(self, sample_image_28x28):
        """Test that normalization preserves image shape."""
        from src.data.preprocessing import normalize
        
        normalized = normalize(sample_image_28x28)
        assert normalized.shape == (28, 28)

    def test_normalization_all_zeros(self):
        """Test normalization of all-zero image."""
        from src.data.preprocessing import normalize
        
        zero_image = np.zeros((28, 28), dtype=np.uint8)
        normalized = normalize(zero_image)
        assert normalized.min() == 0.0

    def test_normalization_all_white(self):
        """Test normalization of all-white (255) image."""
        from src.data.preprocessing import normalize
        
        white_image = np.full((28, 28), 255, dtype=np.uint8)
        normalized = normalize(white_image)
        assert normalized.max() == pytest.approx(1.0, abs=0.01)


class TestGrayscaleConversion:
    """Tests for grayscale conversion."""

    def test_rgb_to_grayscale(self, sample_image_rgb):
        """Test RGB to grayscale conversion."""
        from src.data.preprocessing import to_grayscale
        
        gray = to_grayscale(sample_image_rgb)
        assert len(gray.shape) == 2 or gray.shape[-1] == 1

    def test_grayscale_unchanged(self, sample_image_28x28):
        """Test that grayscale images are unchanged."""
        from src.data.preprocessing import to_grayscale
        
        result = to_grayscale(sample_image_28x28)
        assert len(result.shape) == 2 or result.shape[-1] == 1


class TestCentering:
    """Tests for image centering."""

    def test_center_image(self, sample_numpy_image):
        """Test that centering produces correct output shape."""
        from src.data.preprocessing import center_image
        
        centered = center_image(sample_numpy_image)
        assert centered.shape == sample_numpy_image.shape

    def test_center_preserves_content(self):
        """Test that centering doesn't destroy image content."""
        from src.data.preprocessing import center_image
        
        # Create image with content in corner
        image = np.zeros((28, 28), dtype=np.uint8)
        image[0:5, 0:5] = 255
        
        centered = center_image(image)
        # Some non-zero content should exist
        assert centered.sum() > 0


class TestPreprocessPipeline:
    """Tests for the full preprocessing pipeline."""

    def test_preprocess_returns_tensor(self, sample_image_rgb):
        """Test that preprocess returns proper output."""
        from src.data.preprocessing import preprocess
        
        result = preprocess(sample_image_rgb)
        assert result.shape == (28, 28) or result.shape == (1, 28, 28)

    def test_preprocess_handles_pil_image(self, sample_image_28x28):
        """Test preprocessing of PIL Image."""
        from src.data.preprocessing import preprocess
        
        result = preprocess(sample_image_28x28)
        assert result is not None

    def test_preprocess_handles_numpy_array(self, sample_numpy_image):
        """Test preprocessing of numpy array."""
        from src.data.preprocessing import preprocess
        
        result = preprocess(sample_numpy_image)
        assert result is not None
