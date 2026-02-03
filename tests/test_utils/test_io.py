"""
Tests for I/O utilities.

Tests model saving/loading and image I/O.
"""
import pytest
import torch
import numpy as np
from PIL import Image
import io


class TestModelIO:
    """Tests for model save/load operations."""

    def test_save_model(self, temp_model_path, model_config):
        """Test saving a model."""
        from src.utils.io import save_model
        from src.models.cnn_classifier import CNNClassifier
        
        model = CNNClassifier(**model_config)
        save_model(model, temp_model_path)
        
        assert temp_model_path.exists()

    def test_load_model(self, temp_model_path, model_config):
        """Test loading a model."""
        from src.utils.io import save_model, load_model
        from src.models.cnn_classifier import CNNClassifier
        
        # Save first
        original_model = CNNClassifier(**model_config)
        save_model(original_model, temp_model_path)
        
        # Load
        loaded_model = load_model(CNNClassifier, temp_model_path, **model_config)
        
        assert loaded_model is not None

    def test_loaded_model_same_weights(self, temp_model_path, model_config, sample_batch):
        """Test that loaded model has same weights."""
        from src.utils.io import save_model, load_model
        from src.models.cnn_classifier import CNNClassifier
        
        # Save
        original_model = CNNClassifier(**model_config)
        original_model.eval()
        with torch.no_grad():
            original_output = original_model(sample_batch)
        
        save_model(original_model, temp_model_path)
        
        # Load
        loaded_model = load_model(CNNClassifier, temp_model_path, **model_config)
        loaded_model.eval()
        with torch.no_grad():
            loaded_output = loaded_model(sample_batch)
        
        assert torch.allclose(original_output, loaded_output)

    def test_load_nonexistent_model(self, tmp_path):
        """Test error on loading non-existent model."""
        from src.utils.io import load_model
        from src.models.cnn_classifier import CNNClassifier
        
        with pytest.raises(FileNotFoundError):
            load_model(CNNClassifier, tmp_path / "nonexistent.pt", num_classes=62)


class TestImageIO:
    """Tests for image I/O operations."""

    def test_load_image_from_path(self, tmp_path, sample_image_28x28):
        """Test loading image from file path."""
        from src.utils.io import load_image
        
        # Save image
        image_path = tmp_path / "test_image.png"
        sample_image_28x28.save(image_path)
        
        # Load
        loaded = load_image(image_path)
        
        assert loaded is not None

    def test_load_image_grayscale(self, tmp_path, sample_image_28x28):
        """Test loading image as grayscale."""
        from src.utils.io import load_image
        
        image_path = tmp_path / "test_image.png"
        sample_image_28x28.save(image_path)
        
        loaded = load_image(image_path, grayscale=True)
        
        # Should be 2D or have 1 channel
        if isinstance(loaded, np.ndarray):
            assert len(loaded.shape) == 2 or loaded.shape[-1] == 1

    def test_save_image(self, tmp_path, sample_numpy_image):
        """Test saving image to file."""
        from src.utils.io import save_image
        
        image_path = tmp_path / "output.png"
        save_image(sample_numpy_image, image_path)
        
        assert image_path.exists()

    def test_load_image_invalid_path(self):
        """Test error on invalid image path."""
        from src.utils.io import load_image
        
        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/path/image.png")


class TestBase64IO:
    """Tests for base64 image I/O."""

    def test_image_to_base64(self, sample_image_28x28):
        """Test converting image to base64."""
        from src.utils.io import image_to_base64
        
        b64 = image_to_base64(sample_image_28x28)
        
        assert isinstance(b64, str)
        # Should be valid base64
        import base64
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0

    def test_base64_to_image(self, mock_image_base64):
        """Test converting base64 to image."""
        from src.utils.io import base64_to_image
        
        image = base64_to_image(mock_image_base64)
        
        assert isinstance(image, (Image.Image, np.ndarray))

    def test_base64_roundtrip(self, sample_image_28x28):
        """Test base64 encode/decode roundtrip."""
        from src.utils.io import image_to_base64, base64_to_image
        
        # Encode
        b64 = image_to_base64(sample_image_28x28)
        
        # Decode
        decoded = base64_to_image(b64)
        
        # Should have same size
        if isinstance(decoded, Image.Image):
            assert decoded.size == sample_image_28x28.size


class TestTensorIO:
    """Tests for tensor I/O operations."""

    def test_image_to_tensor(self, sample_image_28x28):
        """Test converting image to tensor."""
        from src.utils.io import image_to_tensor
        
        tensor = image_to_tensor(sample_image_28x28)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape[-2:] == (28, 28)

    def test_tensor_to_image(self, sample_single_image):
        """Test converting tensor to image."""
        from src.utils.io import tensor_to_image
        
        image = tensor_to_image(sample_single_image)
        
        assert isinstance(image, (Image.Image, np.ndarray))

    def test_tensor_image_roundtrip(self, sample_image_28x28):
        """Test tensor conversion roundtrip."""
        from src.utils.io import image_to_tensor, tensor_to_image
        
        tensor = image_to_tensor(sample_image_28x28)
        image = tensor_to_image(tensor)
        
        assert image is not None
