"""
Shared pytest fixtures for handwriting recognition tests.
"""
import pytest
import numpy as np
import torch
from PIL import Image
import base64
import io


# =============================================================================
# Image Fixtures
# =============================================================================

@pytest.fixture
def sample_image_28x28():
    """Create a sample 28x28 grayscale image."""
    arr = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
    return Image.fromarray(arr, mode='L')


@pytest.fixture
def sample_image_rgb():
    """Create a sample RGB image."""
    arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode='RGB')


@pytest.fixture
def sample_image_large():
    """Create a larger grayscale image for resize testing."""
    arr = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    return Image.fromarray(arr, mode='L')


@pytest.fixture
def sample_numpy_image():
    """Create a sample numpy array image."""
    return np.random.randint(0, 256, (28, 28), dtype=np.uint8)


# =============================================================================
# Tensor Fixtures
# =============================================================================

@pytest.fixture
def sample_batch():
    """Create a batch of normalized images."""
    return torch.randn(8, 1, 28, 28).float()


@pytest.fixture
def sample_single_image():
    """Create a single normalized image tensor."""
    return torch.randn(1, 1, 28, 28).float()


@pytest.fixture
def sample_batch_numpy():
    """Create a batch of normalized images as numpy array."""
    return np.random.rand(8, 1, 28, 28).astype(np.float32)


@pytest.fixture
def sample_labels():
    """Create sample labels for a batch of 8."""
    return torch.randint(0, 62, (8,))


@pytest.fixture
def sample_embeddings():
    """Create sample embedding vectors."""
    return torch.randn(8, 128)


@pytest.fixture
def sample_embedding_pair():
    """Create a pair of embeddings for similarity testing."""
    emb1 = torch.randn(4, 128)
    emb2 = torch.randn(4, 128)
    return emb1, emb2


# =============================================================================
# API/Base64 Fixtures
# =============================================================================

@pytest.fixture
def mock_image_base64(sample_image_28x28):
    """Create a base64 encoded image string."""
    buffer = io.BytesIO()
    sample_image_28x28.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


@pytest.fixture
def invalid_base64():
    """Create an invalid base64 string."""
    return "not_valid_base64!@#$%"


# =============================================================================
# Model Configuration Fixtures
# =============================================================================

@pytest.fixture
def model_config():
    """Sample model configuration."""
    return {
        'num_classes': 62,
        'embedding_dim': 128,
        'input_size': 28,
    }


@pytest.fixture
def train_config():
    """Sample training configuration."""
    return {
        'epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.001,
        'patience': 5,
    }


# =============================================================================
# Dataset Fixtures
# =============================================================================

@pytest.fixture
def mock_dataset():
    """Create a mock dataset with random data."""
    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, size=100):
            self.size = size
            self.images = torch.randn(size, 1, 28, 28)
            self.labels = torch.randint(0, 62, (size,))
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return self.images[idx], self.labels[idx]
    
    return MockDataset()


@pytest.fixture
def mock_dataloader(mock_dataset):
    """Create a mock dataloader."""
    return torch.utils.data.DataLoader(mock_dataset, batch_size=8, shuffle=True)


# =============================================================================
# Character/Label Fixtures
# =============================================================================

@pytest.fixture
def label_map():
    """Create label mapping for 62 classes (0-9, A-Z, a-z)."""
    chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    return {i: c for i, c in enumerate(chars)}


@pytest.fixture
def reverse_label_map():
    """Create reverse label mapping (char -> index)."""
    chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    return {c: i for i, c in enumerate(chars)}


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================

@pytest.fixture
def temp_model_path(tmp_path):
    """Create a temporary path for saving models."""
    return tmp_path / "test_model.pt"


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Create a temporary directory for checkpoints."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary YAML config file."""
    config_content = """
model:
  num_classes: 62
  embedding_dim: 128
training:
  epochs: 10
  batch_size: 32
  learning_rate: 0.001
data:
  train_path: data/train
  val_path: data/val
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)
    return config_file
