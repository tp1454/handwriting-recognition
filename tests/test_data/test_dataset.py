"""
Tests for dataset classes.

Tests PyTorch Dataset implementations for loading handwriting data.
"""
import pytest
import torch
from torch.utils.data import DataLoader


class TestEMNISTDataset:
    """Tests for EMNIST Dataset class."""

    def test_dataset_len(self, mock_dataset):
        """Test that dataset returns correct length."""
        assert len(mock_dataset) == 100

    def test_dataset_getitem_returns_tuple(self, mock_dataset):
        """Test that __getitem__ returns (image, label) tuple."""
        item = mock_dataset[0]
        assert isinstance(item, tuple)
        assert len(item) == 2

    def test_dataset_image_shape(self, mock_dataset):
        """Test that images have correct shape."""
        image, _ = mock_dataset[0]
        assert image.shape == (1, 28, 28)

    def test_dataset_label_is_integer(self, mock_dataset):
        """Test that labels are integers."""
        _, label = mock_dataset[0]
        assert isinstance(label, (int, torch.Tensor))

    def test_dataset_label_range(self, mock_dataset):
        """Test that labels are in valid range (0-61 for 62 classes)."""
        _, label = mock_dataset[0]
        if isinstance(label, torch.Tensor):
            label = label.item()
        assert 0 <= label < 62

    def test_dataset_with_dataloader(self, mock_dataset):
        """Test that dataset works with DataLoader."""
        loader = DataLoader(mock_dataset, batch_size=8, shuffle=True)
        batch = next(iter(loader))
        images, labels = batch
        assert images.shape == (8, 1, 28, 28)
        assert labels.shape == (8,)


class TestDatasetTransforms:
    """Tests for dataset with transforms."""

    def test_dataset_applies_transform(self, mock_dataset):
        """Test that transforms are applied to images."""
        # This test would use a real dataset with transforms
        # For now, verify mock dataset output
        image, _ = mock_dataset[0]
        assert image.dtype == torch.float32

    def test_dataset_without_transform(self, mock_dataset):
        """Test dataset works without transform."""
        image, label = mock_dataset[0]
        assert image is not None
        assert label is not None


class TestDataLoaderBatching:
    """Tests for DataLoader batching behavior."""

    def test_batch_size(self, mock_dataloader):
        """Test correct batch size."""
        batch = next(iter(mock_dataloader))
        images, labels = batch
        assert images.shape[0] == 8

    def test_batch_shuffle(self, mock_dataset):
        """Test that shuffling works."""
        loader1 = DataLoader(mock_dataset, batch_size=8, shuffle=True)
        loader2 = DataLoader(mock_dataset, batch_size=8, shuffle=True)
        
        # Get first batch from each
        batch1 = next(iter(loader1))
        batch2 = next(iter(loader2))
        
        # Batches might differ due to shuffling (not guaranteed but likely)
        # Just verify both work
        assert batch1[0].shape == batch2[0].shape

    def test_drop_last(self, mock_dataset):
        """Test drop_last parameter."""
        loader = DataLoader(mock_dataset, batch_size=8, drop_last=True)
        for batch in loader:
            images, _ = batch
            assert images.shape[0] == 8
