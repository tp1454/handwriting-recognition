"""
Tests for data augmentation transforms.

Tests augmentation operations for training robustness.
"""
import pytest
import numpy as np
import torch


class TestRotationAugmentation:
    """Tests for rotation augmentation."""

    def test_rotation_preserves_shape(self, sample_batch):
        """Test that rotation preserves image shape."""
        from src.data.augmentation import random_rotation
        
        rotated = random_rotation(sample_batch, max_angle=15)
        assert rotated.shape == sample_batch.shape

    def test_rotation_within_range(self, sample_single_image):
        """Test that rotation is within specified range."""
        from src.data.augmentation import random_rotation
        
        # Multiple rotations should produce different results
        results = [random_rotation(sample_single_image, max_angle=15) for _ in range(10)]
        # Check that we get some variation (not all identical)
        assert not all(torch.allclose(results[0], r) for r in results[1:])


class TestScaleAugmentation:
    """Tests for scale augmentation."""

    def test_scale_preserves_shape(self, sample_batch):
        """Test that scaling preserves output shape."""
        from src.data.augmentation import random_scale
        
        scaled = random_scale(sample_batch, scale_range=(0.9, 1.1))
        assert scaled.shape == sample_batch.shape


class TestTranslationAugmentation:
    """Tests for translation augmentation."""

    def test_translation_preserves_shape(self, sample_batch):
        """Test that translation preserves image shape."""
        from src.data.augmentation import random_translation
        
        translated = random_translation(sample_batch, max_shift=3)
        assert translated.shape == sample_batch.shape


class TestNoiseAugmentation:
    """Tests for noise augmentation."""

    def test_gaussian_noise(self, sample_batch):
        """Test Gaussian noise augmentation."""
        from src.data.augmentation import add_gaussian_noise
        
        noisy = add_gaussian_noise(sample_batch, std=0.1)
        assert noisy.shape == sample_batch.shape
        # Should not be identical
        assert not torch.allclose(noisy, sample_batch)

    def test_noise_preserves_range(self, sample_batch):
        """Test that noise doesn't push values out of valid range."""
        from src.data.augmentation import add_gaussian_noise
        
        # Clamp original to [0, 1]
        clamped = torch.clamp(sample_batch, 0, 1)
        noisy = add_gaussian_noise(clamped, std=0.1, clamp=True)
        assert noisy.min() >= 0.0
        assert noisy.max() <= 1.0


class TestAugmentationPipeline:
    """Tests for combined augmentation pipeline."""

    def test_compose_augmentations(self, sample_batch):
        """Test composing multiple augmentations."""
        from src.data.augmentation import AugmentationPipeline
        
        pipeline = AugmentationPipeline([
            ('rotation', {'max_angle': 15}),
            ('noise', {'std': 0.05}),
        ])
        
        augmented = pipeline(sample_batch)
        assert augmented.shape == sample_batch.shape

    def test_augmentation_training_mode(self, sample_batch):
        """Test that augmentation only applies in training mode."""
        from src.data.augmentation import AugmentationPipeline
        
        pipeline = AugmentationPipeline([
            ('rotation', {'max_angle': 15}),
        ])
        
        # Training mode - should augment
        pipeline.train()
        train_result = pipeline(sample_batch)
        
        # Eval mode - should not augment
        pipeline.eval()
        eval_result = pipeline(sample_batch)
        
        assert torch.allclose(eval_result, sample_batch)
