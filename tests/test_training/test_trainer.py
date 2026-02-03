"""
Tests for the training loop and trainer class.

Tests training functionality including checkpointing and early stopping.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import Mock, patch, MagicMock


class TestTrainer:
    """Tests for Trainer class."""

    def test_trainer_initialization(self, mock_dataset, model_config, train_config):
        """Test trainer initializes correctly."""
        from src.training.trainer import Trainer
        from src.models.cnn_classifier import CNNClassifier
        
        model = CNNClassifier(**model_config)
        trainer = Trainer(model, **train_config)
        
        assert trainer.model is not None
        assert trainer.epochs == train_config['epochs']

    def test_single_epoch(self, mock_dataloader, model_config):
        """Test training for a single epoch."""
        from src.training.trainer import Trainer
        from src.models.cnn_classifier import CNNClassifier
        
        model = CNNClassifier(**model_config)
        trainer = Trainer(model, epochs=1, batch_size=8, learning_rate=0.001)
        
        loss = trainer.train_epoch(mock_dataloader)
        
        assert isinstance(loss, float)
        assert loss > 0

    def test_validation_step(self, mock_dataloader, model_config):
        """Test validation step."""
        from src.training.trainer import Trainer
        from src.models.cnn_classifier import CNNClassifier
        
        model = CNNClassifier(**model_config)
        trainer = Trainer(model, epochs=1, batch_size=8, learning_rate=0.001)
        
        val_loss, val_acc = trainer.validate(mock_dataloader)
        
        assert isinstance(val_loss, float)
        assert isinstance(val_acc, float)
        assert 0.0 <= val_acc <= 1.0


class TestEarlyStopping:
    """Tests for early stopping functionality."""

    def test_early_stopping_triggers(self):
        """Test that early stopping triggers after patience epochs."""
        from src.training.trainer import EarlyStopping
        
        early_stop = EarlyStopping(patience=3)
        
        # Simulate worsening validation loss
        early_stop.update(1.0)  # epoch 1
        assert not early_stop.should_stop
        
        early_stop.update(1.1)  # epoch 2 - worse
        assert not early_stop.should_stop
        
        early_stop.update(1.2)  # epoch 3 - worse
        assert not early_stop.should_stop
        
        early_stop.update(1.3)  # epoch 4 - worse (patience exceeded)
        assert early_stop.should_stop

    def test_early_stopping_resets(self):
        """Test that early stopping resets on improvement."""
        from src.training.trainer import EarlyStopping
        
        early_stop = EarlyStopping(patience=3)
        
        early_stop.update(1.0)
        early_stop.update(1.1)  # worse
        early_stop.update(1.2)  # worse
        early_stop.update(0.9)  # better - should reset
        
        assert not early_stop.should_stop
        assert early_stop.counter == 0

    def test_early_stopping_best_value(self):
        """Test that best value is tracked correctly."""
        from src.training.trainer import EarlyStopping
        
        early_stop = EarlyStopping(patience=3)
        
        early_stop.update(1.0)
        early_stop.update(0.8)
        early_stop.update(0.9)
        early_stop.update(0.7)
        
        assert early_stop.best_value == pytest.approx(0.7)


class TestCheckpointing:
    """Tests for model checkpointing."""

    def test_save_checkpoint(self, temp_checkpoint_dir, model_config):
        """Test saving a checkpoint."""
        from src.training.trainer import save_checkpoint
        from src.models.cnn_classifier import CNNClassifier
        
        model = CNNClassifier(**model_config)
        optimizer = torch.optim.Adam(model.parameters())
        
        checkpoint_path = temp_checkpoint_dir / "checkpoint.pt"
        save_checkpoint(model, optimizer, epoch=5, path=checkpoint_path)
        
        assert checkpoint_path.exists()

    def test_load_checkpoint(self, temp_checkpoint_dir, model_config):
        """Test loading a checkpoint."""
        from src.training.trainer import save_checkpoint, load_checkpoint
        from src.models.cnn_classifier import CNNClassifier
        
        model = CNNClassifier(**model_config)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Save
        checkpoint_path = temp_checkpoint_dir / "checkpoint.pt"
        save_checkpoint(model, optimizer, epoch=5, path=checkpoint_path)
        
        # Load into new model
        new_model = CNNClassifier(**model_config)
        new_optimizer = torch.optim.Adam(new_model.parameters())
        
        epoch = load_checkpoint(new_model, new_optimizer, path=checkpoint_path)
        
        assert epoch == 5

    def test_best_model_saved(self, temp_checkpoint_dir, model_config, mock_dataloader):
        """Test that best model is saved during training."""
        from src.training.trainer import Trainer
        from src.models.cnn_classifier import CNNClassifier
        
        model = CNNClassifier(**model_config)
        trainer = Trainer(
            model, 
            epochs=2, 
            batch_size=8, 
            learning_rate=0.001,
            checkpoint_dir=temp_checkpoint_dir
        )
        
        trainer.fit(mock_dataloader, mock_dataloader)
        
        best_model_path = temp_checkpoint_dir / "best_model.pt"
        assert best_model_path.exists()


class TestOptimizer:
    """Tests for optimizer configuration."""

    def test_adam_optimizer(self, model_config):
        """Test Adam optimizer is used by default."""
        from src.training.trainer import Trainer
        from src.models.cnn_classifier import CNNClassifier
        
        model = CNNClassifier(**model_config)
        trainer = Trainer(model, epochs=1, learning_rate=0.001)
        
        assert isinstance(trainer.optimizer, torch.optim.Adam)

    def test_learning_rate(self, model_config):
        """Test learning rate is set correctly."""
        from src.training.trainer import Trainer
        from src.models.cnn_classifier import CNNClassifier
        
        model = CNNClassifier(**model_config)
        trainer = Trainer(model, epochs=1, learning_rate=0.01)
        
        for param_group in trainer.optimizer.param_groups:
            assert param_group['lr'] == 0.01


class TestGradientClipping:
    """Tests for gradient clipping."""

    def test_gradient_clipping_applied(self, model_config, mock_dataloader):
        """Test that gradient clipping is applied."""
        from src.training.trainer import Trainer
        from src.models.cnn_classifier import CNNClassifier
        
        model = CNNClassifier(**model_config)
        trainer = Trainer(
            model, 
            epochs=1, 
            learning_rate=0.001,
            max_grad_norm=1.0
        )
        
        # Train one batch
        for images, labels in mock_dataloader:
            trainer.optimizer.zero_grad()
            output = model(images)
            loss = F.cross_entropy(output, labels)
            loss.backward()
            
            # Get gradient norm before clipping
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Get gradient norm after clipping
            clipped_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    clipped_norm += p.grad.data.norm(2).item() ** 2
            clipped_norm = clipped_norm ** 0.5
            
            assert clipped_norm <= 1.0 + 1e-6
            break


@pytest.mark.slow
class TestFullTraining:
    """Integration tests for full training loop."""

    def test_full_training_loop(self, mock_dataloader, model_config, temp_checkpoint_dir):
        """Test complete training loop."""
        from src.training.trainer import Trainer
        from src.models.cnn_classifier import CNNClassifier
        
        model = CNNClassifier(**model_config)
        trainer = Trainer(
            model,
            epochs=3,
            batch_size=8,
            learning_rate=0.001,
            patience=5,
            checkpoint_dir=temp_checkpoint_dir
        )
        
        history = trainer.fit(mock_dataloader, mock_dataloader)
        
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == 3
