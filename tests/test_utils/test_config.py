"""
Tests for configuration utilities.

Tests YAML configuration loading and validation.
"""

import pytest
import yaml


class TestConfigLoading:
    """Tests for configuration loading."""

    def test_load_yaml_config(self, temp_config_file):
        """Test loading YAML configuration file."""
        from src.utils.config import load_config

        config = load_config(temp_config_file)

        assert config is not None
        assert "model" in config
        assert "training" in config

    def test_config_model_section(self, temp_config_file):
        """Test model configuration section."""
        from src.utils.config import load_config

        config = load_config(temp_config_file)

        assert config["model"]["num_classes"] == 62
        assert config["model"]["embedding_dim"] == 128

    def test_config_training_section(self, temp_config_file):
        """Test training configuration section."""
        from src.utils.config import load_config

        config = load_config(temp_config_file)

        assert config["training"]["epochs"] == 10
        assert config["training"]["batch_size"] == 32
        assert config["training"]["learning_rate"] == 0.001

    def test_missing_config_file(self, tmp_path):
        """Test error on missing config file."""
        from src.utils.config import load_config

        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_invalid_yaml(self, tmp_path):
        """Test error on invalid YAML."""
        from src.utils.config import load_config

        invalid_file = tmp_path / "invalid.yaml"
        invalid_file.write_text("invalid: yaml: content:")

        with pytest.raises(yaml.YAMLError):
            load_config(invalid_file)


class TestConfigDataclass:
    """Tests for Config dataclass."""

    def test_config_from_dict(self):
        """Test creating Config from dictionary."""
        from src.utils.config import Config

        data = {
            "model": {"num_classes": 62, "embedding_dim": 128},
            "training": {"epochs": 10, "batch_size": 32, "learning_rate": 0.001},
            "data": {"train_path": "data/train", "val_path": "data/val"},
        }

        config = Config.from_dict(data)

        assert config.model.num_classes == 62
        assert config.training.epochs == 10

    def test_config_from_yaml(self, temp_config_file):
        """Test creating Config from YAML file."""
        from src.utils.config import Config

        config = Config.from_yaml(temp_config_file)

        assert config is not None

    def test_config_defaults(self):
        """Test Config with default values."""
        from src.utils.config import ModelConfig, TrainConfig

        model_config = ModelConfig()
        assert model_config.num_classes == 62  # Default

        train_config = TrainConfig()
        assert train_config.epochs == 10  # Default


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_validate_positive_epochs(self):
        """Test that epochs must be positive."""
        from src.utils.config import TrainConfig

        with pytest.raises(ValueError):
            TrainConfig(epochs=0)

    def test_validate_positive_batch_size(self):
        """Test that batch_size must be positive."""
        from src.utils.config import TrainConfig

        with pytest.raises(ValueError):
            TrainConfig(batch_size=0)

    def test_validate_learning_rate_range(self):
        """Test that learning_rate must be positive."""
        from src.utils.config import TrainConfig

        with pytest.raises(ValueError):
            TrainConfig(learning_rate=-0.001)

    def test_validate_num_classes(self):
        """Test that num_classes must be positive."""
        from src.utils.config import ModelConfig

        with pytest.raises(ValueError):
            ModelConfig(num_classes=0)


class TestEnvironmentVariables:
    """Tests for environment variable overrides."""

    def test_env_override_model_path(self, monkeypatch):
        """Test MODEL_PATH environment variable."""
        from src.utils.config import get_model_path

        monkeypatch.setenv("MODEL_PATH", "/custom/path/model.pt")

        path = get_model_path()
        assert path == "/custom/path/model.pt"

    def test_default_model_path(self, monkeypatch):
        """Test default MODEL_PATH when not set."""
        from src.utils.config import get_model_path

        monkeypatch.delenv("MODEL_PATH", raising=False)

        path = get_model_path()
        assert "models/exports/classifier.pt" in path

    def test_env_override_log_level(self, monkeypatch):
        """Test LOG_LEVEL environment variable."""
        from src.utils.config import get_log_level

        monkeypatch.setenv("LOG_LEVEL", "DEBUG")

        level = get_log_level()
        assert level == "DEBUG"
