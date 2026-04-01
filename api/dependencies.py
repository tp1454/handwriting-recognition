"""Dependency providers for API handlers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import yaml
from src.easy_ocr.easyocr import EasyOCRDetector

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
CHARSET = (
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)


class DummyClassifier(torch.nn.Module):
    """Fallback classifier used when exported model files are unavailable."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = int(inputs.shape[0]) if inputs.ndim >= 1 else 1
        logits = torch.zeros(
            (batch_size, len(CHARSET)), dtype=torch.float32
        )
        logits[:, 0] = 1.0
        return logits


class DummyEncoder(torch.nn.Module):
    """Fallback embedding model used when encoder export is unavailable."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = int(inputs.shape[0]) if inputs.ndim >= 1 else 1
        output = torch.zeros((batch_size, 128), dtype=torch.float32)
        output[:, 0] = 1.0
        return output


def _load_config() -> dict[str, Any]:
    if not DEFAULT_CONFIG_PATH.exists():
        return {}

    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        return {}
    return loaded


def _get_model_path(key: str) -> Path | None:
    config = _load_config()
    inference_config = config.get("inference", {})
    model_path = inference_config.get(key)
    if not model_path:
        return None
    return PROJECT_ROOT / str(model_path)


def _load_torchscript_module(path: Path) -> torch.nn.Module | None:
    if not path.exists():
        return None

    try:
        module = torch.jit.load(str(path), map_location="cpu")
        module.eval()
        return module
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_model() -> torch.nn.Module:
    """Return classifier model (cached)."""
    model_path = _get_model_path("classifier_path")
    if model_path is not None:
        loaded = _load_torchscript_module(model_path)
        if loaded is not None:
            return loaded

    fallback = DummyClassifier()
    fallback.eval()
    return fallback


@lru_cache(maxsize=1)
def get_encoder() -> torch.nn.Module:
    """Return encoder model (cached)."""
    encoder_path = _get_model_path("encoder_path")
    if encoder_path is not None:
        loaded = _load_torchscript_module(encoder_path)
        if loaded is not None:
            return loaded

    fallback = DummyEncoder()
    fallback.eval()
    return fallback


@lru_cache(maxsize=1)
def get_ocr_detector() -> Any:
    """Return EasyOCR detector (cached) or raise if initialization fails."""
    try:
        return EasyOCRDetector(languages=["vi"], gpu=False)
    except Exception as exc:
        raise RuntimeError(
            "EasyOCR detector is unavailable. Verify EasyOCR installation and models."
        ) from exc
