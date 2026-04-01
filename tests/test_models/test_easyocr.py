"""Tests for EasyOCR detection wrapper."""

from unittest.mock import Mock, patch

import pytest
from src.data.processing import EasyOCRDetector


class TestEasyOCRDetector:
    """Unit tests for the EasyOCR detector wrapper."""

    def test_initializes_reader_with_defaults(self):
        """Detector should create easyocr.Reader with default language and gpu flag."""
        with patch(
            "src.models.easyocr.importlib.import_module"
        ) as mock_import:
            fake_module = Mock()
            fake_reader = Mock()
            fake_module.Reader = Mock(return_value=fake_reader)
            mock_import.return_value = fake_module

            detector = EasyOCRDetector()

            fake_module.Reader.assert_called_once_with(
                ["en"], gpu=False
            )
            assert detector.reader is fake_reader

    def test_initializes_reader_with_custom_options(self):
        """Detector should pass custom ctor options into easyocr.Reader."""
        with patch(
            "src.models.easyocr.importlib.import_module"
        ) as mock_import:
            fake_module = Mock()
            fake_module.Reader = Mock(return_value=Mock())
            mock_import.return_value = fake_module

            EasyOCRDetector(
                languages=["en", "th"],
                gpu=True,
                reader_kwargs={
                    "model_storage_directory": "models/easyocr"
                },
            )

            fake_module.Reader.assert_called_once_with(
                ["en", "th"],
                gpu=True,
                model_storage_directory="models/easyocr",
            )

    def test_detect_calls_easyocr_with_defaults(self):
        """Detect should forward documented defaults and return normalized payload."""
        with patch(
            "src.models.easyocr.importlib.import_module"
        ) as mock_import:
            fake_module = Mock()
            fake_reader = Mock()
            fake_reader.detect.return_value = (["h"], ["f"])
            fake_module.Reader = Mock(return_value=fake_reader)
            mock_import.return_value = fake_module

            detector = EasyOCRDetector()
            result = detector.detect_char_boxes("image-bytes")

            fake_reader.detect.assert_called_once_with(
                "image-bytes",
                min_size=10,
                text_threshold=0.7,
                low_text=0.4,
                link_threshold=0.4,
                canvas_size=2560,
                mag_ratio=1.0,
                slope_ths=0.1,
                ycenter_ths=0.5,
                height_ths=0.5,
                width_ths=0.5,
                add_margin=0.1,
                optimal_num_chars=None,
            )
            assert result == {
                "horizontal_list": ["h"],
                "free_list": ["f"],
            }

    def test_detect_accepts_overrides(self):
        """Detect should forward custom thresholds and optional char count."""
        with patch(
            "src.models.easyocr.importlib.import_module"
        ) as mock_import:
            fake_module = Mock()
            fake_reader = Mock()
            fake_reader.detect.return_value = ([], [])
            fake_module.Reader = Mock(return_value=fake_reader)
            mock_import.return_value = fake_module

            detector = EasyOCRDetector()
            detector.detect_char_boxes(
                image="img",
                min_size=3,
                text_threshold=0.9,
                optimal_num_chars=1,
            )

            _, kwargs = fake_reader.detect.call_args
            assert kwargs["min_size"] == 3
            assert kwargs["text_threshold"] == 0.9
            assert kwargs["optimal_num_chars"] == 1

    def test_detect_raises_on_malformed_output(self):
        """Malformed detect response should raise ValueError."""
        with patch(
            "src.models.easyocr.importlib.import_module"
        ) as mock_import:
            fake_module = Mock()
            fake_reader = Mock()
            fake_reader.detect.return_value = "invalid"
            fake_module.Reader = Mock(return_value=fake_reader)
            mock_import.return_value = fake_module

            detector = EasyOCRDetector()

            with pytest.raises(ValueError, match="malformed output"):
                detector.detect_char_boxes("img")

    def test_import_raises_when_reader_missing(self):
        """Import helper should reject module objects without Reader attribute."""
        with patch(
            "src.models.easyocr.importlib.import_module"
        ) as mock_import:
            mock_import.return_value = object()

            with pytest.raises(
                ImportError, match="does not expose Reader"
            ):
                EasyOCRDetector()
