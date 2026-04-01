"""Tests for sheet upload scoring endpoint."""

from __future__ import annotations

import io
from unittest.mock import patch

from PIL import Image


class _MockDetector:
    def detect_char_boxes(self, _image, **_kwargs):
        return {
            "horizontal_list": [
                [2, 12, 2, 14],
                [2, 12, 16, 28],
            ],
            "free_list": [],
        }

    def extract_labeled_boxes(self, _image):
        return [
            {
                "index": 0,
                "box": [2, 12, 2, 14],
                "label": "A",
                "confidence": 0.91,
            },
            {
                "index": 1,
                "box": [2, 12, 16, 28],
                "label": "B",
                "confidence": 0.89,
            },
        ]


class _MockSplitDetector:
    def detect_char_boxes(self, _image, **_kwargs):
        return {
            "horizontal_list": [
                [2, 50, 2, 14],
                [2, 6, 16, 28],
            ],
            "free_list": [],
        }

    def extract_labeled_boxes(self, _image):
        return [
            {
                "index": 0,
                "box": [2, 50, 2, 14],
                "label": "split-row",
                "confidence": 0.87,
            },
            {
                "index": 1,
                "box": [2, 6, 16, 28],
                "label": "single-row",
                "confidence": 0.84,
            },
        ]


def _png_bytes(width: int = 32, height: int = 32) -> bytes:
    image = Image.new("L", (width, height), color=0)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class TestSheetScoreEndpoint:
    """Endpoint tests for /sheet/score."""

    def test_sheet_score_returns_200(self):
        from fastapi.testclient import TestClient

        with patch(
            "api.dependencies.get_ocr_detector",
            return_value=_MockDetector(),
        ):
            from api.main import app

            client = TestClient(app)
            response = client.post(
                "/sheet/score",
                files={
                    "file": ("sheet.png", _png_bytes(), "image/png")
                },
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["detected_rows"] == 2
        assert len(payload["rows"]) == 2
        assert "expected_text" not in payload
        assert all(
            "expected_char" not in row for row in payload["rows"]
        )
        assert all(
            isinstance(row["ocr_label"], str) and row["ocr_label"]
            for row in payload["rows"]
        )
        assert "extracted_boxes" in payload
        assert len(payload["extracted_boxes"]) == 2
        assert all(
            {"index", "box", "label", "confidence"}.issubset(
                set(item.keys())
            )
            for item in payload["extracted_boxes"]
        )
        assert payload["rows"][0]["segment_scores"] == []
        assert payload["rows"][1]["segment_scores"] == []
        assert 0.0 <= payload["overall_score"] <= 100.0

    def test_sheet_score_segment_scores_exclude_reference(self):
        from fastapi.testclient import TestClient

        with patch(
            "api.dependencies.get_ocr_detector",
            return_value=_MockSplitDetector(),
        ):
            from api.main import app

            client = TestClient(app)
            response = client.post(
                "/sheet/score",
                files={
                    "file": (
                        "sheet.png",
                        _png_bytes(width=64, height=32),
                        "image/png",
                    )
                },
            )

        assert response.status_code == 200
        rows = response.json()["rows"]
        assert rows[0]["split_count"] == 2
        assert (
            len(rows[0]["segment_scores"])
            == rows[0]["split_count"] - 1
        )
        assert rows[1]["split_count"] == 1
        assert rows[1]["segment_scores"] == []
        assert all(
            0.0 <= score <= 100.0
            for score in rows[0]["segment_scores"]
        )

    def test_sheet_score_rejects_non_image_upload(self):
        from api.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/sheet/score",
            files={"file": ("sheet.txt", b"abc", "text/plain")},
        )

        assert response.status_code == 422

    def test_sheet_score_returns_503_when_easyocr_unavailable(self):
        from fastapi.testclient import TestClient

        with patch(
            "api.dependencies.get_ocr_detector",
            side_effect=RuntimeError(
                "EasyOCR detector is unavailable. Verify EasyOCR installation and models."
            ),
        ):
            from api.main import app

            client = TestClient(app)
            response = client.post(
                "/sheet/score",
                files={
                    "file": ("sheet.png", _png_bytes(), "image/png")
                },
            )

        assert response.status_code == 503
        assert (
            "EasyOCR detector is unavailable"
            in response.json()["detail"]
        )
