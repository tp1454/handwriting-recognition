"""Safe lazy importer and wrapper for the easyocr package."""

from __future__ import annotations

import importlib
import sys
from typing import Any


def safe_import_easyocr() -> Any:
    """Import easyocr while avoiding common local module shadowing."""
    original_path = sys.path[:]

    try:
        if "" in sys.path:
            sys.path.remove("")
        module = importlib.import_module("easyocr")
    finally:
        sys.path = original_path

    if not hasattr(module, "Reader"):
        raise ImportError("easyocr module does not expose Reader")

    return module


EASYOCR_DETECT_DEFAULTS: dict[str, Any] = {
    "min_size": 10,
    "text_threshold": 0.7,
    "low_text": 0.4,
    "link_threshold": 0.4,
    "canvas_size": 2560,
    "mag_ratio": 1.0,
    "slope_ths": 0.1,
    "ycenter_ths": 0.5,
    "height_ths": 0.5,
    "width_ths": 0.5,
    "add_margin": 0.1,
    "optimal_num_chars": None,
}


class EasyOCRDetector:
    """Thin wrapper around ``easyocr.Reader`` for character box detection."""

    def __init__(
        self,
        languages: list[str] | None = None,
        gpu: bool = False,
        reader_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if languages is None:
            languages = ["en"]
        if reader_kwargs is None:
            reader_kwargs = {}

        easyocr = safe_import_easyocr()
        self.reader: Any = easyocr.Reader(
            languages,
            gpu=gpu,
            **reader_kwargs,
        )

    def detect_char_boxes(
        self,
        image: Any,
        min_size: int | None = None,
        text_threshold: float | None = None,
        low_text: float | None = None,
        link_threshold: float | None = None,
        canvas_size: int | None = None,
        mag_ratio: float | None = None,
        slope_ths: float | None = None,
        ycenter_ths: float | None = None,
        height_ths: float | None = None,
        width_ths: float | None = None,
        add_margin: float | None = None,
        optimal_num_chars: int | None = None,
    ) -> dict[str, list]:
        """Detect character boxes and normalize EasyOCR output."""
        detect_kwargs = dict(EASYOCR_DETECT_DEFAULTS)
        overrides = {
            "min_size": min_size,
            "text_threshold": text_threshold,
            "low_text": low_text,
            "link_threshold": link_threshold,
            "canvas_size": canvas_size,
            "mag_ratio": mag_ratio,
            "slope_ths": slope_ths,
            "ycenter_ths": ycenter_ths,
            "height_ths": height_ths,
            "width_ths": width_ths,
            "add_margin": add_margin,
            "optimal_num_chars": optimal_num_chars,
        }
        for key, value in overrides.items():
            if value is not None:
                detect_kwargs[key] = value

        result = self.reader.detect(image, **detect_kwargs)
        return self._normalize_detect_output(result)

    def extract_labeled_boxes(
        self,
        image: Any,
    ) -> list[dict[str, Any]]:
        """Extract OCR labels and normalize polygon boxes to axis-aligned boxes."""
        raw_items = self.reader.readtext(image)
        if not isinstance(raw_items, list):
            return []

        extracted: list[dict[str, Any]] = []
        for index, raw_item in enumerate(raw_items):
            if (
                not isinstance(raw_item, (tuple, list))
                or len(raw_item) < 3
            ):
                continue

            raw_box = raw_item[0]
            raw_label = raw_item[1]
            raw_confidence = raw_item[2]

            normalized_box = self._normalize_polygon_to_box(raw_box)
            if normalized_box is None:
                continue

            label = str(raw_label).strip()
            if not label:
                continue

            try:
                confidence = float(raw_confidence)
            except (TypeError, ValueError):
                confidence = 0.0

            extracted.append(
                {
                    "index": index,
                    "box": normalized_box,
                    "label": label,
                    "confidence": float(
                        max(0.0, min(1.0, confidence))
                    ),
                }
            )

        return extracted

    @staticmethod
    def _normalize_polygon_to_box(raw_box: Any) -> list[int] | None:
        """Convert EasyOCR polygon points to [x_min, x_max, y_min, y_max]."""
        if (
            not isinstance(raw_box, (list, tuple))
            or len(raw_box) == 0
        ):
            return None

        points: list[tuple[float, float]] = []
        for point in raw_box:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                x_value = float(point[0])
                y_value = float(point[1])
            except (TypeError, ValueError):
                continue
            points.append((x_value, y_value))

        if not points:
            return None

        xs = [point[0] for point in points]
        ys = [point[1] for point in points]

        x_min = int(round(min(xs)))
        x_max = int(round(max(xs)))
        y_min = int(round(min(ys)))
        y_max = int(round(max(ys)))

        if x_max <= x_min or y_max <= y_min:
            return None

        return [x_min, x_max, y_min, y_max]

    @staticmethod
    def _normalize_detect_output(result: Any) -> dict[str, list]:
        if not isinstance(result, (tuple, list)) or len(result) < 2:
            raise ValueError(
                "easyocr.Reader.detect returned malformed output"
            )

        horizontal_list = result[0] if result[0] is not None else []
        free_list = result[1] if result[1] is not None else []

        # EasyOCR may return nested batch output for a single image.
        if (
            isinstance(horizontal_list, list)
            and len(horizontal_list) == 1
            and isinstance(horizontal_list[0], list)
        ):
            horizontal_list = horizontal_list[0]

        if (
            isinstance(free_list, list)
            and len(free_list) == 1
            and isinstance(free_list[0], list)
        ):
            free_list = free_list[0]

        return {
            "horizontal_list": horizontal_list,
            "free_list": free_list,
        }
