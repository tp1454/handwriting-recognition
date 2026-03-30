"""EasyOCR text-box detection wrapper for character-region proposals."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


def _import_easyocr_module() -> ModuleType:
    """Import the external easyocr package while guarding against local shadowing."""
    current_file = Path(__file__).resolve()
    loaded = importlib.import_module("easyocr")

    loaded_file = getattr(loaded, "__file__", None)
    if loaded_file and Path(loaded_file).resolve() == current_file:
        raise ImportError(
            "Local module shadowed the easyocr package. "
            "Ensure the external 'easyocr' dependency is importable."
        )

    if not hasattr(loaded, "Reader"):
        raise ImportError("easyocr package does not expose Reader")

    return loaded


class EasyOCRDetector:
    """Character-region detector backed by easyocr.Reader.detect."""

    def __init__(
        self,
        languages: Sequence[str] | None = None,
        gpu: bool = False,
        reader_kwargs: dict[str, Any] | None = None,
    ) -> None:
        easyocr_module = _import_easyocr_module()
        reader_options = {"gpu": gpu}
        if reader_kwargs:
            reader_options.update(reader_kwargs)

        selected_languages = list(languages) if languages else ["en"]
        self._reader: Any = easyocr_module.Reader(
            selected_languages, **reader_options
        )

    @property
    def reader(self) -> Any:
        """Expose underlying easyocr reader for advanced usage and testing."""
        return self._reader

    def detect_char_boxes(
        self,
        image: str | bytes | Any,
        min_size: int = 10,
        text_threshold: float = 0.7,
        low_text: float = 0.4,
        link_threshold: float = 0.4,
        canvas_size: int = 2560,
        mag_ratio: float = 1.0,
        slope_ths: float = 0.1,
        ycenter_ths: float = 0.5,
        height_ths: float = 0.5,
        width_ths: float = 0.5,
        add_margin: float = 0.1,
        optimal_num_chars: int | None = None,
    ) -> dict[str, list[Any]]:
        """Detect text boxes and return EasyOCR raw output lists.

        Returns a dictionary with:
        - horizontal_list: rectangular boxes in [x_min, x_max, y_min, y_max]
        - free_list: quadrilateral boxes in [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        """
        result = self._reader.detect(
            image,
            min_size=min_size,
            text_threshold=text_threshold,
            low_text=low_text,
            link_threshold=link_threshold,
            canvas_size=canvas_size,
            mag_ratio=mag_ratio,
            slope_ths=slope_ths,
            ycenter_ths=ycenter_ths,
            height_ths=height_ths,
            width_ths=width_ths,
            add_margin=add_margin,
            optimal_num_chars=optimal_num_chars,
        )

        if not isinstance(result, (tuple, list)) or len(result) < 2:
            raise ValueError(
                "easyocr.Reader.detect returned malformed output"
            )

        horizontal_list = result[0] if result[0] is not None else []
        free_list = result[1] if result[1] is not None else []

        return {
            "horizontal_list": horizontal_list,
            "free_list": free_list,
        }
