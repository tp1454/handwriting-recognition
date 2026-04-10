"""Request and response schemas for API endpoints."""

from __future__ import annotations

import base64

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)


def _validate_base64_image(value: str) -> str:
    """Ensure a non-empty string is valid base64 image payload."""
    if not isinstance(value, str):
        raise ValueError("Image must be a base64 string")
    if not value.strip():
        raise ValueError("Image cannot be empty")

    try:
        base64.b64decode(value, validate=True)
    except (
        Exception
    ) as exc:  # pragma: no cover - exact decoder error varies
        raise ValueError("Invalid base64 image") from exc

    return value


class ClassifyRequest(BaseModel):
    """Request payload for single-image classification."""

    image: str

    @field_validator("image")
    @classmethod
    def validate_image(cls, value: str) -> str:
        return _validate_base64_image(value)


class SimilarityRequest(BaseModel):
    """Request payload for pairwise similarity."""

    image1: str
    image2: str

    @field_validator("image1", "image2")
    @classmethod
    def validate_images(cls, value: str) -> str:
        return _validate_base64_image(value)


class AnalyzeRequest(BaseModel):
    """Request payload for combined analysis endpoint."""

    image: str
    reference_id: str | None = None

    @field_validator("image")
    @classmethod
    def validate_image(cls, value: str) -> str:
        return _validate_base64_image(value)


class ClassifyResponse(BaseModel):
    """Response payload for classification."""

    character: str = Field(min_length=1, max_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class SimilarityResponse(BaseModel):
    """Response payload for similarity and combined analysis."""

    character: str = Field(min_length=1, max_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    similarity: float = Field(ge=0.0, le=100.0)
    reference_id: str


class ExtractedBox(BaseModel):
    """Raw EasyOCR extracted box and label payload."""

    index: int = Field(ge=0)
    box: list[int]
    label: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("box")
    @classmethod
    def validate_box(cls, value: list[int]) -> list[int]:
        if len(value) != 4:
            raise ValueError(
                "box must be [x_min, x_max, y_min, y_max]"
            )
        return value

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        label = value.strip()
        if not label:
            raise ValueError("label cannot be empty")
        return label


class SheetCharacterDetail(BaseModel):
    """Split-segment score details used by expanded score rows in the UI."""

    split_index: int = Field(ge=0)
    box: list[int]
    similarity_to_reference: float = Field(ge=0.0, le=100.0)
    is_reference: bool
    label: str = Field(min_length=1)
    image: str | None = Field(
        default=None,
        description="Base64 PNG crop for this split segment",
    )

    @field_validator("box")
    @classmethod
    def validate_box(cls, value: list[int]) -> list[int]:
        if len(value) != 4:
            raise ValueError(
                "box must be [x_min, x_max, y_min, y_max]"
            )
        return value

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        label = value.strip()
        if not label:
            raise ValueError("label cannot be empty")
        return label

    @field_validator("image")
    @classmethod
    def validate_image(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _validate_base64_image(value)


class SheetRowDetail(BaseModel):
    """Expanded detail payload for one score row."""

    row_image: str | None = Field(
        default=None,
        description="Base64 PNG crop for source_box",
    )
    row_image_mime_type: str = Field(
        default="image/png", min_length=1
    )
    split_images: dict[str, str | None] = Field(
        default_factory=dict,
        description=(
            "Map of split index to base64 PNG crop for each row split"
        ),
    )
    characters: list[SheetCharacterDetail] = Field(
        default_factory=list
    )

    @field_validator("row_image")
    @classmethod
    def validate_row_image(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _validate_base64_image(value)

    @field_validator("row_image_mime_type")
    @classmethod
    def validate_row_image_mime_type(cls, value: str) -> str:
        mime = value.strip().lower()
        if not mime:
            raise ValueError("row_image_mime_type cannot be empty")
        return mime

    @field_validator("split_images")
    @classmethod
    def validate_split_images(
        cls,
        value: dict[str, str | None],
    ) -> dict[str, str | None]:
        normalized: dict[str, str | None] = {}
        for key, image_payload in value.items():
            split_key = str(key).strip()
            if not split_key:
                raise ValueError("split_images keys cannot be empty")

            if image_payload is None:
                normalized[split_key] = None
                continue

            normalized[split_key] = _validate_base64_image(
                str(image_payload)
            )

        return normalized


class SheetRowScore(BaseModel):
    """Per-row score details for handwriting sheet scoring.

    `ocr_label` is the first visual character from the OCR label matched to
    split segment 0 (reference char). If no label matches, it remains
    `UNKNOWN`.
    `segment_scores` contains only non-reference comparisons for a split row,
    where each value scores split segment 1..N-1 against split segment 0.
    """

    row_index: int = Field(ge=0)
    source_box: list[int]
    ocr_label: str = Field(
        min_length=1,
        description=(
            "First visual character for the matched reference label "
            "(split index 0), or UNKNOWN when unmatched"
        ),
    )
    split_count: int = Field(ge=1)
    was_split: bool
    segment_scores: list[float]
    row_score: float = Field(ge=0.0, le=100.0)
    details: SheetRowDetail | None = None

    @field_validator("source_box")
    @classmethod
    def validate_source_box(cls, value: list[int]) -> list[int]:
        if len(value) != 4:
            raise ValueError(
                "source_box must be [x_min, x_max, y_min, y_max]"
            )
        return value

    @field_validator("ocr_label")
    @classmethod
    def validate_ocr_label(cls, value: str) -> str:
        label = value.strip()
        if not label:
            raise ValueError("ocr_label cannot be empty")
        return label

    @field_validator("segment_scores")
    @classmethod
    def validate_segment_scores(
        cls, value: list[float]
    ) -> list[float]:
        bounded = []
        for score in value:
            numeric_score = float(score)
            if numeric_score < 0.0 or numeric_score > 100.0:
                raise ValueError(
                    "segment_scores values must be between 0 and 100"
                )
            bounded.append(numeric_score)
        return bounded


class SheetScoreResponse(BaseModel):
    """Response payload for sheet upload scoring."""

    detected_rows: int = Field(ge=0)
    overall_score: float = Field(ge=0.0, le=100.0)
    rows: list[SheetRowScore]
    extracted_boxes: list[ExtractedBox] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_rows(self) -> "SheetScoreResponse":
        if self.detected_rows != len(self.rows):
            raise ValueError("detected_rows must match row count")
        return self


class SheetFontOption(BaseModel):
    """Server-hosted font option for sheet generation."""

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)


class SheetLayoutDefaults(BaseModel):
    """Default and editable sheet layout values exposed to the web client."""

    font_size: int = Field(ge=1, le=200)
    line_spacing: int = Field(ge=1, le=400)
    word_spacing: int = Field(ge=1, le=400)
    margin_left: int = Field(ge=0, le=2000)
    margin_right: int = Field(ge=0, le=2000)
    margin_top: int = Field(ge=0, le=2000)
    margin_bottom: int = Field(ge=0, le=2000)
    divide_horizontal: int = Field(ge=0, le=2000)
    divide_vertical: int = Field(ge=0, le=2000)
    show_vertical_line: bool = Field(default=False)


class SheetOptionsResponse(BaseModel):
    """Configuration payload used by the web app sheet-creation form."""

    available_languages: list[str] = Field(default_factory=list)
    default_language: str = Field(min_length=2)
    defaults: SheetLayoutDefaults
    server_fonts: list[SheetFontOption] = Field(default_factory=list)
    max_custom_text_length: int = Field(ge=1, le=100000)

    @field_validator("available_languages")
    @classmethod
    def validate_available_languages(
        cls, value: list[str]
    ) -> list[str]:
        allowed = {"auto", "en", "vi"}
        normalized = [str(item).strip().lower() for item in value]
        if not normalized:
            raise ValueError("available_languages cannot be empty")
        invalid = [item for item in normalized if item not in allowed]
        if invalid:
            raise ValueError(
                "available_languages contains unsupported values"
            )
        return normalized

    @field_validator("default_language")
    @classmethod
    def validate_default_language(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"auto", "en", "vi"}:
            raise ValueError(
                "default_language must be auto, en, or vi"
            )
        return normalized


class SheetCreateResponse(BaseModel):
    """Response payload for generated handwriting sheet artifacts."""

    language: str = Field(min_length=2)
    font_source: str = Field(min_length=1)
    font_name: str = Field(min_length=1)
    pdf_url: str = Field(min_length=1)
    preview_image_url: str = Field(min_length=1)
    pdf_filename: str = Field(min_length=1)
    preview_image_filename: str = Field(min_length=1)

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"en", "vi"}:
            raise ValueError("language must be en or vi")
        return normalized

    @field_validator("font_source")
    @classmethod
    def validate_font_source(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"upload", "server"}:
            raise ValueError("font_source must be upload or server")
        return normalized


class HealthResponse(BaseModel):
    """Health endpoint response."""

    status: str
    model_loaded: bool
    version: str
