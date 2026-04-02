"""Request and response schemas for API endpoints."""

from __future__ import annotations

import base64
from typing import Any

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


class PredictionItem(BaseModel):
    """Single prediction item used in top-k responses."""

    character: str = Field(min_length=1, max_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class TopKResponse(BaseModel):
    """Top-k prediction response sorted by confidence descending."""

    predictions: list[PredictionItem]

    @model_validator(mode="after")
    def sort_predictions(self) -> "TopKResponse":
        self.predictions = sorted(
            self.predictions,
            key=lambda item: float(item.confidence),
            reverse=True,
        )
        return self


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


class SheetRowScore(BaseModel):
    """Per-row score details for handwriting sheet scoring.

    `segment_scores` contains only non-reference comparisons for a split row,
    where each value scores split segment 1..N-1 against split segment 0.
    """

    row_index: int = Field(ge=0)
    source_box: list[int]
    ocr_label: str = Field(min_length=1)
    split_count: int = Field(ge=1)
    was_split: bool
    segment_scores: list[float]
    row_score: float = Field(ge=0.0, le=100.0)

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


class HealthResponse(BaseModel):
    """Health endpoint response."""

    status: str
    model_loaded: bool
    version: str


def model_to_dict(model: BaseModel) -> dict[str, Any]:
    """Compatibility helper used by debug/testing code paths."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()
