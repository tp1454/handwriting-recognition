"""Core service logic for classification, similarity, and sheet scoring."""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from src.data.processing import (
    preprocess,
    split_big_boxes_and_score_similarity,
)

CHARSET = (
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)


def decode_base64_image(image_base64: str) -> np.ndarray:
    """Decode base64 image payload into an RGB numpy array."""
    try:
        raw = base64.b64decode(image_base64, validate=True)
    except (
        Exception
    ) as exc:  # pragma: no cover - decoder error specifics vary
        raise ValueError("Invalid base64 image payload") from exc

    return decode_image_bytes(raw)


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes into an RGB numpy array."""
    if not image_bytes:
        raise ValueError("Uploaded image is empty")

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Unable to decode uploaded image") from exc

    return np.asarray(image)


def _prepare_model_input(image: np.ndarray) -> torch.Tensor:
    """Convert image array into model-ready tensor shape (B, C, H, W)."""
    tensor = preprocess(image, target_size=28, return_type="torch")

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError("Unexpected preprocessed tensor shape")

    return tensor.float()


def _normalize_tensor_output(output: Any) -> torch.Tensor:
    """Normalize model outputs to a tensor for downstream math."""
    if isinstance(output, torch.Tensor):
        tensor = output
    elif isinstance(output, (list, tuple)) and output:
        first = output[0]
        tensor = (
            first
            if isinstance(first, torch.Tensor)
            else torch.tensor(first)
        )
    else:
        tensor = torch.tensor(output)

    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)

    if tensor.ndim > 2:
        tensor = tensor.reshape(tensor.shape[0], -1)

    return tensor.float()


def classify_image(
    model: Any, image: np.ndarray
) -> tuple[str, float]:
    """Predict character and confidence for one image array."""
    model_input = _prepare_model_input(image)

    with torch.no_grad():
        logits = _normalize_tensor_output(model(model_input))

    if logits.shape[-1] != len(CHARSET):
        if logits.shape[-1] < len(CHARSET):
            pad_width = len(CHARSET) - logits.shape[-1]
            logits = torch.nn.functional.pad(logits, (0, pad_width))
        else:
            logits = logits[:, : len(CHARSET)]

    probabilities = torch.nn.functional.softmax(logits, dim=1)
    top_index = int(torch.argmax(probabilities, dim=1)[0].item())
    confidence = float(probabilities[0, top_index].item())

    return CHARSET[top_index], confidence


def classify_base64(
    model: Any, image_base64: str
) -> tuple[str, float]:
    """Classify base64 image payload."""
    image = decode_base64_image(image_base64)
    return classify_image(model, image)


def compute_similarity(
    encoder: Any, image1: np.ndarray, image2: np.ndarray
) -> float:
    """Compute cosine similarity score scaled to [0, 100]."""
    tensor1 = _prepare_model_input(image1)
    tensor2 = _prepare_model_input(image2)

    with torch.no_grad():
        embedding1 = _normalize_tensor_output(encoder(tensor1))
        embedding2 = _normalize_tensor_output(encoder(tensor2))

    embedding1 = embedding1.reshape(embedding1.shape[0], -1)
    embedding2 = embedding2.reshape(embedding2.shape[0], -1)

    min_width = min(embedding1.shape[1], embedding2.shape[1])
    if min_width == 0:
        return 0.0

    embedding1 = embedding1[:, :min_width]
    embedding2 = embedding2[:, :min_width]

    similarity = torch.nn.functional.cosine_similarity(
        embedding1,
        embedding2,
        dim=1,
    )
    scaled = (float(similarity[0].item()) + 1.0) * 50.0
    return float(np.clip(scaled, 0.0, 100.0))


def compute_similarity_from_base64(
    encoder: Any,
    image1_base64: str,
    image2_base64: str,
) -> float:
    """Compute similarity score from two base64 payloads."""
    image1 = decode_base64_image(image1_base64)
    image2 = decode_base64_image(image2_base64)
    return compute_similarity(encoder, image1, image2)


def _sort_groups(
    groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sort detected groups by top-to-bottom, left-to-right order."""
    return sorted(
        groups,
        key=lambda group: (
            int(group.get("source_box", [0, 0, 0, 0])[2]),
            int(group.get("source_box", [0, 0, 0, 0])[0]),
        ),
    )


def _box_iou(box_a: list[int], box_b: list[int]) -> float:
    """Compute IoU between [x_min, x_max, y_min, y_max] boxes."""
    ax0, ax1, ay0, ay1 = [int(value) for value in box_a[:4]]
    bx0, bx1, by0, by1 = [int(value) for value in box_b[:4]]

    inter_x0 = max(ax0, bx0)
    inter_x1 = min(ax1, bx1)
    inter_y0 = max(ay0, by0)
    inter_y1 = min(ay1, by1)

    inter_w = max(0, inter_x1 - inter_x0)
    inter_h = max(0, inter_y1 - inter_y0)
    inter_area = float(inter_w * inter_h)

    area_a = float(max(0, ax1 - ax0) * max(0, ay1 - ay0))
    area_b = float(max(0, bx1 - bx0) * max(0, by1 - by0))
    union = area_a + area_b - inter_area
    if union <= 1e-8:
        return 0.0
    return inter_area / union


def _box_center_distance(box_a: list[int], box_b: list[int]) -> float:
    """Compute Euclidean distance between centers of two boxes."""
    ax0, ax1, ay0, ay1 = [float(value) for value in box_a[:4]]
    bx0, bx1, by0, by1 = [float(value) for value in box_b[:4]]

    center_a = ((ax0 + ax1) / 2.0, (ay0 + ay1) / 2.0)
    center_b = ((bx0 + bx1) / 2.0, (by0 + by1) / 2.0)

    return float(
        np.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1])
    )


def _match_row_label(
    source_box: list[int],
    extracted_boxes: list[dict[str, Any]],
) -> str:
    """Find best OCR label for a row source box using IoU-first matching."""
    best_entry: dict[str, Any] | None = None
    best_iou = -1.0
    best_distance = float("inf")

    for entry in extracted_boxes:
        candidate_box = entry.get("box")
        label = str(entry.get("label", "")).strip()
        if (
            not isinstance(candidate_box, list)
            or len(candidate_box) < 4
            or not label
        ):
            continue

        iou = _box_iou(source_box, candidate_box)
        distance = _box_center_distance(source_box, candidate_box)

        if iou > best_iou:
            best_entry = entry
            best_iou = iou
            best_distance = distance
            continue

        if np.isclose(iou, best_iou) and distance < best_distance:
            best_entry = entry
            best_distance = distance

    if best_entry is None:
        return "UNKNOWN"
    return str(best_entry.get("label", "UNKNOWN"))


def score_handwriting_sheet(
    image_bytes: bytes,
    detector: Any,
    width_ratio: float = 1.8,
    vector_size: int = 28,
) -> dict[str, Any]:
    """Score handwriting sheet rows using split index 0 as reference.

    The response `segment_scores` contains comparison-only values, i.e. scores for
    split segments 1..N-1 against split segment 0 within the same row group.
    """
    image = decode_image_bytes(image_bytes)
    detect_output = detector.detect_char_boxes(image)
    extracted_boxes = detector.extract_labeled_boxes(image)

    split_result = split_big_boxes_and_score_similarity(
        image=image,
        detect_output=detect_output,
        width_ratio=width_ratio,
        vector_size=vector_size,
    )

    groups = _sort_groups(split_result.get("horizontal_groups", []))
    if not groups:
        raise ValueError("No handwriting rows were detected")

    rows: list[dict[str, Any]] = []
    row_scores: list[float] = []

    for row_index, group in enumerate(groups):
        items = group.get("items", [])
        source_box = [
            int(value)
            for value in group.get("source_box", [0, 0, 0, 0])
        ]
        all_segment_scores = [
            float(item.get("similarity_to_first", 0.0))
            for item in items
        ]

        # Expose only non-reference comparisons in API payload.
        segment_scores = all_segment_scores[1:]
        row_score = (
            float(np.mean(segment_scores))
            if segment_scores
            else 100.0
        )
        row_scores.append(row_score)

        ocr_label = _match_row_label(source_box, extracted_boxes)

        rows.append(
            {
                "row_index": row_index,
                "source_box": source_box,
                "ocr_label": ocr_label,
                "split_count": len(items) if items else 1,
                "was_split": bool(group.get("was_split", False)),
                "segment_scores": segment_scores,
                "row_score": row_score,
            }
        )

    overall_score = float(np.mean(row_scores)) if row_scores else 0.0
    return {
        "detected_rows": len(rows),
        "overall_score": overall_score,
        "rows": rows,
        "extracted_boxes": extracted_boxes,
    }
