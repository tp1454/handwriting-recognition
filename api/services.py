"""Core service logic for classification, similarity, and sheet scoring."""

from __future__ import annotations

import base64
import io
import logging
import os
import unicodedata
import uuid
from pathlib import Path
from time import perf_counter, process_time
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from src.data.processing import (
    preprocess,
    split_big_boxes_and_score_similarity,
)
from src.easy_ocr.easyocr import EASYOCR_DETECT_DEFAULTS
from src.utils.handwriting_sheet_generation import (
    generate_handwriting_sheet,
)

CHARSET = (
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)
_FONT_EXTENSIONS = {".ttf", ".otf"}
_LANGUAGES = {"auto", "en", "vi"}
_SHEET_OVERRIDE_FIELDS = (
    "font_size",
    "line_spacing",
    "word_spacing",
    "margin_left",
    "margin_right",
    "margin_top",
    "margin_bottom",
    "divide_horizontal",
    "divide_vertical",
)
_SHEET_BOOLEAN_OVERRIDE_FIELDS = ("show_vertical_line",)

logger = logging.getLogger(__name__)


def _is_env_flag_enabled(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _safe_float(
    raw_value: Any,
    *,
    fallback: float,
    min_value: float,
    max_value: float,
) -> float:
    try:
        parsed = float(raw_value)
    except (TypeError, ValueError):
        parsed = fallback
    return float(np.clip(parsed, min_value, max_value))


def _safe_optional_int(
    raw_value: Any,
    *,
    fallback: int | None,
    min_value: int,
    max_value: int,
) -> int | None:
    if raw_value in (None, ""):
        return fallback
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        return fallback
    return int(np.clip(parsed, min_value, max_value))


def _safe_bool(raw_value: Any, *, fallback: bool) -> bool:
    if raw_value in (None, ""):
        return fallback
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, (int, float)):
        return bool(raw_value)

    normalized = str(raw_value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("show_vertical_line must be a boolean")


def _build_easyocr_detect_kwargs(
    config: Any | None,
) -> dict[str, Any]:
    defaults = dict(EASYOCR_DETECT_DEFAULTS)

    if config is None:
        return defaults

    inference = getattr(config, "inference", None)
    detect_config = (
        getattr(inference, "easyocr_detect", None)
        if inference is not None
        else None
    )
    if detect_config is None:
        return defaults

    return {
        "min_size": _safe_int(
            getattr(detect_config, "min_size", defaults["min_size"]),
            fallback=int(defaults["min_size"]),
            min_value=1,
            max_value=4096,
        ),
        "text_threshold": _safe_float(
            getattr(
                detect_config,
                "text_threshold",
                defaults["text_threshold"],
            ),
            fallback=float(defaults["text_threshold"]),
            min_value=0.0,
            max_value=1.0,
        ),
        "low_text": _safe_float(
            getattr(detect_config, "low_text", defaults["low_text"]),
            fallback=float(defaults["low_text"]),
            min_value=0.0,
            max_value=1.0,
        ),
        "link_threshold": _safe_float(
            getattr(
                detect_config,
                "link_threshold",
                defaults["link_threshold"],
            ),
            fallback=float(defaults["link_threshold"]),
            min_value=0.0,
            max_value=1.0,
        ),
        "canvas_size": _safe_int(
            getattr(
                detect_config,
                "canvas_size",
                defaults["canvas_size"],
            ),
            fallback=int(defaults["canvas_size"]),
            min_value=256,
            max_value=8192,
        ),
        "mag_ratio": _safe_float(
            getattr(
                detect_config, "mag_ratio", defaults["mag_ratio"]
            ),
            fallback=float(defaults["mag_ratio"]),
            min_value=0.1,
            max_value=10.0,
        ),
        "slope_ths": _safe_float(
            getattr(
                detect_config, "slope_ths", defaults["slope_ths"]
            ),
            fallback=float(defaults["slope_ths"]),
            min_value=0.0,
            max_value=1.0,
        ),
        "ycenter_ths": _safe_float(
            getattr(
                detect_config,
                "ycenter_ths",
                defaults["ycenter_ths"],
            ),
            fallback=float(defaults["ycenter_ths"]),
            min_value=0.0,
            max_value=1.0,
        ),
        "height_ths": _safe_float(
            getattr(
                detect_config,
                "height_ths",
                defaults["height_ths"],
            ),
            fallback=float(defaults["height_ths"]),
            min_value=0.0,
            max_value=1.0,
        ),
        "width_ths": _safe_float(
            getattr(
                detect_config, "width_ths", defaults["width_ths"]
            ),
            fallback=float(defaults["width_ths"]),
            min_value=0.0,
            max_value=1.0,
        ),
        "add_margin": _safe_float(
            getattr(
                detect_config,
                "add_margin",
                defaults["add_margin"],
            ),
            fallback=float(defaults["add_margin"]),
            min_value=0.0,
            max_value=1.0,
        ),
        "optimal_num_chars": _safe_optional_int(
            getattr(
                detect_config,
                "optimal_num_chars",
                defaults["optimal_num_chars"],
            ),
            fallback=defaults["optimal_num_chars"],
            min_value=1,
            max_value=256,
        ),
    }


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


def _match_box_label(
    target_box: list[int],
    extracted_boxes: list[dict[str, Any]],
) -> str:
    """Find best OCR label for a target box using IoU-first matching."""
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

        iou = _box_iou(target_box, candidate_box)
        distance = _box_center_distance(target_box, candidate_box)

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


def _is_variation_selector(char: str) -> bool:
    codepoint = ord(char)
    return (0xFE00 <= codepoint <= 0xFE0F) or (
        0xE0100 <= codepoint <= 0xE01EF
    )


def _first_grapheme(value: str) -> str:
    """Return the first visual character (grapheme-like cluster)."""
    normalized = unicodedata.normalize("NFC", value)
    if not normalized:
        return ""

    grapheme_chars = [normalized[0]]
    index = 1
    while index < len(normalized):
        char = normalized[index]
        if unicodedata.combining(char) > 0 or _is_variation_selector(
            char
        ):
            grapheme_chars.append(char)
            index += 1
            continue

        if char == "\u200d":
            grapheme_chars.append(char)
            index += 1
            if index >= len(normalized):
                break

            grapheme_chars.append(normalized[index])
            index += 1
            while index < len(normalized):
                join_char = normalized[index]
                if unicodedata.combining(
                    join_char
                ) > 0 or _is_variation_selector(join_char):
                    grapheme_chars.append(join_char)
                    index += 1
                    continue
                break
            continue

        break

    return "".join(grapheme_chars)


def _format_row_ocr_label(raw_label: str) -> str:
    """Format row label as first grapheme while preserving UNKNOWN."""
    label = str(raw_label).strip()
    if not label:
        return "UNKNOWN"
    if label == "UNKNOWN":
        return "UNKNOWN"

    first_grapheme = _first_grapheme(label)
    return first_grapheme or "UNKNOWN"


def _coerce_box(
    raw_box: Any,
    *,
    row_index: int,
    field_name: str,
) -> list[int]:
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) < 4:
        raise ValueError(
            f"Row {row_index}: {field_name} must be [x_min, x_max, y_min, y_max]"
        )

    try:
        return [int(raw_box[idx]) for idx in range(4)]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Row {row_index}: {field_name} must contain integers"
        ) from exc


def _coerce_similarity_score(
    raw_score: Any,
    *,
    row_index: int,
    split_index: int,
) -> float:
    try:
        score = float(raw_score)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Row {row_index} split {split_index}: similarity_to_first must be numeric"
        ) from exc

    if score < 0.0 or score > 100.0:
        raise ValueError(
            f"Row {row_index} split {split_index}: similarity_to_first must be between 0 and 100"
        )
    return score


def _validate_row_items(
    raw_items: Any,
    *,
    row_index: int,
) -> list[dict[str, Any]]:
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError(
            f"Row {row_index}: items must be a non-empty list"
        )

    validated_items: list[dict[str, Any]] = []
    seen_split_indices: set[int] = set()

    for item_position, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise ValueError(
                f"Row {row_index}: items[{item_position}] must be an object"
            )

        raw_split_index = item.get("split_index")
        try:
            split_index = int(raw_split_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Row {row_index}: items[{item_position}].split_index must be an integer"
            ) from exc

        if split_index < 0:
            raise ValueError(
                f"Row {row_index} split {split_index}: split_index must be >= 0"
            )
        if split_index in seen_split_indices:
            raise ValueError(
                f"Row {row_index}: split_index {split_index} is duplicated"
            )
        seen_split_indices.add(split_index)

        raw_is_reference = item.get("is_reference")
        if not isinstance(raw_is_reference, bool):
            raise ValueError(
                f"Row {row_index} split {split_index}: is_reference must be a boolean"
            )

        expected_is_reference = split_index == 0
        if raw_is_reference != expected_is_reference:
            raise ValueError(
                f"Row {row_index} split {split_index}: is_reference must be {expected_is_reference}"
            )

        if "similarity_to_first" not in item:
            raise ValueError(
                f"Row {row_index} split {split_index}: missing similarity_to_first"
            )

        validated_items.append(
            {
                "split_index": split_index,
                "box": _coerce_box(
                    item.get("box"),
                    row_index=row_index,
                    field_name=f"items[{item_position}].box",
                ),
                "is_reference": raw_is_reference,
                "similarity_to_first": _coerce_similarity_score(
                    item["similarity_to_first"],
                    row_index=row_index,
                    split_index=split_index,
                ),
            }
        )

    validated_items.sort(key=lambda item: int(item["split_index"]))
    split_indices = [
        int(item["split_index"]) for item in validated_items
    ]
    expected_indices = list(range(len(validated_items)))

    if split_indices != expected_indices:
        raise ValueError(
            f"Row {row_index}: split_index values must be contiguous starting at 0"
        )
    if validated_items[0]["is_reference"] is not True:
        raise ValueError(
            f"Row {row_index}: split index 0 must be the reference item"
        )

    return validated_items


def _resolve_reference_box(
    group: dict[str, Any],
    *,
    row_index: int,
    reference_item_box: list[int],
) -> list[int]:
    explicit_reference_box = group.get("reference_box")
    if explicit_reference_box is None:
        return reference_item_box

    reference_box = _coerce_box(
        explicit_reference_box,
        row_index=row_index,
        field_name="reference_box",
    )

    if reference_box != reference_item_box:
        raise ValueError(
            f"Row {row_index}: reference_box must match items[0].box"
        )

    return reference_box


def _validate_was_split(
    group: dict[str, Any],
    *,
    row_index: int,
    split_count: int,
) -> bool:
    raw_was_split = group.get("was_split")
    if not isinstance(raw_was_split, bool):
        raise ValueError(
            f"Row {row_index}: was_split must be a boolean"
        )

    expected_was_split = split_count > 1
    if raw_was_split != expected_was_split:
        raise ValueError(
            f"Row {row_index}"
            + f": was_split must be {expected_was_split} when split_count is {split_count}"
        )

    return raw_was_split


def _clip_box_to_image(
    box: list[int],
    *,
    image_h: int,
    image_w: int,
) -> list[int] | None:
    x_min, x_max, y_min, y_max = [int(value) for value in box]
    clipped_box = [
        max(0, min(image_w, x_min)),
        max(0, min(image_w, x_max)),
        max(0, min(image_h, y_min)),
        max(0, min(image_h, y_max)),
    ]
    if clipped_box[1] <= clipped_box[0]:
        return None
    if clipped_box[3] <= clipped_box[2]:
        return None
    return clipped_box


def _normalize_candidate_box_for_target(
    raw_box: Any,
    *,
    target_box: list[int],
    image_h: int,
    image_w: int,
) -> list[int] | None:
    """Normalize OCR candidate box from crop-local or global coordinates."""
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) < 4:
        return None

    try:
        parsed_box = [
            int(round(float(raw_box[idx]))) for idx in range(4)
        ]
    except (TypeError, ValueError):
        return None

    tx0, tx1, ty0, ty1 = target_box
    target_w = tx1 - tx0
    target_h = ty1 - ty0
    if target_w <= 0 or target_h <= 0:
        return None

    local_candidate = _clip_box_to_image(
        parsed_box,
        image_h=target_h,
        image_w=target_w,
    )
    local_mapped = None
    if local_candidate is not None:
        local_mapped = [
            local_candidate[0] + tx0,
            local_candidate[1] + tx0,
            local_candidate[2] + ty0,
            local_candidate[3] + ty0,
        ]
        local_mapped = _clip_box_to_image(
            local_mapped,
            image_h=image_h,
            image_w=image_w,
        )

    global_candidate = _clip_box_to_image(
        parsed_box,
        image_h=image_h,
        image_w=image_w,
    )

    if local_mapped is None:
        return global_candidate
    if global_candidate is None:
        return local_mapped

    local_iou = _box_iou(local_mapped, target_box)
    global_iou = _box_iou(global_candidate, target_box)
    if local_iou > global_iou:
        return local_mapped
    if global_iou > local_iou:
        return global_candidate

    local_distance = _box_center_distance(local_mapped, target_box)
    global_distance = _box_center_distance(
        global_candidate,
        target_box,
    )
    if local_distance <= global_distance:
        return local_mapped
    return global_candidate


def _extract_best_labeled_boxes_for_targets(
    image: np.ndarray,
    detector: Any,
    target_boxes: list[list[int]],
) -> list[dict[str, Any]]:
    """Run OCR per target box and keep the best matched label for each target."""
    image_h, image_w = image.shape[:2]
    extracted: list[dict[str, Any]] = []

    for target_index, target_box in enumerate(target_boxes):
        bounded_target_box = _clip_box_to_image(
            target_box,
            image_h=image_h,
            image_w=image_w,
        )
        if bounded_target_box is None:
            continue

        tx0, tx1, ty0, ty1 = bounded_target_box
        crop = image[ty0:ty1, tx0:tx1]
        if crop.size == 0:
            continue

        raw_items = detector.extract_labeled_boxes(crop)
        if not isinstance(raw_items, list):
            continue

        best_candidate: dict[str, Any] | None = None
        best_iou = -1.0
        best_confidence = -1.0
        best_distance = float("inf")

        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue

            mapped_box = _normalize_candidate_box_for_target(
                raw_item.get("box"),
                target_box=bounded_target_box,
                image_h=image_h,
                image_w=image_w,
            )
            if mapped_box is None:
                continue

            label = str(raw_item.get("label", "")).strip()
            if not label:
                continue

            try:
                confidence = float(raw_item.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            confidence = float(np.clip(confidence, 0.0, 1.0))

            iou = _box_iou(mapped_box, bounded_target_box)
            distance = _box_center_distance(
                mapped_box,
                bounded_target_box,
            )

            is_better = iou > best_iou
            if np.isclose(iou, best_iou):
                if confidence > best_confidence:
                    is_better = True
                elif np.isclose(confidence, best_confidence):
                    is_better = distance < best_distance

            if not is_better:
                continue

            best_candidate = {
                "index": len(extracted),
                "box": mapped_box,
                "label": label,
                "confidence": confidence,
                "target_index": target_index,
            }
            best_iou = iou
            best_confidence = confidence
            best_distance = distance

        if best_candidate is not None:
            extracted.append(best_candidate)

    return extracted


def _encode_box_crop_png_base64(
    image: np.ndarray,
    box: list[int],
) -> str | None:
    image_h, image_w = image.shape[:2]
    bounded_box = _clip_box_to_image(
        box,
        image_h=image_h,
        image_w=image_w,
    )
    if bounded_box is None:
        return None

    x_min, x_max, y_min, y_max = bounded_box
    crop = image[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None

    try:
        crop_image = Image.fromarray(crop)
    except (TypeError, ValueError, OSError):
        return None

    with io.BytesIO() as buffer:
        crop_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def _build_split_image_map(
    *,
    image: np.ndarray,
    items: list[dict[str, Any]],
) -> dict[str, str | None]:
    """Build split-index keyed map of extracted split PNG crops."""
    split_images: dict[str, str | None] = {}
    for item in items:
        split_index = int(item["split_index"])
        split_images[str(split_index)] = _encode_box_crop_png_base64(
            image,
            list(item["box"]),
        )
    return split_images


def _build_row_detail_payload(
    *,
    image: np.ndarray,
    source_box: list[int],
    items: list[dict[str, Any]],
    extracted_boxes: list[dict[str, Any]],
) -> dict[str, Any]:
    split_images = _build_split_image_map(
        image=image,
        items=items,
    )

    characters: list[dict[str, Any]] = []
    for item in items:
        split_box = list(item["box"])
        matched_split_label = _match_box_label(
            split_box,
            extracted_boxes,
        )
        split_key = str(int(item["split_index"]))
        characters.append(
            {
                "split_index": int(item["split_index"]),
                "box": split_box,
                "similarity_to_reference": float(
                    item["similarity_to_first"]
                ),
                "is_reference": bool(item["is_reference"]),
                "label": _format_row_ocr_label(matched_split_label),
                "image": split_images.get(split_key),
            }
        )

    return {
        "row_image": _encode_box_crop_png_base64(image, source_box),
        "row_image_mime_type": "image/png",
        "split_images": split_images,
        "characters": characters,
    }


def score_handwriting_sheet(
    image_bytes: bytes,
    detector: Any,
    config: Any | None = None,
    width_ratio: float = 1.8,
    vector_size: int = 28,
    include_details: bool = False,
) -> dict[str, Any]:
    """Score handwriting sheet rows using split index 0 as reference.

    The response `segment_scores` contains only comparison values for split
    segments 1..N-1 against split segment 0 within each row group.
    When `include_details` is true, each row includes base64 PNG crops for
    the full row and split segments with per-segment scores.

    Invariants enforced:
    - Row items must provide contiguous split_index values starting at 0.
    - Split index 0 must be the unique reference item.
    - similarity_to_first values must be numeric and bounded in [0, 100].
    - was_split must be consistent with split_count.
    """
    profile_enabled = _is_env_flag_enabled(
        "API_PROFILE_SCORE", default=False
    )
    skip_readtext = _is_env_flag_enabled(
        "API_SCORE_SKIP_READTEXT", default=False
    )

    stage_wall_ms: dict[str, float] = {}
    stage_cpu_ms: dict[str, float] = {}

    def run_stage(stage_name: str, callback: Any) -> Any:
        if not profile_enabled:
            return callback()

        wall_start = perf_counter()
        cpu_start = process_time()
        try:
            return callback()
        finally:
            stage_wall_ms[stage_name] = (
                perf_counter() - wall_start
            ) * 1000.0
            stage_cpu_ms[stage_name] = (
                process_time() - cpu_start
            ) * 1000.0

    total_wall_start = perf_counter() if profile_enabled else 0.0
    total_cpu_start = process_time() if profile_enabled else 0.0

    image = run_stage(
        "decode_image",
        lambda: decode_image_bytes(image_bytes),
    )
    detect_kwargs = run_stage(
        "build_detect_kwargs",
        lambda: _build_easyocr_detect_kwargs(config),
    )
    detect_output = run_stage(
        "easyocr_detect",
        lambda: detector.detect_char_boxes(
            image,
            **detect_kwargs,
        ),
    )

    initial_split_result = run_stage(
        "split_geometry",
        lambda: split_big_boxes_and_score_similarity(
            image=image,
            detect_output=detect_output,
            width_ratio=width_ratio,
            vector_size=vector_size,
            extracted_labeled_boxes=None,
        ),
    )

    initial_groups = _sort_groups(
        initial_split_result.get("horizontal_groups", [])
    )
    if not initial_groups:
        raise ValueError("No handwriting rows were detected")

    if skip_readtext:
        ocr_hint_boxes: list[dict[str, Any]] = []
    else:
        source_boxes = [
            _coerce_box(
                group.get("source_box"),
                row_index=row_index,
                field_name="source_box",
            )
            for row_index, group in enumerate(initial_groups)
        ]
        ocr_hint_boxes = run_stage(
            "easyocr_readtext_rows",
            lambda: _extract_best_labeled_boxes_for_targets(
                image=image,
                detector=detector,
                target_boxes=source_boxes,
            ),
        )

    split_result = run_stage(
        "split_and_similarity",
        lambda: split_big_boxes_and_score_similarity(
            image=image,
            detect_output=detect_output,
            width_ratio=width_ratio,
            vector_size=vector_size,
            extracted_labeled_boxes=ocr_hint_boxes,
        ),
    )

    groups = _sort_groups(split_result.get("horizontal_groups", []))
    if not groups:
        raise ValueError("No handwriting rows were detected")

    if skip_readtext:
        extracted_boxes: list[dict[str, Any]] = []
    else:
        reference_target_boxes: list[list[int]] = []
        for row_index, group in enumerate(groups):
            items = _validate_row_items(
                group.get("items"),
                row_index=row_index,
            )
            reference_box = _resolve_reference_box(
                group,
                row_index=row_index,
                reference_item_box=list(items[0]["box"]),
            )
            reference_target_boxes.append(reference_box)

        extracted_boxes = run_stage(
            "easyocr_readtext_reference_splits",
            lambda: _extract_best_labeled_boxes_for_targets(
                image=image,
                detector=detector,
                target_boxes=reference_target_boxes,
            ),
        )

    detail_label_boxes = list(extracted_boxes)
    detail_label_boxes.extend(ocr_hint_boxes)

    def build_rows_and_scores() -> tuple[
        list[dict[str, Any]],
        list[float],
    ]:
        rows: list[dict[str, Any]] = []
        row_scores: list[float] = []

        for row_index, group in enumerate(groups):
            source_box = _coerce_box(
                group.get("source_box"),
                row_index=row_index,
                field_name="source_box",
            )
            items = _validate_row_items(
                group.get("items"), row_index=row_index
            )
            reference_box = _resolve_reference_box(
                group,
                row_index=row_index,
                reference_item_box=list(items[0]["box"]),
            )

            all_segment_scores = [
                float(item["similarity_to_first"]) for item in items
            ]

            # Expose only non-reference comparisons in API payload.
            segment_scores = all_segment_scores[1:]
            row_score = (
                float(np.mean(segment_scores))
                if segment_scores
                # Unsplit rows have only the reference item and no comparisons.
                else 100.0
            )
            row_scores.append(row_score)

            split_count = len(items)
            was_split = _validate_was_split(
                group,
                row_index=row_index,
                split_count=split_count,
            )

            # Row label is anchored to the reference character (split index 0).
            matched_label = _match_box_label(
                reference_box, extracted_boxes
            )
            ocr_label = _format_row_ocr_label(matched_label)

            details_payload = None
            if include_details:
                details_payload = _build_row_detail_payload(
                    image=image,
                    source_box=source_box,
                    items=items,
                    extracted_boxes=detail_label_boxes,
                )

            rows.append(
                {
                    "row_index": row_index,
                    "source_box": source_box,
                    "ocr_label": ocr_label,
                    "split_count": split_count,
                    "was_split": was_split,
                    "segment_scores": segment_scores,
                    "row_score": row_score,
                    "details": details_payload,
                }
            )

        return rows, row_scores

    rows, row_scores = run_stage(
        "build_response_rows",
        build_rows_and_scores,
    )

    overall_score = run_stage(
        "aggregate_overall",
        lambda: float(np.mean(row_scores)) if row_scores else 0.0,
    )

    if profile_enabled:
        total_wall_ms = (perf_counter() - total_wall_start) * 1000.0
        total_cpu_ms = (process_time() - total_cpu_start) * 1000.0
        cpu_pct = (
            (total_cpu_ms / total_wall_ms) * 100.0
            if total_wall_ms > 1e-9
            else 0.0
        )
        ranked_stages = sorted(
            stage_wall_ms.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        top_stage = ranked_stages[0][0] if ranked_stages else "n/a"
        stage_summary = ", ".join(
            (
                f"{name}={wall_ms:.1f}ms"
                f"(cpu={stage_cpu_ms.get(name, 0.0):.1f}ms)"
            )
            for name, wall_ms in ranked_stages
        )
        logger.info(
            "sheet_score_profile total_wall_ms=%.1f total_cpu_ms=%.1f "
            "cpu_pct=%.1f rows=%d extracted_boxes=%d skip_readtext=%s "
            "top_stage=%s stages=[%s]",
            total_wall_ms,
            total_cpu_ms,
            cpu_pct,
            len(rows),
            len(extracted_boxes),
            skip_readtext,
            top_stage,
            stage_summary,
        )

    return {
        "detected_rows": len(rows),
        "overall_score": overall_score,
        "rows": rows,
        "extracted_boxes": extracted_boxes,
    }


def _sheet_section(config: Any) -> Any:
    sheet = getattr(config, "sheet", None)
    if sheet is None:
        raise ValueError("Sheet configuration is missing")
    return sheet


def _safe_int(
    raw_value: Any,
    *,
    fallback: int,
    min_value: int,
    max_value: int,
) -> int:
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        parsed = fallback
    return int(np.clip(parsed, min_value, max_value))


def _sheet_save_dir(config: Any) -> Path:
    sheet = _sheet_section(config)
    return Path(str(sheet.save_path)).expanduser().resolve()


def _sheet_fonts_dir(config: Any) -> Path:
    sheet = _sheet_section(config)
    raw_fonts_dir = getattr(sheet, "fonts_dir", None)
    if isinstance(raw_fonts_dir, str) and raw_fonts_dir.strip():
        return Path(raw_fonts_dir).expanduser().resolve()
    return _sheet_save_dir(config)


def list_server_fonts(config: Any) -> list[dict[str, str]]:
    fonts_dir = _sheet_fonts_dir(config)
    if not fonts_dir.exists() or not fonts_dir.is_dir():
        return []

    fonts: list[dict[str, str]] = []
    for font_path in sorted(fonts_dir.iterdir()):
        if not font_path.is_file():
            continue
        if font_path.suffix.lower() not in _FONT_EXTENSIONS:
            continue

        label = font_path.stem.replace("_", " ").strip()
        fonts.append(
            {
                "id": font_path.name,
                "label": label or font_path.name,
            }
        )
    return fonts


def build_sheet_options(config: Any) -> dict[str, Any]:
    sheet = _sheet_section(config)

    default_language = (
        str(getattr(sheet, "default_language", "auto"))
        .strip()
        .lower()
    )
    if default_language not in _LANGUAGES:
        default_language = "auto"

    max_custom_text_length = _safe_int(
        getattr(sheet, "max_custom_text_length", 4000),
        fallback=4000,
        min_value=32,
        max_value=100000,
    )

    defaults = {
        "font_size": _safe_int(
            getattr(sheet, "font_size", 15),
            fallback=15,
            min_value=1,
            max_value=200,
        ),
        "line_spacing": _safe_int(
            getattr(sheet, "line_spacing", 20),
            fallback=30,
            min_value=1,
            max_value=400,
        ),
        "word_spacing": _safe_int(
            getattr(sheet, "word_spacing", 20),
            fallback=20,
            min_value=1,
            max_value=400,
        ),
        "margin_left": _safe_int(
            getattr(sheet, "margin_left", 40),
            fallback=40,
            min_value=0,
            max_value=2000,
        ),
        "margin_right": _safe_int(
            getattr(sheet, "margin_right", 555),
            fallback=555,
            min_value=0,
            max_value=2000,
        ),
        "margin_top": _safe_int(
            getattr(sheet, "margin_top", 802),
            fallback=802,
            min_value=0,
            max_value=2000,
        ),
        "margin_bottom": _safe_int(
            getattr(sheet, "margin_bottom", 40),
            fallback=40,
            min_value=0,
            max_value=2000,
        ),
        "divide_horizontal": _safe_int(
            getattr(sheet, "divide_horizontal", 421),
            fallback=421,
            min_value=0,
            max_value=2000,
        ),
        "divide_vertical": _safe_int(
            getattr(sheet, "divide_vertical", 297),
            fallback=297,
            min_value=0,
            max_value=2000,
        ),
        "show_vertical_line": _safe_bool(
            getattr(sheet, "show_vertical_line", False),
            fallback=False,
        ),
    }

    return {
        "available_languages": ["auto", "en", "vi"],
        "default_language": default_language,
        "defaults": defaults,
        "server_fonts": list_server_fonts(config),
        "max_custom_text_length": max_custom_text_length,
    }


def _normalize_overrides(
    overrides: dict[str, Any] | None,
) -> dict[str, int | bool]:
    if not overrides:
        return {}

    normalized: dict[str, int | bool] = {}
    for field in _SHEET_OVERRIDE_FIELDS:
        raw_value = overrides.get(field)
        if raw_value in (None, ""):
            continue
        try:
            normalized[field] = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be an integer") from exc

    for field in _SHEET_BOOLEAN_OVERRIDE_FIELDS:
        raw_value = overrides.get(field)
        if raw_value in (None, ""):
            continue
        normalized[field] = _safe_bool(raw_value, fallback=False)

    return normalized


def _resolve_server_font_path(
    config: Any, server_font_id: str
) -> Path:
    font_id = Path(str(server_font_id).strip()).name
    if not font_id:
        raise ValueError("server_font is required")

    fonts_dir = _sheet_fonts_dir(config)
    candidate = (fonts_dir / font_id).resolve()

    if candidate.parent != fonts_dir.resolve():
        raise ValueError("Invalid server font selection")
    if candidate.suffix.lower() not in _FONT_EXTENSIONS:
        raise ValueError("Unsupported font extension")
    if not candidate.exists() or not candidate.is_file():
        raise ValueError("Selected server font does not exist")

    return candidate


def _save_uploaded_font(
    config: Any,
    *,
    file_bytes: bytes,
    original_filename: str,
) -> Path:
    if not file_bytes:
        raise ValueError("Uploaded font file is empty")

    suffix = Path(original_filename or "").suffix.lower()
    if suffix not in _FONT_EXTENSIONS:
        raise ValueError("Uploaded font must be .ttf or .otf")

    save_dir = _sheet_save_dir(config)
    save_dir.mkdir(parents=True, exist_ok=True)
    temp_name = f"uploaded_font_{uuid.uuid4().hex}{suffix}"
    temp_path = save_dir / temp_name
    temp_path.write_bytes(file_bytes)
    return temp_path


def _normalize_language(language: str) -> str:
    normalized = str(language or "auto").strip().lower()
    if normalized not in _LANGUAGES:
        raise ValueError("language must be auto, en, or vi")
    return normalized


def create_sheet_artifacts(
    config: Any,
    *,
    language: str,
    server_font_id: str | None = None,
    uploaded_font_bytes: bytes | None = None,
    uploaded_font_filename: str | None = None,
    custom_text: str | None = None,
    overrides: dict[str, Any] | None = None,
    output_basename: str | None = None,
) -> dict[str, str]:
    normalized_language = _normalize_language(language)
    has_server_font = bool(
        isinstance(server_font_id, str) and server_font_id.strip()
    )
    has_uploaded_font = uploaded_font_bytes is not None
    if has_server_font == has_uploaded_font:
        raise ValueError(
            "Provide exactly one font source: server_font or font_file"
        )

    options = build_sheet_options(config)
    normalized_text: str | None = None
    if custom_text is not None:
        stripped = custom_text.strip()
        if stripped:
            max_length = int(options["max_custom_text_length"])
            if len(stripped) > max_length:
                raise ValueError(
                    "custom_text exceeds configured max length"
                )
            normalized_text = stripped

    normalized_overrides = _normalize_overrides(overrides)

    font_source = "server" if has_server_font else "upload"
    font_path: Path
    font_name: str
    temp_font_path: Path | None = None

    if has_server_font:
        font_path = _resolve_server_font_path(
            config, str(server_font_id)
        )
        font_name = font_path.name
    else:
        if not uploaded_font_filename:
            raise ValueError("Uploaded font filename is missing")
        temp_font_path = _save_uploaded_font(
            config,
            file_bytes=uploaded_font_bytes or b"",
            original_filename=uploaded_font_filename,
        )
        font_path = temp_font_path
        font_name = Path(uploaded_font_filename).name

    try:
        generation = generate_handwriting_sheet(
            config,
            font_path=str(font_path),
            language=normalized_language,
            custom_text=normalized_text,
            overrides=normalized_overrides,
            output_basename=output_basename,
        )
    finally:
        if temp_font_path is not None:
            temp_font_path.unlink(missing_ok=True)

    pdf_path = Path(str(generation.get("pdf_path", ""))).resolve()
    preview_png_path = Path(
        str(generation.get("preview_png_path", ""))
    ).resolve()
    save_dir = _sheet_save_dir(config)

    if save_dir not in pdf_path.parents:
        raise RuntimeError(
            "Generated PDF path is outside save directory"
        )
    if save_dir not in preview_png_path.parents:
        raise RuntimeError(
            "Generated preview path is outside save directory"
        )
    if not pdf_path.exists() or not preview_png_path.exists():
        raise RuntimeError("Expected generated files were not found")

    return {
        "language": str(generation.get("language", "en")),
        "font_source": font_source,
        "font_name": font_name,
        "pdf_filename": pdf_path.name,
        "preview_image_filename": preview_png_path.name,
    }


def resolve_sheet_output_file(config: Any, file_name: str) -> Path:
    clean_name = Path(str(file_name)).name
    if clean_name != str(file_name):
        raise ValueError("Invalid file name")

    suffix = Path(clean_name).suffix.lower()
    if suffix not in {".pdf", ".png"}:
        raise ValueError("Unsupported file type")

    save_dir = _sheet_save_dir(config)
    candidate = (save_dir / clean_name).resolve()
    if save_dir not in candidate.parents:
        raise ValueError("Invalid file path")
    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError("Generated file not found")

    return candidate
