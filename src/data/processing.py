"""OCR processing helpers built around EasyOCR detection."""

from __future__ import annotations

import unicodedata
from typing import Any, Literal

import cv2
import numpy as np
import torch
from PIL import Image


def draw_horizontal_boxes(
    image: np.ndarray,
    boxes: list[Any],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw axis-aligned EasyOCR boxes in ``[x_min, x_max, y_min, y_max]`` format."""
    for box in boxes:
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            continue
        x_min, x_max, y_min, y_max = [int(v) for v in box[:4]]
        cv2.rectangle(
            image,
            (x_min, y_min),
            (x_max, y_max),
            color,
            thickness,
        )
    return image


def _to_grayscale_uint8(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale uint8 for crop scoring."""
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3:
        if image.shape[2] == 1:
            gray = image[:, :, 0]
        else:
            gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(
            "Expected 2D or 3D image array for split scoring"
        )

    if gray.dtype == np.uint8:
        return gray

    clipped = np.clip(gray, 0, 255)
    return clipped.astype(np.uint8)


def _normalize_horizontal_box(
    box: Any, image_h: int, image_w: int
) -> list[int] | None:
    """Normalize and clamp a horizontal box to image bounds."""
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None

    x_min, x_max, y_min, y_max = [
        int(round(float(v))) for v in box[:4]
    ]

    x_min = max(0, min(x_min, image_w - 1))
    x_max = max(0, min(x_max, image_w))
    y_min = max(0, min(y_min, image_h - 1))
    y_max = max(0, min(y_max, image_h))

    if x_max <= x_min or y_max <= y_min:
        return None

    return [x_min, x_max, y_min, y_max]


def _split_horizontal_box_uniform(
    box: list[int], split_count: int
) -> list[list[int]]:
    """Split [x_min, x_max, y_min, y_max] into uniform x-chunks."""
    x_min, x_max, y_min, y_max = box
    width = x_max - x_min
    if split_count <= 1 or width <= 1:
        return [box]

    max_parts = max(1, width)
    split_count = max(1, min(split_count, max_parts))
    if split_count == 1:
        return [box]

    base = width // split_count
    remainder = width % split_count

    chunks: list[list[int]] = []
    cursor = x_min
    for idx in range(split_count):
        step = base + (1 if idx < remainder else 0)
        next_x = cursor + max(step, 1)
        if idx == split_count - 1:
            next_x = x_max
        if next_x <= cursor:
            continue
        chunks.append([cursor, next_x, y_min, y_max])
        cursor = next_x

    if not chunks:
        return [box]

    chunks[-1][1] = x_max
    return chunks


def _crop_box(gray_image: np.ndarray, box: list[int]) -> np.ndarray:
    """Crop [x_min, x_max, y_min, y_max] from grayscale image."""
    x_min, x_max, y_min, y_max = box
    return gray_image[y_min:y_max, x_min:x_max]


def _foreground_mask_for_centering(crop: np.ndarray) -> np.ndarray:
    """Build a sparse foreground mask robust to dark/light ink polarity."""
    if crop.size == 0:
        return np.zeros_like(crop, dtype=bool)

    crop_u8 = crop.astype(np.uint8, copy=False)
    _, binary = cv2.threshold(
        crop_u8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    white_count = int(np.count_nonzero(binary))
    black_count = int(binary.size - white_count)
    if white_count <= black_count:
        mask = binary > 0
    else:
        mask = binary == 0

    if np.any(mask):
        return mask

    return crop_u8 > 0


def _compute_center_shift(mask: np.ndarray) -> tuple[int, int]:
    """Compute integer row/col shift required to center mask centroid."""
    rows, cols = np.nonzero(mask)
    row_center = float(rows.mean())
    col_center = float(cols.mean())

    target_row = (mask.shape[0] - 1) / 2.0
    target_col = (mask.shape[1] - 1) / 2.0

    shift_row = int(round(target_row - row_center))
    shift_col = int(round(target_col - col_center))
    return shift_row, shift_col


def _required_padding_for_shift(
    mask: np.ndarray, shift_row: int, shift_col: int
) -> tuple[int, int, int, int]:
    """Return top/bottom/left/right padding needed to avoid shift clipping."""
    rows, cols = np.nonzero(mask)
    min_row = int(rows.min()) + shift_row
    max_row = int(rows.max()) + shift_row
    min_col = int(cols.min()) + shift_col
    max_col = int(cols.max()) + shift_col

    pad_top = max(0, -min_row)
    pad_bottom = max(0, max_row - (mask.shape[0] - 1))
    pad_left = max(0, -min_col)
    pad_right = max(0, max_col - (mask.shape[1] - 1))

    return pad_top, pad_bottom, pad_left, pad_right


def _estimate_background_value(crop: np.ndarray) -> int:
    """Estimate background grayscale fill value used when shifting."""
    if crop.size == 0:
        return 0
    return int(np.clip(np.median(crop), 0, 255))


def _shift_image_without_wrap(
    crop: np.ndarray,
    *,
    shift_row: int,
    shift_col: int,
    fill_value: int,
) -> np.ndarray:
    """Shift image content with empty fill and no wrap-around."""
    shifted = np.full_like(crop, fill_value)

    src_r0 = max(0, -shift_row)
    src_r1 = min(crop.shape[0], crop.shape[0] - shift_row)
    src_c0 = max(0, -shift_col)
    src_c1 = min(crop.shape[1], crop.shape[1] - shift_col)

    if src_r1 <= src_r0 or src_c1 <= src_c0:
        return shifted

    dst_r0 = max(0, shift_row)
    dst_c0 = max(0, shift_col)
    dst_r1 = dst_r0 + (src_r1 - src_r0)
    dst_c1 = dst_c0 + (src_c1 - src_c0)

    shifted[dst_r0:dst_r1, dst_c0:dst_c1] = crop[
        src_r0:src_r1,
        src_c0:src_c1,
    ]
    return shifted


def _center_crop_for_scoring(crop: np.ndarray) -> np.ndarray:
    """Center crop foreground; extend canvas when centering would clip."""
    if crop.size == 0:
        return crop

    working = crop.astype(np.uint8, copy=False)
    fill_value = _estimate_background_value(working)

    # Two passes are enough for pad-then-center convergence in practice.
    for _ in range(2):
        mask = _foreground_mask_for_centering(working)
        if not np.any(mask):
            return working.copy()

        shift_row, shift_col = _compute_center_shift(mask)
        if shift_row == 0 and shift_col == 0:
            return working.copy()

        pad_top, pad_bottom, pad_left, pad_right = (
            _required_padding_for_shift(mask, shift_row, shift_col)
        )
        if (
            pad_top == 0
            and pad_bottom == 0
            and pad_left == 0
            and pad_right == 0
        ):
            return _shift_image_without_wrap(
                working,
                shift_row=shift_row,
                shift_col=shift_col,
                fill_value=fill_value,
            )

        working = np.pad(
            working,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=fill_value,
        )

    # Safety fallback for edge cases where shifts still degenerate.
    return center_image(working)


def _crop_to_unit_vector(
    crop: np.ndarray, vector_size: int
) -> np.ndarray:
    """Resize crop and return a unit vector for cosine scoring."""
    if (
        crop.size == 0
        or crop.shape[0] == 0
        or crop.shape[1] == 0
        or vector_size <= 0
    ):
        return np.zeros(1, dtype=np.float32)

    centered_crop = _center_crop_for_scoring(crop)

    resized = cv2.resize(
        centered_crop,
        (vector_size, vector_size),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)

    if resized.max() > 1.0:
        resized = resized / 255.0

    vec = resized.reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm > 1e-8:
        vec = vec / norm

    return vec.astype(np.float32)


def _cosine_similarity_percent(
    reference_vec: np.ndarray, other_vec: np.ndarray
) -> float:
    """Convert cosine similarity to [0, 100] score."""
    if reference_vec.size == 1 and other_vec.size == 1:
        return 100.0

    if reference_vec.size != other_vec.size:
        min_size = min(reference_vec.size, other_vec.size)
        reference_vec = reference_vec[:min_size]
        other_vec = other_vec[:min_size]

    ref_norm = float(np.linalg.norm(reference_vec))
    other_norm = float(np.linalg.norm(other_vec))

    if ref_norm <= 1e-8 and other_norm <= 1e-8:
        return 100.0
    if ref_norm <= 1e-8 or other_norm <= 1e-8:
        return 0.0

    cosine = float(
        np.dot(reference_vec, other_vec) / (ref_norm * other_norm)
    )
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.clip((cosine + 1.0) * 50.0, 0.0, 100.0))


def _estimate_split_count(
    row_width: int, median_width: float, width_ratio: float
) -> int:
    """Estimate split count for a row box using relative width heuristic."""
    if row_width <= 1:
        return 1
    if median_width <= 0 or width_ratio <= 0:
        return 1

    threshold = median_width * width_ratio
    if row_width <= threshold:
        return 1

    estimated = int(round(row_width / max(median_width, 1.0)))
    return max(2, min(estimated, max(1, row_width)))


def _box_area(box: list[int]) -> float:
    """Return area for [x_min, x_max, y_min, y_max] box."""
    width = max(0, int(box[1]) - int(box[0]))
    height = max(0, int(box[3]) - int(box[2]))
    return float(width * height)


def _box_overlap_ratio(
    row_box: list[int], candidate_box: list[int]
) -> float:
    """Compute intersection ratio relative to candidate box area."""
    inter_x0 = max(row_box[0], candidate_box[0])
    inter_x1 = min(row_box[1], candidate_box[1])
    inter_y0 = max(row_box[2], candidate_box[2])
    inter_y1 = min(row_box[3], candidate_box[3])

    inter_w = max(0, inter_x1 - inter_x0)
    inter_h = max(0, inter_y1 - inter_y0)
    inter_area = float(inter_w * inter_h)

    candidate_area = _box_area(candidate_box)
    if candidate_area <= 1e-8:
        return 0.0
    return inter_area / candidate_area


def _merged_span_width(spans: list[tuple[int, int]]) -> int:
    """Return merged width for 1D [start, end) spans."""
    filtered = [
        (int(start), int(end))
        for start, end in spans
        if int(end) > int(start)
    ]
    if not filtered:
        return 0

    filtered.sort(key=lambda span: span[0])
    merged_width = 0
    active_start, active_end = filtered[0]

    for start, end in filtered[1:]:
        if start <= active_end:
            active_end = max(active_end, end)
            continue

        merged_width += active_end - active_start
        active_start, active_end = start, end

    merged_width += active_end - active_start
    return merged_width


def _count_label_characters(label: str) -> int:
    """Count letter/number characters from OCR labels."""
    normalized = unicodedata.normalize(
        "NFC", str(label or "")
    ).strip()
    if not normalized:
        return 0

    return sum(
        1
        for char in normalized
        if not char.isspace()
        and unicodedata.category(char)[0] in {"L", "N"}
    )


def _estimate_split_count_from_ocr_hints(
    row_box: list[int],
    *,
    image_h: int,
    image_w: int,
    extracted_labeled_boxes: list[dict[str, Any]] | None,
    min_confidence: float = 0.35,
    min_overlap_ratio: float = 0.45,
) -> int | None:
    """Estimate split count from OCR labels that overlap a row box."""
    if not extracted_labeled_boxes:
        return None

    row_width = row_box[1] - row_box[0]
    if row_width <= 1:
        return None

    char_widths: list[float] = []
    overlap_spans: list[tuple[int, int]] = []
    max_label_chars = 0

    for entry in extracted_labeled_boxes:
        if not isinstance(entry, dict):
            continue

        normalized_box = _normalize_horizontal_box(
            entry.get("box"),
            image_h=image_h,
            image_w=image_w,
        )
        if normalized_box is None:
            continue

        try:
            confidence = float(entry.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < min_confidence:
            continue

        overlap_ratio = _box_overlap_ratio(row_box, normalized_box)
        if overlap_ratio < min_overlap_ratio:
            continue

        label_chars = _count_label_characters(
            str(entry.get("label", ""))
        )
        if label_chars <= 0:
            continue

        box_width = normalized_box[1] - normalized_box[0]
        if box_width <= 0:
            continue

        char_widths.append(box_width / float(label_chars))
        max_label_chars = max(max_label_chars, label_chars)

        overlap_x0 = max(row_box[0], normalized_box[0])
        overlap_x1 = min(row_box[1], normalized_box[1])
        if overlap_x1 > overlap_x0:
            overlap_spans.append((overlap_x0, overlap_x1))

    if not char_widths:
        return None

    coverage_ratio = _merged_span_width(overlap_spans) / float(
        max(1, row_width)
    )
    if len(char_widths) == 1 and coverage_ratio < 0.6:
        return None
    if len(char_widths) >= 2 and coverage_ratio < 0.35:
        return None

    median_char_width = float(np.median(char_widths))
    if median_char_width < 1.0:
        return None

    width_based_guess = int(round(row_width / median_char_width))
    ocr_guess = max(width_based_guess, max_label_chars)

    if ocr_guess <= 1:
        return None

    return int(np.clip(ocr_guess, 1, row_width))


def _estimate_row_split_count(
    row_box: list[int],
    *,
    median_width: float,
    width_ratio: float,
    image_h: int,
    image_w: int,
    extracted_labeled_boxes: list[dict[str, Any]] | None,
) -> int:
    """Estimate split count from geometry, optionally boosted by OCR."""
    row_width = row_box[1] - row_box[0]
    geometry_guess = _estimate_split_count(
        row_width,
        median_width,
        width_ratio,
    )

    ocr_guess = _estimate_split_count_from_ocr_hints(
        row_box,
        image_h=image_h,
        image_w=image_w,
        extracted_labeled_boxes=extracted_labeled_boxes,
    )

    if ocr_guess is None or ocr_guess <= geometry_guess:
        return geometry_guess

    # Keep OCR boosts conservative once geometry already found splits.
    if geometry_guess > 1 and ocr_guess > geometry_guess + 2:
        return geometry_guess

    # OCR hints only raise split counts to reduce under-splitting.
    return ocr_guess


def _score_row_splits(
    gray_image: np.ndarray,
    split_boxes: list[list[int]],
    *,
    vector_size: int,
) -> list[dict[str, Any]]:
    """Score non-reference splits against split index 0."""
    if not split_boxes:
        return []

    reference_box = split_boxes[0]
    reference_vec = _crop_to_unit_vector(
        _crop_box(gray_image, reference_box),
        vector_size=vector_size,
    )

    items: list[dict[str, Any]] = []
    for split_index, split_box in enumerate(split_boxes):
        similarity_to_first = 100.0
        if split_index > 0:
            split_vec = _crop_to_unit_vector(
                _crop_box(gray_image, split_box),
                vector_size=vector_size,
            )
            similarity_to_first = _cosine_similarity_percent(
                reference_vec,
                split_vec,
            )

        items.append(
            {
                "split_index": split_index,
                "box": split_box,
                "similarity_to_first": similarity_to_first,
                "is_reference": split_index == 0,
            }
        )

    return items


def _split_detected_rows(
    image: np.ndarray,
    detect_output: dict[str, list],
    *,
    width_ratio: float,
    extracted_labeled_boxes: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Normalize rows and build split plans before similarity scoring."""
    gray = _to_grayscale_uint8(image)
    image_h, image_w = gray.shape[:2]

    raw_horizontal = detect_output.get("horizontal_list", [])
    free_list = detect_output.get("free_list", [])

    normalized: list[tuple[int, list[int]]] = []
    for index, box in enumerate(raw_horizontal):
        normalized_box = _normalize_horizontal_box(
            box, image_h=image_h, image_w=image_w
        )
        if normalized_box is None:
            continue
        normalized.append((index, normalized_box))

    widths = [box[1] - box[0] for _, box in normalized]
    median_width = float(np.median(widths)) if widths else 0.0

    horizontal_groups: list[dict[str, Any]] = []
    flattened_horizontal: list[list[int]] = []

    for source_index, row_box in normalized:
        split_count = _estimate_row_split_count(
            row_box,
            median_width=median_width,
            width_ratio=width_ratio,
            image_h=image_h,
            image_w=image_w,
            extracted_labeled_boxes=extracted_labeled_boxes,
        )

        split_boxes = _split_horizontal_box_uniform(
            row_box,
            split_count=split_count,
        )
        flattened_horizontal.extend(split_boxes)

        horizontal_groups.append(
            {
                "source_index": source_index,
                "source_box": row_box,
                "reference_box": split_boxes[0],
                "was_split": len(split_boxes) > 1,
                "split_boxes": split_boxes,
            }
        )

    return {
        "median_width": median_width,
        "horizontal_groups": horizontal_groups,
        "horizontal_list": flattened_horizontal,
        "free_list": free_list,
    }


def split_big_boxes_and_score_similarity(
    image: np.ndarray,
    detect_output: dict[str, list],
    width_ratio: float = 1.8,
    vector_size: int = 28,
    extracted_labeled_boxes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Detect row boxes, split rows, then score against row reference split.

    Flow:
    1. Normalize valid row boxes from EasyOCR horizontal detections.
    2. Split an oversized row into uniform left-to-right chunks.
    3. Treat split index 0 as the reference character for the row.
    4. Score split indices 1..N-1 against split index 0.

    Oversized decision uses width > width_ratio * median_width of valid
    horizontal boxes in the same image. When OCR labeled boxes are provided,
    the estimator can raise split_count to reduce under-splitting.
    Similarity is cosine-style percentage in [0, 100].
    """
    gray = _to_grayscale_uint8(image)
    split_plan = _split_detected_rows(
        image,
        detect_output,
        width_ratio=width_ratio,
        extracted_labeled_boxes=extracted_labeled_boxes,
    )

    horizontal_groups: list[dict[str, Any]] = []
    for group in split_plan["horizontal_groups"]:
        split_boxes = [list(box) for box in group["split_boxes"]]
        items = _score_row_splits(
            gray,
            split_boxes,
            vector_size=vector_size,
        )
        horizontal_groups.append(
            {
                "source_index": group["source_index"],
                "source_box": list(group["source_box"]),
                "reference_box": list(group["reference_box"]),
                "was_split": bool(group["was_split"]),
                "items": items,
            }
        )

    return {
        "median_width": float(split_plan["median_width"]),
        "horizontal_groups": horizontal_groups,
        "horizontal_list": list(split_plan["horizontal_list"]),
        "free_list": list(split_plan["free_list"]),
    }


def draw_free_boxes(
    image: np.ndarray,
    boxes: list[Any],
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw free-form EasyOCR polygon boxes."""
    for box in boxes:
        if not isinstance(box, (list, tuple)) or len(box) == 0:
            continue
        points = np.array(box, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            image,
            [points],
            isClosed=True,
            color=color,
            thickness=thickness,
        )
    return image


def draw_boxes(
    image: np.ndarray,
    horizontal_boxes: list[Any] | None = None,
    free_boxes: list[Any] | None = None,
    copy: bool = True,
) -> np.ndarray:
    """Draw horizontal and free-form boxes on the image."""
    output = image.copy() if copy else image

    if horizontal_boxes is not None:
        draw_horizontal_boxes(output, horizontal_boxes)
    if free_boxes is not None:
        draw_free_boxes(output, free_boxes)

    return output


def _to_numpy(image: Image.Image | np.ndarray) -> np.ndarray:
    """Convert a PIL image or NumPy array to a NumPy array."""
    if isinstance(image, Image.Image):
        return np.asarray(image)
    if isinstance(image, np.ndarray):
        return image
    raise TypeError(f"Unsupported image type: {type(image)!r}")


def to_grayscale(image: Image.Image | np.ndarray) -> np.ndarray:
    """Convert input image to a 2D grayscale NumPy array."""
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("L"))

    arr = _to_numpy(image)

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            return arr[..., 0]
        if arr.shape[-1] >= 3:
            rgb = arr[..., :3].astype(np.float32)
            gray = (
                0.299 * rgb[..., 0]
                + 0.587 * rgb[..., 1]
                + 0.114 * rgb[..., 2]
            )
            return (
                gray.astype(arr.dtype)
                if np.issubdtype(arr.dtype, np.integer)
                else gray
            )

    raise ValueError(
        f"Expected 2D grayscale or 3D channel-last image, got shape {arr.shape}"
    )


def resize(
    image: Image.Image | np.ndarray, target_size: int = 28
) -> np.ndarray:
    """Resize an image to ``(target_size, target_size)`` and return as NumPy array."""
    gray = to_grayscale(image)
    pil_img = Image.fromarray(gray.astype(np.uint8))
    resized = pil_img.resize(
        (target_size, target_size), Image.Resampling.BILINEAR
    )
    return np.asarray(resized)


def extract_character_bbox(
    image: Image.Image | np.ndarray, target_size: int = 28
) -> np.ndarray:
    """Extract all characters by bounding boxes and fit each into a square canvas.

    Foreground contours are detected, sorted left-to-right, cropped, then resized with
    aspect-ratio preservation and centered into ``target_size x target_size`` outputs.

    Returns:
            Array of shape ``(N, target_size, target_size)`` where ``N`` is the number of
            detected characters. If none are detected, returns an empty array with shape
            ``(0, target_size, target_size)``.
    """
    gray = to_grayscale(image)
    gray_u8 = gray.astype(np.uint8)

    _, binary = cv2.threshold(
        gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Make character white on black for contour detection.
    if np.count_nonzero(binary) > binary.size / 2:
        binary = cv2.bitwise_not(binary)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return np.empty((0, target_size, target_size), dtype=np.uint8)

    contours_sorted = sorted(
        contours, key=lambda contour: cv2.boundingRect(contour)[0]
    )
    extracted_chars: list[np.ndarray] = []

    for contour in contours_sorted:
        x, y, width, height = cv2.boundingRect(contour)
        if width <= 0 or height <= 0:
            continue

        char_crop = gray_u8[y : y + height, x : x + width]

        scale = min(
            target_size / max(width, 1), target_size / max(height, 1)
        )
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        resized_char = cv2.resize(
            char_crop, (new_w, new_h), interpolation=cv2.INTER_AREA
        )

        canvas = np.zeros((target_size, target_size), dtype=np.uint8)
        x_offset = (target_size - new_w) // 2
        y_offset = (target_size - new_h) // 2
        canvas[
            y_offset : y_offset + new_h, x_offset : x_offset + new_w
        ] = resized_char
        extracted_chars.append(canvas)

    if not extracted_chars:
        return np.empty((0, target_size, target_size), dtype=np.uint8)

    return np.stack(extracted_chars, axis=0)


def normalize(image: Image.Image | np.ndarray) -> np.ndarray:
    """Normalize image values to the [0, 1] range as float32."""
    arr = _to_numpy(image).astype(np.float32)

    if arr.size == 0:
        return arr

    if arr.max() > 1.0:
        arr = arr / 255.0

    return np.clip(arr, 0.0, 1.0)


def center_image(image: Image.Image | np.ndarray) -> np.ndarray:
    """Center non-background content within the image canvas.

    The image shape is preserved. For blank images, input content is returned unchanged.
    """
    arr = _to_numpy(image)
    if arr.ndim != 2:
        arr = to_grayscale(arr)

    working = arr.astype(np.float32)
    mask = working > 0
    if not np.any(mask):
        return arr.copy()

    rows, cols = np.nonzero(mask)
    row_center = float(rows.mean())
    col_center = float(cols.mean())

    target_row = (arr.shape[0] - 1) / 2.0
    target_col = (arr.shape[1] - 1) / 2.0

    shift_row = int(round(target_row - row_center))
    shift_col = int(round(target_col - col_center))

    centered = np.zeros_like(arr)

    src_r0 = max(0, -shift_row)
    src_r1 = min(arr.shape[0], arr.shape[0] - shift_row)
    src_c0 = max(0, -shift_col)
    src_c1 = min(arr.shape[1], arr.shape[1] - shift_col)

    dst_r0 = max(0, shift_row)
    dst_r1 = dst_r0 + (src_r1 - src_r0)
    dst_c0 = max(0, shift_col)
    dst_c1 = dst_c0 + (src_c1 - src_c0)

    if src_r1 > src_r0 and src_c1 > src_c0:
        centered[dst_r0:dst_r1, dst_c0:dst_c1] = arr[
            src_r0:src_r1, src_c0:src_c1
        ]

    return centered


def preprocess(
    image: Image.Image | np.ndarray,
    target_size: int = 28,
    return_type: Literal["torch", "numpy"] = "torch",
) -> torch.Tensor | np.ndarray:
    """Run full preprocessing pipeline.

    Steps:
    1. Convert to grayscale
    2. Extract character by OpenCV bounding box
    3. Fit and center character in ``target_size x target_size``
    4. Normalize to [0, 1]

    Args:
            image: PIL image or NumPy array.
            target_size: Output image side length.
            return_type: ``"torch"`` returns tensor ``(1, H, W)``,
                    ``"numpy"`` returns array ``(H, W)``.
    """
    extracted_chars = extract_character_bbox(
        image, target_size=target_size
    )
    if extracted_chars.shape[0] == 0:
        selected = np.zeros(
            (target_size, target_size), dtype=np.uint8
        )
    else:
        selected = max(
            extracted_chars,
            key=lambda char: int(np.count_nonzero(char)),
        )

    processed = normalize(selected)

    if return_type == "numpy":
        return processed.astype(np.float32)
    if return_type == "torch":
        return torch.from_numpy(
            processed.astype(np.float32)
        ).unsqueeze(0)

    raise ValueError("return_type must be either 'torch' or 'numpy'")


def peprocess(
    imrage: Image.Image | np.ndarray,
    target_size: int = 28,
    return_type: Literal["torch", "numpy"] = "torch",
) -> torch.Tensor | np.ndarray:
    """Backward-compatible alias for preprocess (deprecated typo)."""
    return preprocess(
        image=imrage, target_size=target_size, return_type=return_type
    )


__all__ = [
    "to_grayscale",
    "resize",
    "extract_character_bbox",
    "normalize",
    "center_image",
    "preprocess",
    "peprocess",
    "split_big_boxes_and_score_similarity",
    "draw_boxes",
    "draw_free_boxes",
    "draw_horizontal_boxes",
]
