"""OCR processing helpers built around EasyOCR detection."""

from __future__ import annotations

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

    resized = cv2.resize(
        crop,
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


def split_big_boxes_and_score_similarity(
    image: np.ndarray,
    detect_output: dict[str, list],
    width_ratio: float = 1.8,
    vector_size: int = 28,
) -> dict[str, Any]:
    """Split oversized horizontal boxes and score each split vs first split.

    Oversized decision uses width > width_ratio * median_width of valid horizontal
    boxes in the same image. Similarity is cosine-style percentage in [0, 100].
    """
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

    for source_index, box in normalized:
        width = box[1] - box[0]
        split_count = 1
        if median_width > 0 and width_ratio > 0:
            if width > median_width * width_ratio:
                estimated = int(round(width / max(median_width, 1.0)))
                split_count = max(2, estimated)

        split_boxes = _split_horizontal_box_uniform(
            box, split_count=split_count
        )
        reference_vec = _crop_to_unit_vector(
            _crop_box(gray, split_boxes[0]),
            vector_size=vector_size,
        )

        group_items: list[dict[str, Any]] = []
        for split_index, split_box in enumerate(split_boxes):
            split_vec = _crop_to_unit_vector(
                _crop_box(gray, split_box),
                vector_size=vector_size,
            )
            score = 100.0
            if split_index > 0:
                score = _cosine_similarity_percent(
                    reference_vec, split_vec
                )

            group_items.append(
                {
                    "split_index": split_index,
                    "box": split_box,
                    "similarity_to_first": score,
                }
            )
            flattened_horizontal.append(split_box)

        horizontal_groups.append(
            {
                "source_index": source_index,
                "source_box": box,
                "was_split": len(split_boxes) > 1,
                "items": group_items,
            }
        )

    return {
        "median_width": median_width,
        "horizontal_groups": horizontal_groups,
        "horizontal_list": flattened_horizontal,
        "free_list": free_list,
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
