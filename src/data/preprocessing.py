"""Image preprocessing utilities for handwriting recognition.

This module provides a small preprocessing pipeline that accepts either
PIL images or NumPy arrays and produces normalized 28x28 outputs.
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
import torch
from PIL import Image


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
