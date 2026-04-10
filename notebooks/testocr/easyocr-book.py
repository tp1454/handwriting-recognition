import sys
from pathlib import Path

import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
from api.services import _build_easyocr_detect_kwargs
from src.utils.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

INPUT_IMAGE_PATH = Path(
    "data/raw/vi_sheet_20260402_234837_300097_page_1.png"
)

OUTPUT_DIR = Path("data/processed/easyocr")

OUTPUT_IMAGE_PATH = (
    OUTPUT_DIR
    / f"{INPUT_IMAGE_PATH.stem}_boxed{INPUT_IMAGE_PATH.suffix}"
)


def draw_boxes(image, boxes):
    pass


def _normalize_detect_output(result):
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

    return horizontal_list, free_list


def draw_horizontal_boxes(image, boxes):
    for box in boxes:
        x_min, x_max, y_min, y_max = [int(v) for v in box[:4]]
        cv2.rectangle(
            image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2
        )
    return image


def draw_free_boxes(image, boxes):
    for box in boxes:
        points = np.array(box, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            image,
            [points],
            isClosed=True,
            color=(255, 0, 0),
            thickness=2,
        )
    return image


# min_size=3,  # smaller chars
#         text_threshold=0.4,  # slightly lower
#         low_text=0.05,  # detect faint chars
#         link_threshold=0.1,  # VERY IMPORTANT: reduce linking
#         canvas_size=2560,
#         mag_ratio=3.0,  # zoom in more
#         slope_ths=0.1,  # stricter alignment
#         ycenter_ths=0.3,  # reduce row merging
#         height_ths=0.3,  # reduce grouping
#         width_ths=0.3,  # reduce grouping
#         add_margin=0.05,
def main():
    cfg = load_config()
    detect_kwargs = _build_easyocr_detect_kwargs(cfg)

    reader = easyocr.Reader(["vi"], gpu=False)

    result = reader.detect(
        str(INPUT_IMAGE_PATH),
        **detect_kwargs,
    )

    image = cv2.imread(str(INPUT_IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {INPUT_IMAGE_PATH}"
        )
    print(
        f"Loaded image: {INPUT_IMAGE_PATH} with shape {image.shape}"
    )
    horizontal_list, free_list = _normalize_detect_output(result)

    image_with_boxes = draw_horizontal_boxes(
        image.copy(), horizontal_list
    )
    image_with_boxes = draw_free_boxes(image_with_boxes, free_list)

    print(
        f"Detected {len(horizontal_list)} horizontal boxes and {len(free_list)} free boxes."
    )
    print(f"EasyOCR detect kwargs from YAML: {detect_kwargs}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_IMAGE_PATH), image_with_boxes)

    print(f"Saved boxed image to: {OUTPUT_IMAGE_PATH}")

    plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
