import argparse
import os
import time

import cv2
import numpy as np
from imutils.object_detection import non_max_suppression


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--image",
        required=True,
        type=str,
        help="path to input image",
    )
    parser.add_argument(
        "--east",
        required=True,
        type=str,
        help="path to input EAST text detector (.pb)",
    )
    parser.add_argument(
        "-c",
        "--min-confidence",
        type=float,
        default=0.5,
        help="minimum probability required to inspect a region",
    )
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        default=320,
        help="resized image width (must be multiple of 32)",
    )
    parser.add_argument(
        "-e",
        "--height",
        type=int,
        default=320,
        help="resized image height (must be multiple of 32)",
    )
    args = parser.parse_args()

    if args.width % 32 != 0 or args.height % 32 != 0:
        parser.error("--width and --height must be multiples of 32")

    if not os.path.exists(args.image):
        parser.error(f"image not found: {args.image}")

    if not os.path.exists(args.east):
        parser.error(f"EAST model not found: {args.east}")

    return args


def main():
    args = parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"failed to read image: {args.image}")

    orig = image.copy()
    orig_h, orig_w = image.shape[:2]
    new_w, new_h = (args.width, args.height)
    r_w = orig_w / float(new_w)
    r_h = orig_h / float(new_h)

    image = cv2.resize(image, (new_w, new_h))
    h, w = image.shape[:2]

    layer_names = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3",
    ]

    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(args.east)

    blob = cv2.dnn.blobFromImage(
        image,
        1.0,
        (w, h),
        (123.68, 116.78, 103.94),
        swapRB=True,
        crop=False,
    )

    start = time.time()
    net.setInput(blob)
    scores, geometry = net.forward(layer_names)
    end = time.time()
    print(f"[INFO] text detection took {end - start:.6f} seconds")

    num_rows, num_cols = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, num_rows):
        scores_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]
        x_data1 = geometry[0, 1, y]
        x_data2 = geometry[0, 2, y]
        x_data3 = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]

        for x in range(0, num_cols):
            if scores_data[x] < args.min_confidence:
                continue

            offset_x, offset_y = (x * 4.0, y * 4.0)
            angle = angles_data[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = x_data0[x] + x_data2[x]
            w = x_data1[x] + x_data3[x]

            end_x = int(
                offset_x + (cos * x_data1[x]) + (sin * x_data2[x])
            )
            end_y = int(
                offset_y - (sin * x_data1[x]) + (cos * x_data2[x])
            )
            start_x = int(end_x - w)
            start_y = int(end_y - h)

            rects.append((start_x, start_y, end_x, end_y))
            confidences.append(scores_data[x])

    boxes = []
    if rects:
        boxes = non_max_suppression(
            np.array(rects), probs=confidences
        )

    for start_x, start_y, end_x, end_y in boxes:
        start_x = int(start_x * r_w)
        start_y = int(start_y * r_h)
        end_x = int(end_x * r_w)
        end_y = int(end_y * r_h)
        cv2.rectangle(
            orig, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2
        )

    cv2.imshow("Text Detection", orig)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
