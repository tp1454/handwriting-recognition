from pathlib import Path

import cv2
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = REPO_ROOT / "data" / "raw" / "img.jpg"
OUTPUT_DIR = REPO_ROOT / "data" / "user_samples" / "extracted_digits"
MIN_CONTOUR_AREA = 20

if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Input image not found: {IMAGE_PATH}")

frame = cv2.imread(str(IMAGE_PATH))
if frame is None:
    raise ValueError(f"OpenCV could not read image: {IMAGE_PATH}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# make plot for debugging


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.subplot(1, 2, 2)
plt.title("Thresholded Image")
plt.imshow(thresh, cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()

contours, _ = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
contours = sorted(
    contours, key=lambda contour: cv2.boundingRect(contour)[0]
)

saved_count = 0
for contour in contours:
    area = cv2.contourArea(contour)
    if area < MIN_CONTOUR_AREA:
        continue

    x, y, w, h = cv2.boundingRect(contour)
    roi = thresh[y : y + h, x : x + w]
    if roi.size == 0:
        continue

    roi_resized = cv2.resize(
        roi, (28, 28), interpolation=cv2.INTER_AREA
    )

    output_path = OUTPUT_DIR / f"digit_{saved_count:03d}.png"
    cv2.imwrite(str(output_path), roi_resized)
    saved_count += 1

if saved_count == 0:
    print(
        "No digits detected. Try adjusting thresholding or contour area."
    )
else:
    print(f"Saved {saved_count} digit crops to: {OUTPUT_DIR}")
