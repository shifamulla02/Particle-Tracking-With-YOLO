"""
Draws bounding boxes for all annotations belonging to image_id 514139.
(Duplicates from your pasted list are removed, and the one annotation
that actually belongs to image_id 178181 has been excluded.)

pip install opencv-python
python draw_boxes_514139.py
"""

import cv2

IMAGE_PATH = "000000514139.jpg"      # <-- change this to your actual image path
OUTPUT_PATH = "output_514139.jpg"

# Optional: map category_id -> readable name. Leave empty {} to just show the id number.
CATEGORY_NAMES = {}

ANNOTATIONS = [
    {"bbox": [447.34, 173.58, 9.54, 21.0], "category_id": 3, "image_id": 514139, "id": 1785294},
    {"bbox": [159.72, 161.88, 24.95, 17.8], "category_id": 8, "image_id": 514139, "id": 1368710},
    {"bbox": [187.87, 165.65, 44.66, 15.67], "category_id": 3, "image_id": 514139, "id": 1343364},
    {"bbox": [623.52, 168.01, 16.34, 40.14], "category_id": 3, "image_id": 514139, "id": 1337577},
    {"bbox": [531.46, 161.56, 94.91, 42.88], "category_id": 3, "image_id": 514139, "id": 357570},
    {"bbox": [0.0, 167.99, 193.55, 65.23], "category_id": 3, "image_id": 514139, "id": 360571},
    {"bbox": [79.49, 126.57, 509.15, 286.8], "category_id": 8, "image_id": 514139, "id": 395388},
]

# --- code below just draws them, no need to touch ---

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Could not open image: {IMAGE_PATH}")

for ann in ANNOTATIONS:
    x, y, w, h = ann["bbox"]
    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
    cat_id = ann["category_id"]
    label = CATEGORY_NAMES.get(cat_id, str(cat_id))

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), (0, 255, 0), -1)
    cv2.putText(img, label, (x1 + 2, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

cv2.imwrite(OUTPUT_PATH, img)
print(f"Saved: {OUTPUT_PATH}")