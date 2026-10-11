from pathlib import Path
import shutil
import csv

from PIL import Image, ImageDraw
from ultralytics import YOLO


# ============================================================
# 1. PATHS
# ============================================================

DATASET = Path(
    r"C:\Users\shiif\Downloads\yoloduke\datasets\coco_instance_balanced5_undersampled"
)

MODEL_PATH = Path(
    r"C:\Users\shiif\Downloads\yoloduke\runs\detect\train-7\weights\best.pt"
)

IMAGE_DIR = DATASET / "images" / "val"
LABEL_DIR = DATASET / "labels" / "val"


# ============================================================
# 2. OUTPUT FOLDER
# ============================================================

OUTPUT_DIR = DATASET / "localization_failures"

OUTPUT_IMAGE_DIR = OUTPUT_DIR / "images"
OUTPUT_LABEL_DIR = OUTPUT_DIR / "labels"

OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. SETTINGS
# ============================================================

CLASS_NAMES = {
    0: "person",
    1: "car",
    2: "chair",
    3: "book",
    4: "bottle"
}

CAR_CLASS = 1

CONF_THRESHOLD = 0.25

IOU_THRESHOLD = 0.50


# ============================================================
# 4. LOAD MODEL
# ============================================================

print("=" * 70)
print("Loading YOLO model...")
print("=" * 70)

model = YOLO(str(MODEL_PATH))

print("Model loaded:")
print(MODEL_PATH)
print()


# ============================================================
# 5. IoU FUNCTION
# ============================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(
        0,
        x2 - x1
    )

    intersection_height = max(
        0,
        y2 - y1
    )

    intersection_area = (
        intersection_width *
        intersection_height
    )

    area1 = (
        (box1[2] - box1[0]) *
        (box1[3] - box1[1])
    )

    area2 = (
        (box2[2] - box2[0]) *
        (box2[3] - box2[1])
    )

    union_area = (
        area1 +
        area2 -
        intersection_area
    )

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


# ============================================================
# 6. YOLO FORMAT -> PIXEL BOX
# ============================================================

def yolo_to_xyxy(
    x_center,
    y_center,
    width,
    height,
    image_width,
    image_height
):

    x_center = x_center * image_width
    y_center = y_center * image_height

    width = width * image_width
    height = height * image_height

    x1 = x_center - width / 2
    y1 = y_center - height / 2

    x2 = x_center + width / 2
    y2 = y_center + height / 2

    return [
        x1,
        y1,
        x2,
        y2
    ]


# ============================================================
# 7. READ ALL GROUND-TRUTH BOXES
# ============================================================

def get_all_ground_truth_boxes(
    label_file,
    image_width,
    image_height
):

    boxes = []

    if not label_file.exists():
        return boxes

    with open(label_file, "r") as f:

        for line in f:

            values = line.strip().split()

            if len(values) != 5:
                continue

            class_id = int(values[0])

            x_center = float(values[1])
            y_center = float(values[2])
            width = float(values[3])
            height = float(values[4])

            box = yolo_to_xyxy(
                x_center,
                y_center,
                width,
                height,
                image_width,
                image_height
            )

            boxes.append({
                "class_id": class_id,
                "box": box
            })

    return boxes


# ============================================================
# 8. READ ONLY GROUND-TRUTH CAR BOXES
# ============================================================

def get_ground_truth_cars(
    label_file,
    image_width,
    image_height
):

    all_boxes = get_all_ground_truth_boxes(
        label_file,
        image_width,
        image_height
    )

    car_boxes = []

    for item in all_boxes:

        if item["class_id"] == CAR_CLASS:

            car_boxes.append(
                item["box"]
            )

    return car_boxes


# ============================================================
# 9. DRAW GROUND-TRUTH BOXES
# ============================================================

def draw_ground_truth_boxes(
    image_path,
    label_path,
    output_path,
    image_width,
    image_height
):

    # Read original image
    image = Image.open(
        image_path
    ).convert("RGB")

    draw = ImageDraw.Draw(image)

    # Read ALL GT boxes
    gt_boxes = get_all_ground_truth_boxes(
        label_path,
        image_width,
        image_height
    )

    for item in gt_boxes:

        class_id = item["class_id"]
        box = item["box"]

        x1, y1, x2, y2 = box

        class_name = CLASS_NAMES.get(
            class_id,
            str(class_id)
        )

        # Red = CAR
        # Blue = other GT objects
        if class_id == CAR_CLASS:
            outline_color = "red"
        else:
            outline_color = "blue"

        # Draw box
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=outline_color,
            width=3
        )

        # Label
        label_text = f"GT: {class_name}"

        try:

            bbox = draw.textbbox(
                (x1, y1),
                label_text
            )

            text_width = (
                bbox[2] - bbox[0]
            )

            text_height = (
                bbox[3] - bbox[1]
            )

        except:

            text_width = 80
            text_height = 15

        draw.rectangle(
            [
                x1,
                max(
                    0,
                    y1 - text_height - 4
                ),
                x1 + text_width + 6,
                y1
            ],
            fill=outline_color
        )

        draw.text(
            (
                x1 + 3,
                max(
                    0,
                    y1 - text_height - 2
                )
            ),
            label_text,
            fill="white"
        )

    # Save annotated COPY
    image.save(output_path)


# ============================================================
# 10. GET VALIDATION IMAGES
# ============================================================

image_files = []

for extension in [
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.JPG",
    "*.JPEG",
    "*.PNG"
]:

    image_files.extend(
        IMAGE_DIR.glob(extension)
    )

image_files = sorted(set(image_files))


# ============================================================
# 11. COUNTERS
# ============================================================

total_images_with_cars = 0

total_gt_cars = 0

total_localization_failures = 0

images_with_localization_failures = 0

report_rows = []


# ============================================================
# 12. PROCESS VALIDATION IMAGES
# ============================================================

print("=" * 70)
print("SEARCHING FOR LOCALIZATION FAILURES")
print("=" * 70)


for index, image_path in enumerate(
    image_files,
    start=1
):

    print(
        f"[{index}/{len(image_files)}] "
        f"{image_path.name}"
    )

    # --------------------------------------------------------
    # Corresponding label
    # --------------------------------------------------------

    label_path = (
        LABEL_DIR /
        f"{image_path.stem}.txt"
    )

    # --------------------------------------------------------
    # Image dimensions
    # --------------------------------------------------------

    try:

        with Image.open(image_path) as img:

            image_width, image_height = img.size

    except Exception as e:

        print(
            f"  Could not open image: {e}"
        )

        continue

    # --------------------------------------------------------
    # Ground-truth cars
    # --------------------------------------------------------

    gt_car_boxes = get_ground_truth_cars(
        label_path,
        image_width,
        image_height
    )

    if len(gt_car_boxes) == 0:
        continue

    total_images_with_cars += 1

    total_gt_cars += len(gt_car_boxes)

    # --------------------------------------------------------
    # YOLO prediction
    # --------------------------------------------------------

    results = model.predict(
        source=str(image_path),
        conf=CONF_THRESHOLD,
        verbose=False
    )

    result = results[0]

    # --------------------------------------------------------
    # Extract confident CAR predictions
    # --------------------------------------------------------

    predicted_car_boxes = []

    predicted_car_confidences = []

    if result.boxes is not None:

        for box, cls, conf in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
            result.boxes.conf.cpu().numpy()
        ):

            class_id = int(cls)

            confidence = float(conf)

            # Only CAR predictions
            if class_id != CAR_CLASS:
                continue

            # Confidence >= 0.25
            if confidence < CONF_THRESHOLD:
                continue

            predicted_car_boxes.append(
                box.tolist()
            )

            predicted_car_confidences.append(
                confidence
            )

    # --------------------------------------------------------
    # Match GT cars to predicted cars
    # --------------------------------------------------------

    matched_predictions = set()

    image_has_localization_failure = False

    for gt_index, gt_box in enumerate(
        gt_car_boxes
    ):

        best_iou = 0.0

        best_prediction_index = None

        # ----------------------------------------------------
        # Find prediction with highest IoU
        # ----------------------------------------------------

        for pred_index, pred_box in enumerate(
            predicted_car_boxes
        ):

            # One prediction can match only one GT
            if pred_index in matched_predictions:
                continue

            iou = calculate_iou(
                gt_box,
                pred_box
            )

            if iou > best_iou:

                best_iou = iou

                best_prediction_index = (
                    pred_index
                )

        # ====================================================
        # LOCALIZATION FAILURE
        # ====================================================

        if (
            best_prediction_index is not None
            and best_iou < IOU_THRESHOLD
        ):

            image_has_localization_failure = True

            total_localization_failures += 1

            prediction_confidence = (
                predicted_car_confidences[
                    best_prediction_index
                ]
            )

            report_rows.append([
                image_path.name,
                gt_index,
                round(
                    prediction_confidence,
                    4
                ),
                round(
                    best_iou,
                    4
                )
            ])

        # ----------------------------------------------------
        # Correctly localized car
        # ----------------------------------------------------

        elif (
            best_prediction_index is not None
            and best_iou >= IOU_THRESHOLD
        ):

            matched_predictions.add(
                best_prediction_index
            )

    # ========================================================
    # COPY IMAGE + LABEL
    # ========================================================

    if image_has_localization_failure:

        images_with_localization_failures += 1

        # ----------------------------------------------------
        # Create annotated COPY containing GT boxes
        # ----------------------------------------------------

        output_image_path = (
            OUTPUT_IMAGE_DIR /
            image_path.name
        )

        draw_ground_truth_boxes(
            image_path,
            label_path,
            output_image_path,
            image_width,
            image_height
        )

        # Copy original label
        if label_path.exists():

            shutil.copy2(
                label_path,
                OUTPUT_LABEL_DIR /
                label_path.name
            )


# ============================================================
# 13. SAVE CSV
# ============================================================

csv_path = OUTPUT_DIR / "report.csv"

with open(
    csv_path,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "image",
        "gt_car_index",
        "prediction_confidence",
        "best_iou"
    ])

    writer.writerows(report_rows)


# ============================================================
# 14. FINAL RESULTS
# ============================================================

print()
print("=" * 70)
print("LOCALIZATION FAILURE RESULTS")
print("=" * 70)

print(
    f"Validation images: "
    f"{len(image_files)}"
)

print(
    f"Images containing cars: "
    f"{total_images_with_cars}"
)

print(
    f"Ground-truth car instances: "
    f"{total_gt_cars}"
)

print(
    f"Localization failures: "
    f"{total_localization_failures}"
)

print(
    f"Images containing localization failures: "
    f"{images_with_localization_failures}"
)

print()
print("Annotated output images:")
print(OUTPUT_IMAGE_DIR)

print()
print("Output labels:")
print(OUTPUT_LABEL_DIR)

print()
print("CSV report:")
print(csv_path)

print()
print("=" * 70)
print("DONE")
print("=" * 70)