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

OUTPUT_DIR = DATASET / "false_negatives"

OUTPUT_IMAGE_DIR = OUTPUT_DIR / "images"
OUTPUT_LABEL_DIR = OUTPUT_DIR / "labels"

OUTPUT_IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_LABEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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

# Use a very low inference threshold so that
# even weak predictions can be inspected.
INFERENCE_THRESHOLD = 0.001


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
# 5. YOLO FORMAT -> PIXEL BOX
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
# 6. READ ALL GROUND-TRUTH BOXES
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
# 7. GET GROUND-TRUTH CAR BOXES
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
# 8. DRAW GROUND-TRUTH BOXES
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

    # Get ALL ground-truth boxes
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

        # RED = CAR
        # BLUE = OTHER GT OBJECT
        if class_id == CAR_CLASS:
            outline_color = "red"
        else:
            outline_color = "blue"

        # Draw GT box
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=outline_color,
            width=3
        )

        # Text
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

        # Text background
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

        # Text
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

    # ========================================================
    # SAVE ANNOTATED COPY
    # ========================================================

    image.save(output_path)


# ============================================================
# 9. GET VALIDATION IMAGES
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
# 10. COUNTERS
# ============================================================

total_images = len(image_files)

images_with_cars = 0

total_gt_car_instances = 0

false_negative_instances = 0

false_negative_images = 0

report_rows = []


# ============================================================
# 11. PROCESS EVERY VALIDATION IMAGE
# ============================================================

print("=" * 70)
print("SEARCHING FOR FALSE NEGATIVES")
print("=" * 70)


for index, image_path in enumerate(
    image_files,
    start=1
):

    print(
        f"[{index}/{total_images}] "
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
    # Get image size
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
    # Find GT car boxes
    # --------------------------------------------------------

    gt_car_boxes = get_ground_truth_cars(
        label_path,
        image_width,
        image_height
    )

    # If there is no GT car, skip the image
    if len(gt_car_boxes) == 0:
        continue

    images_with_cars += 1

    total_gt_car_instances += len(
        gt_car_boxes
    )

    # ========================================================
    # RUN YOLO
    # ========================================================

    results = model.predict(
        source=str(image_path),
        conf=INFERENCE_THRESHOLD,
        verbose=False
    )

    result = results[0]

    # ========================================================
    # GET CAR PREDICTIONS
    # ========================================================

    car_predictions = []

    if result.boxes is not None:

        for box, cls, conf in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
            result.boxes.conf.cpu().numpy()
        ):

            class_id = int(cls)

            confidence = float(conf)

            # Keep only CAR predictions
            if class_id != CAR_CLASS:
                continue

            car_predictions.append({
                "box": box.tolist(),
                "confidence": confidence
            })

    # ========================================================
    # CHECK EACH GT CAR
    # ========================================================

    image_has_false_negative = False

    for gt_index, gt_box in enumerate(
        gt_car_boxes
    ):

        # ----------------------------------------------------
        # Find whether ANY CAR prediction exists
        # with confidence >= 0.25
        # ----------------------------------------------------

        confident_car_prediction = False

        best_car_confidence = 0.0

        for prediction in car_predictions:

            confidence = prediction[
                "confidence"
            ]

            if confidence > best_car_confidence:

                best_car_confidence = confidence

            if confidence >= CONF_THRESHOLD:

                confident_car_prediction = True

        # ====================================================
        # FALSE NEGATIVE
        # ====================================================

        if not confident_car_prediction:

            image_has_false_negative = True

            false_negative_instances += 1

            # Determine the situation
            if len(car_predictions) == 0:

                reason = (
                    "no_car_prediction"
                )

            else:

                reason = (
                    "car_prediction_below_confidence"
                )

            report_rows.append([
                image_path.name,
                gt_index,
                reason,
                round(
                    best_car_confidence,
                    4
                )
            ])

    # ========================================================
    # SAVE IMAGE IF IT CONTAINS A FALSE NEGATIVE
    # ========================================================

    if image_has_false_negative:

        false_negative_images += 1

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Original image is READ only.
        #
        # GT boxes are drawn on a COPY.
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

        # ----------------------------------------------------
        # Copy label file
        # ----------------------------------------------------

        if label_path.exists():

            shutil.copy2(
                label_path,
                OUTPUT_LABEL_DIR /
                label_path.name
            )


# ============================================================
# 12. SAVE CSV
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
        "reason",
        "best_car_confidence"
    ])

    writer.writerows(report_rows)


# ============================================================
# 13. FINAL RESULTS
# ============================================================

print()
print("=" * 70)
print("FALSE NEGATIVE RESULTS")
print("=" * 70)

print(
    f"Validation images: "
    f"{total_images}"
)

print(
    f"Images containing cars: "
    f"{images_with_cars}"
)

print(
    f"Ground-truth car instances: "
    f"{total_gt_car_instances}"
)

print(
    f"False-negative car instances: "
    f"{false_negative_instances}"
)

print(
    f"Images containing false negatives: "
    f"{false_negative_images}"
)

print()
print("Annotated images:")
print(OUTPUT_IMAGE_DIR)

print()
print("Copied labels:")
print(OUTPUT_LABEL_DIR)

print()
print("CSV report:")
print(csv_path)

print()
print("=" * 70)
print("DONE")
print("=" * 70)