import os
import cv2
from ultralytics import YOLO

# ---- CHANGE THESE 4 LINES ----
IMAGES_FOLDER = r"C:\Users\shiif\Downloads\yoloduke\datasets\coco_instance_balanced5_undersampled\hard_examples\images"
OUTPUT_FOLDER = r"C:\Users\shiif\Downloads\yoloduke\datasets\coco_instance_balanced5_undersampled\hard_examples\bbfolder"
PRETRAINED_MODEL = r"C:\Users\shiif\Downloads\yoloduke\yolov8s.pt"
MY_MODEL = r"C:\Users\shiif\Downloads\yoloduke\runs\detect\train-7\weights\best.pt"
# ------------------------------

BLUE = (255, 0, 0)   # Pretrained = Blue
GREEN = (0, 255, 0)  # Trained = Green

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

pretrained = YOLO(PRETRAINED_MODEL)
mine = YOLO(MY_MODEL)


def draw(img, result, color, thickness, model_label):

    for box in result.boxes:

        # Bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Class name
        name = result.names[int(box.cls[0])]

        # Confidence
        conf = float(box.conf[0])

        # Label
        label = f"{model_label}: {name} {conf:.2f}"

        # Draw bounding box
        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            color,
            thickness
        )

        # Draw label
        cv2.putText(
            img,
            label,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )


for file in os.listdir(IMAGES_FOLDER):

    if not file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        continue

    img = cv2.imread(
        os.path.join(IMAGES_FOLDER, file)
    )

    if img is None:
        continue

    # --------------------------------
    # PRETRAINED MODEL
    # Blue boxes
    # --------------------------------
    pretrained_result = pretrained(
        img,
        verbose=False
    )[0]

    draw(
        img,
        pretrained_result,
        BLUE,
        4,
        "Pretrained"
    )

    # --------------------------------
    # TRAINED MODEL
    # Green boxes
    # --------------------------------
    trained_result = mine(
        img,
        verbose=False
    )[0]

    draw(
        img,
        trained_result,
        GREEN,
        2,
        "Trained"
    )

    # Save comparison image
    output_path = os.path.join(
        OUTPUT_FOLDER,
        file
    )

    cv2.imwrite(
        output_path,
        img
    )

    print("Saved:", output_path)