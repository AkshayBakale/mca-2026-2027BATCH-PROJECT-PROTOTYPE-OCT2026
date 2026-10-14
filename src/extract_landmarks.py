import cv2
import mediapipe as mp
import os
import csv

DATASET_DIR = "data/raw_images"
SAVE_PATH = "data/landmarks/landmark_data.csv"
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Optional: cap how many images per class to process (speeds things up a LOT)
MAX_PER_CLASS = 400  # set to None to use all images

with open(SAVE_PATH, mode="w", newline="") as f:
    writer = csv.writer(f)
    header = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
    writer.writerow(header)

    labels = sorted(os.listdir(DATASET_DIR))
    for label in labels:
        label_dir = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(label_dir):
            continue

        images = os.listdir(label_dir)
        if MAX_PER_CLASS:
            images = images[:MAX_PER_CLASS]

        print(f"Processing '{label}' — {len(images)} images")
        saved_count = 0

        for img_name in images:
            img_path = os.path.join(label_dir, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                row = [label]
                for lm in result.multi_hand_landmarks[0].landmark:
                    row.extend([lm.x, lm.y, lm.z])
                writer.writerow(row)
                saved_count += 1

        print(f"  -> {saved_count} hands detected and saved")

print("\nDone. Landmark CSV saved at:", SAVE_PATH)