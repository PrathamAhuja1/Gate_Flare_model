import cv2
import os
import numpy as np
import albumentations as A
from tqdm import tqdm

# --- CONFIGURATION ---
VIDEO_PATH = 'Gate_1.mp4'
OUTPUT_FOLDER = 'dataset_gate'
TARGET_COUNT = 200

INPUT_DIR = "dataset_gate_full"
OUTPUT_DIR = "augmented_dataset"
NUM_AUGS_PER_IMAGE = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

transform = A.Compose([
    A.HorizontalFlip(p=0.4),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=30, p=0.5), 

    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    
    # Reduced hue/sat shift slightly to keep colors more natural
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
    
    A.MotionBlur(blur_limit=5, p=0.2),

    # Reduced color shift limits slightly to avoid "colored noise" artifacts
    A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3),
])

# def create_dataset():
#     if not os.path.exists(OUTPUT_FOLDER):
#         os.makedirs(OUTPUT_FOLDER)
#         print(f"Created folder: {OUTPUT_FOLDER}")
    
#     cap = cv2.VideoCapture(VIDEO_PATH)

#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     print(f"Total frames in video: {total_frames}")

#     if total_frames < TARGET_COUNT:
#         print("Warning: Video has fewer frames than the target count. Some frames may be duplicated.")

#     interval = total_frames / TARGET_COUNT
    
#     saved_count = 801
    
#     print(f"Starting extraction of {TARGET_COUNT} frames...")

#     for i in range(TARGET_COUNT):
#         frame_id = int(i * interval)
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
#         ret, frame = cap.read()
#         if not ret:
#             print(f"Could not read frame at index {frame_id}")
#             continue

#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     #    augmented = transform(image=frame_rgb)['image']
#         frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
#         filename = os.path.join(OUTPUT_FOLDER, f"img_{saved_count:04d}.jpg")
#         cv2.imwrite(filename, frame_bgr)
        
#         saved_count += 1
#         if saved_count % 50 == 0:
#             print(f"Saved {saved_count}/{TARGET_COUNT} images...")

#     cap.release()
#     print(f"Saved {saved_count} augmented images to '{OUTPUT_FOLDER}/'")

# if __name__ == "__main__":
#     create_dataset()

# Paths


# Augmentation pipeline

count = 0

for img_name in tqdm(os.listdir(INPUT_DIR)):
    print(count)
    img_path = os.path.join(INPUT_DIR, img_name)
    image = cv2.imread(img_path)

    if image is None:
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    base_name, ext = os.path.splitext(img_name)

    count += 1

    for i in range(NUM_AUGS_PER_IMAGE):
        augmented = transform(image=image)["image"]
        augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

        out_name = f"{base_name}_aug_{i}{ext}"
        cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), augmented)