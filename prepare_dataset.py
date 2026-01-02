import os
import shutil
import random
import yaml

# --- CONFIGURATION ---
# Source Folders (Must match your actual folder names)
IMAGES_SOURCE_DIR = "augmented_dataset"      # Where your images are
LABELS_SOURCE_DIR = "labels_dataset"   # Where your .txt files are

# Destination Folder (Where the ready-to-train dataset will go)
DATASET_DIR = "gate_dataset"

# Class Setup
CLASS_NAMES = {0: 'gate'}  # Single class "flare"

# Split Ratio
TRAIN_RATIO = 0.8  # 80% Training, 20% Validation
# ---------------------

def setup_directories():
    """Creates the YOLO directory structure."""
    if os.path.exists(DATASET_DIR):
        print(f"Removing existing '{DATASET_DIR}' folder to start fresh...")
        shutil.rmtree(DATASET_DIR)
    
    subdirs = [
        "train/images", "train/labels",
        "val/images", "val/labels"
    ]
    for sub in subdirs:
        os.makedirs(os.path.join(DATASET_DIR, sub), exist_ok=True)
    print(f"Created dataset structure in '{DATASET_DIR}/'")

def match_files():
    """Matches images with their corresponding annotation text files."""
    matched_pairs = []
    
    # Get list of images
    try:
        image_files = os.listdir(IMAGES_SOURCE_DIR)
    except FileNotFoundError:
        print(f"Error: Image folder '{IMAGES_SOURCE_DIR}' not found.")
        return []

    print(f"Scanning {len(image_files)} images...")

    for img_file in image_files:
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
            # Get filename without extension (e.g., 'img_0037')
            stem = os.path.splitext(img_file)[0]
            
            # Construct expected label path
            expected_txt = stem + ".txt"
            txt_path = os.path.join(LABELS_SOURCE_DIR, expected_txt)
            
            # Check if the label file exists in the annotations folder
            if os.path.exists(txt_path):
                img_path = os.path.join(IMAGES_SOURCE_DIR, img_file)
                matched_pairs.append((img_path, txt_path, img_file, expected_txt))
            else:
                # Optional: print warning for missing labels
                # print(f"Warning: Label missing for {img_file}")
                pass
                
    return matched_pairs

def copy_files(pairs):
    """Splits the pairs and copies them to the destination."""
    random.shuffle(pairs)
    split_idx = int(len(pairs) * TRAIN_RATIO)
    
    train_set = pairs[:split_idx]
    val_set = pairs[split_idx:]
    
    print(f"\nMatched {len(pairs)} image-label pairs.")
    print(f"Training: {len(train_set)} | Validation: {len(val_set)}")
    
    def copy_batch(batch, split):
        for img_src, txt_src, img_name, txt_name in batch:
            # Copy Image
            shutil.copy2(img_src, os.path.join(DATASET_DIR, split, "images", img_name))
            # Copy Label
            shutil.copy2(txt_src, os.path.join(DATASET_DIR, split, "labels", txt_name))
            
    print("Copying training files...")
    copy_batch(train_set, "train")
    
    print("Copying validation files...")
    copy_batch(val_set, "val")

def create_yaml():
    """Generates the data.yaml file required by YOLO."""
    yaml_content = f"""
path: {os.path.abspath(DATASET_DIR)} # Absolute path to dataset
train: train/images
val: val/images

nc: {len(CLASS_NAMES)}
names:
"""
    for idx, name in CLASS_NAMES.items():
        yaml_content += f"  {idx}: {name}\n"
        
    with open(os.path.join(DATASET_DIR, "data.yaml"), "w") as f:
        f.write(yaml_content.strip())
    print("\nCreated 'data.yaml' file.")

if __name__ == "__main__":
    setup_directories()
    pairs = match_files()
    
    if len(pairs) > 0:
        copy_files(pairs)
        create_yaml()
        print("\nSUCCESS: Dataset is ready for training!")
    else:
        print("\nFAILED: No matching files found. Check your folder names.")