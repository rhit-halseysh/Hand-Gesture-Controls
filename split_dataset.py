"""
Script to split dataset into train and validation sets.
Reorganizes images from hagrid-export_2000_images/ into train/ and val/ folders.

Usage: python split_dataset.py
"""
import os
import shutil
import random
from pathlib import Path


# ========== CONFIGURATION ==========
SOURCE_DIR = 'hagrid-export_2000_images'  # Source directory with class folders
OUTPUT_DIR = 'dataset'                     # Output directory (will create train/ and val/)
TRAIN_RATIO = 0.8                         # 80% train, 20% validation
RANDOM_SEED = 42                          # For reproducibility
# ====================================


def split_dataset(source_dir, output_dir, train_ratio=0.8, seed=42):
    """
    Split dataset into train and validation sets.
    
    Args:
        source_dir: Directory containing class folders with images
        output_dir: Output directory (will create train/ and val/ subdirs)
        train_ratio: Ratio of images to use for training (default 0.8)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    train_dir = output_path / 'train'
    val_dir = output_path / 'val'
    
    if output_path.exists():
        print(f"Warning: {output_dir} already exists!")
        response = input("Do you want to continue? This may overwrite existing files. (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all class directories (excluding non-directory files like .json)
    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    if not class_dirs:
        print(f"Error: No class directories found in {source_dir}")
        return
    
    print(f"Found {len(class_dirs)} gesture classes")
    print(f"Train/Val split: {train_ratio:.0%} / {1-train_ratio:.0%}")
    print("=" * 60)
    
    total_train = 0
    total_val = 0
    
    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        
        # Get all image files in this class directory
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        images = [f for f in class_dir.iterdir() 
                 if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not images:
            print(f"⚠ Skipping {class_name}: No images found")
            continue
        
        # Shuffle images
        random.shuffle(images)
        
        # Calculate split point
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Create class directories in train and val
        train_class_dir = train_dir / class_name
        val_class_dir = val_dir / class_name
        train_class_dir.mkdir(exist_ok=True)
        val_class_dir.mkdir(exist_ok=True)
        
        # Copy training images
        for img in train_images:
            shutil.copy2(img, train_class_dir / img.name)
        
        # Copy validation images
        for img in val_images:
            shutil.copy2(img, val_class_dir / img.name)
        
        total_train += len(train_images)
        total_val += len(val_images)
        
        print(f"✓ {class_name:20s} - Train: {len(train_images):4d}, Val: {len(val_images):4d}, Total: {len(images):4d}")
    
    print("=" * 60)
    print(f"Dataset split complete!")
    print(f"  Total training images:   {total_train}")
    print(f"  Total validation images: {total_val}")
    print(f"  Total images:            {total_train + total_val}")
    print(f"\nOutput directory: {output_path.absolute()}")
    print(f"  - {train_dir.relative_to(Path.cwd())}")
    print(f"  - {val_dir.relative_to(Path.cwd())}")


if __name__ == '__main__':
    print("Dataset Splitter")
    print("=" * 60)
    
    # Check if source directory exists
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory '{SOURCE_DIR}' not found!")
        print("Please update SOURCE_DIR in the script.")
        exit(1)
    
    # Run the split
    split_dataset(SOURCE_DIR, OUTPUT_DIR, TRAIN_RATIO, RANDOM_SEED)
    
    print("\nNext steps:")
    print("1. Verify the dataset structure in the 'dataset/' folder")
    print("2. Update DATA_DIR in train.py to 'dataset'")
    print("3. Run: python train.py")
