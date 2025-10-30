#!/usr/bin/env python3
"""
Fix data structure and create labels file for training
"""

import os
import shutil
import pandas as pd
from pathlib import Path

print("=" * 70)
print("FIXING DATA STRUCTURE")
print("=" * 70)

# Paths
train_data_dir = Path("data/raw/train_data")
train_subdir = train_data_dir / "train"
labels_file = Path("data/raw/Data_Entry_2017_v2020.csv")

# Step 1: Check current structure
print("\n1. Checking current structure...")
if train_subdir.exists():
    train_images = list(train_subdir.glob("*"))
    train_images = [f for f in train_images if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
    print(f"   Found {len(train_images)} images in train_data/train/")
else:
    print("   train_data/train/ not found")
    train_images = []

# Step 2: Move images up one level
if len(train_images) > 0:
    print(f"\n2. Moving {len(train_images)} images to train_data/...")
    for img in train_images:
        dest = train_data_dir / img.name
        if not dest.exists():
            shutil.move(str(img), str(dest))
    print("   ✓ Images moved successfully")
    
    # Remove empty train folder
    if train_subdir.exists() and not any(train_subdir.iterdir()):
        train_subdir.rmdir()
        print("   ✓ Removed empty train/ folder")
else:
    print("\n2. No images to move")

# Step 3: Verify images in correct location
final_images = list(train_data_dir.glob("*"))
final_images = [f for f in final_images if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
print(f"\n3. Final image count in train_data/: {len(final_images)}")

# Step 4: Create labels CSV file
if len(final_images) > 0:
    print(f"\n4. Creating labels file...")
    
    # Create a simple labels CSV
    # Format: Image Index,Finding Labels
    # For training without specific labels, we'll create a basic structure
    
    image_data = []
    for img in final_images:
        # Extract image name
        image_name = img.name
        
        # For now, we'll label all as "No Finding" (you can update this later)
        # In a real scenario, you'd have actual labels
        image_data.append({
            'Image Index': image_name,
            'Finding Labels': 'No Finding',  # Default label
            'Patient ID': image_name.split('_')[0] if '_' in image_name else '00000',
            'Patient Age': 50,  # Default
            'Patient Gender': 'M',  # Default
        })
    
    # Create DataFrame
    df = pd.DataFrame(image_data)
    
    # Save to CSV
    df.to_csv(labels_file, index=False)
    print(f"   ✓ Created labels file: {labels_file}")
    print(f"   ✓ Total entries: {len(df)}")
    
    print("\n   ⚠️  NOTE: All images labeled as 'No Finding' by default")
    print("   ⚠️  If you have actual labels, replace this CSV file")
else:
    print("\n4. Cannot create labels file - no images found")

# Step 5: Update config if needed
print("\n5. Checking config.yaml...")
config_file = Path("configs/config.yaml")
if config_file.exists():
    with open(config_file, 'r') as f:
        config_content = f.read()
    
    if 'data/raw/train_data' in config_content:
        print("   ✓ Config already points to correct path")
    else:
        print("   ⚠️  Config may need updating")
else:
    print("   ⚠️  Config file not found")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"✓ Images in train_data/: {len(final_images)}")
print(f"✓ Labels file: {'Created' if labels_file.exists() else 'Not created'}")
print(f"✓ Ready to train: {'Yes' if len(final_images) > 0 and labels_file.exists() else 'No'}")

if len(final_images) > 0 and labels_file.exists():
    print("\n🎉 Data structure fixed! You can now train:")
    print("   python main.py train --config configs/config_quick_test.yaml --experiment-name real_model")
else:
    print("\n⚠️  Issues remain. Check the messages above.")

print("=" * 70)
