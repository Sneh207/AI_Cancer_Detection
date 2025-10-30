#!/usr/bin/env python3
"""
Create balanced labels for testing (50% cancer, 50% no cancer)
⚠️ WARNING: This creates SYNTHETIC labels for testing only!
⚠️ NOT for medical use - labels are randomly assigned!
"""

import pandas as pd
import random
from pathlib import Path

print("=" * 70)
print("CREATING BALANCED LABELS FOR TESTING")
print("=" * 70)
print("\n⚠️  WARNING: This creates SYNTHETIC labels!")
print("⚠️  Labels are randomly assigned - NOT real medical diagnoses!")
print("⚠️  Use ONLY for testing the system workflow!\n")

# Paths
train_data_dir = Path("data/raw/train_data")
labels_file = Path("data/raw/Data_Entry_2017_v2020.csv")
backup_file = Path("data/raw/Data_Entry_2017_v2020.csv.backup")

# Get all images
images = list(train_data_dir.glob("*"))
images = [f for f in images if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]

print(f"Found {len(images)} images")

if len(images) == 0:
    print("❌ No images found!")
    exit(1)

# Backup existing labels if present
if labels_file.exists():
    import shutil
    shutil.copy(labels_file, backup_file)
    print(f"✓ Backed up existing labels to: {backup_file}")

# Create balanced labels
image_data = []
random.seed(42)  # For reproducibility

# Shuffle images
shuffled_images = images.copy()
random.shuffle(shuffled_images)

# Assign 50% as cancer, 50% as no finding
split_point = len(shuffled_images) // 2

for i, img in enumerate(shuffled_images):
    image_name = img.name
    
    # First half = Cancer, Second half = No Finding
    if i < split_point:
        finding = "Cancer"
    else:
        finding = "No Finding"
    
    image_data.append({
        'Image Index': image_name,
        'Finding Labels': finding,
        'Patient ID': f'{i:05d}',
        'Patient Age': random.randint(30, 80),
        'Patient Gender': random.choice(['M', 'F']),
    })

# Create DataFrame
df = pd.DataFrame(image_data)

# Save to CSV
df.to_csv(labels_file, index=False)

# Statistics
cancer_count = len(df[df['Finding Labels'] == 'Cancer'])
normal_count = len(df[df['Finding Labels'] == 'No Finding'])

print(f"\n✓ Created balanced labels file: {labels_file}")
print(f"\nLabel Distribution:")
print(f"  Cancer: {cancer_count} ({cancer_count/len(df)*100:.1f}%)")
print(f"  No Finding: {normal_count} ({normal_count/len(df)*100:.1f}%)")
print(f"  Total: {len(df)}")

print("\n" + "=" * 70)
print("READY TO TRAIN WITH BALANCED DATA")
print("=" * 70)
print("\nNext step:")
print("  python main.py train --config configs/config_quick_test.yaml --experiment-name balanced_model")
print("\n⚠️  Remember: These labels are SYNTHETIC for testing only!")
print("=" * 70)
