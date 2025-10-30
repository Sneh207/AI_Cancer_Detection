#!/usr/bin/env python3
"""
Check training data and labels
"""
import pandas as pd
import os
from pathlib import Path

# Check CSV
csv_path = 'data/raw/ChestXray_Binary_Labels.csv'
print("=" * 60)
print("CHECKING LABELS CSV")
print("=" * 60)

df = pd.read_csv(csv_path)
print(f"\nShape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df.head())

if len(df.columns) >= 2:
    label_col = df.columns[1]
    print(f"\nLabel column: '{label_col}'")
    print(f"\nLabel distribution:")
    print(df[label_col].value_counts())
    print(f"\nUnique labels: {df[label_col].unique()}")

# Check training images
train_dir = 'data/raw/train_data/train'
print("\n" + "=" * 60)
print("CHECKING TRAINING IMAGES")
print("=" * 60)

if os.path.exists(train_dir):
    image_files = list(Path(train_dir).glob('*.png'))
    print(f"\nTotal images: {len(image_files)}")
    print(f"\nFirst 10 images:")
    for img in image_files[:10]:
        print(f"  - {img.name}")
    
    # Check if images in CSV exist
    print("\n" + "=" * 60)
    print("CHECKING IMAGE-LABEL MATCHING")
    print("=" * 60)
    
    image_col = df.columns[0]
    csv_images = set(df[image_col].values)
    actual_images = set([img.name for img in image_files])
    
    print(f"\nImages in CSV: {len(csv_images)}")
    print(f"Images in folder: {len(actual_images)}")
    
    matching = csv_images.intersection(actual_images)
    print(f"Matching images: {len(matching)}")
    
    if len(matching) > 0:
        print("\n✅ Data is ready for training!")
        print(f"\nSample matched entries:")
        matched_df = df[df[image_col].isin(actual_images)].head()
        print(matched_df)
    else:
        print("\n⚠️ WARNING: No matching images found!")
        print("\nSample CSV entries:")
        print(df[image_col].head())
        print("\nSample image files:")
        for img in list(actual_images)[:5]:
            print(f"  - {img}")
else:
    print(f"\n❌ Training directory not found: {train_dir}")
