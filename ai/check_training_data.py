import os
from pathlib import Path

print("=" * 70)
print("CHECKING TRAINING DATA AND MODELS")
print("=" * 70)

# Check training data
train_data_path = Path("data/raw/train_data")
test_data_path = Path("data/raw/test_data")
labels_path = Path("data/raw/Data_Entry_2017_v2020.csv")

print("\n📁 TRAINING DATA:")
if train_data_path.exists():
    train_files = list(train_data_path.glob("*"))
    train_images = [f for f in train_files if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
    print(f"  Train folder: ✓ Exists")
    print(f"  Train images: {len(train_images)} files")
    if len(train_images) == 0:
        print("  ⚠️  WARNING: No training images found!")
else:
    print(f"  Train folder: ✗ Not found")

if test_data_path.exists():
    test_files = list(test_data_path.glob("*"))
    test_images = [f for f in test_files if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
    print(f"  Test folder: ✓ Exists")
    print(f"  Test images: {len(test_images)} files")
else:
    print(f"  Test folder: ✗ Not found")

if labels_path.exists():
    print(f"  Labels file: ✓ Found")
else:
    print(f"  Labels file: ✗ Not found")

# Check for trained models
print("\n🤖 TRAINED MODELS:")
experiments_path = Path("experiments")
if experiments_path.exists():
    experiment_dirs = [d for d in experiments_path.iterdir() if d.is_dir()]
    
    if len(experiment_dirs) == 0:
        print("  No experiment folders found")
    else:
        for exp_dir in experiment_dirs:
            checkpoint_dir = exp_dir / "checkpoints"
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob("*.pth"))
                if checkpoints:
                    for ckpt in checkpoints:
                        size_mb = ckpt.stat().st_size / (1024 * 1024)
                        print(f"\n  📦 {exp_dir.name}/")
                        print(f"     File: {ckpt.name}")
                        print(f"     Size: {size_mb:.2f} MB")
                        
                        # Check if dummy model
                        import torch
                        try:
                            checkpoint = torch.load(str(ckpt), map_location='cpu', weights_only=False)
                            if 'note' in checkpoint and 'DUMMY' in checkpoint['note']:
                                print(f"     Type: ⚠️  DUMMY MODEL (random weights)")
                            else:
                                print(f"     Type: ✓ Real trained model")
                                print(f"     Epoch: {checkpoint.get('epoch', 'N/A')}")
                                print(f"     Val Acc: {checkpoint.get('val_acc', 'N/A')}")
                        except Exception as e:
                            print(f"     Type: ⚠️  Error loading: {e}")
else:
    print("  Experiments folder not found")

# Recommendations
print("\n" + "=" * 70)
print("RECOMMENDATIONS:")
print("=" * 70)

train_image_count = len(train_images) if train_data_path.exists() else 0

if train_image_count == 0:
    print("\n❌ NO TRAINING DATA FOUND")
    print("   You need to download training data first.")
    print("   See TRAINING_GUIDE.md for instructions.")
    print("\n   Quick option: Download from Kaggle")
    print("   https://www.kaggle.com/datasets/nih-chest-xrays/data")
elif train_image_count < 100:
    print(f"\n⚠️  VERY FEW TRAINING IMAGES ({train_image_count})")
    print("   You need more images for a good model.")
    print("   Recommended: 1000+ images minimum")
else:
    print(f"\n✓ TRAINING DATA AVAILABLE ({train_image_count} images)")
    print("   You can train a model now!")
    print("\n   Quick test (5 epochs, 10-30 mins):")
    print("   python main.py train --config configs/config_quick_test.yaml --experiment-name real_model")
    print("\n   Full training (50 epochs, 2-4 hours):")
    print("   python main.py train --config configs/config.yaml --experiment-name production_model")

# Check current backend config
print("\n" + "=" * 70)
print("CURRENT BACKEND CONFIGURATION:")
print("=" * 70)

backend_env = Path("../backend/.env")
if backend_env.exists():
    with open(backend_env, 'r') as f:
        for line in f:
            if 'CHECKPOINT_PATH' in line:
                print(f"  {line.strip()}")
                if 'test_run_fixed_20250928_224821' in line:
                    print("  ⚠️  This points to the DUMMY MODEL!")
                    print("  ⚠️  Update this after training a real model")
else:
    print("  .env file not found")

print("\n" + "=" * 70)
