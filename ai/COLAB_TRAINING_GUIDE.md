# 🚀 Google Colab GPU Training Guide

Train your AI Lung Cancer Detection model on free GPU with automatic class balancing.

---

## 📋 Quick Start

### 1. Create ZIP on Your PC

```powershell
# Navigate to project root
cd C:\AI_Lung_Cancer\AI_Cancer_Detection

# Create single ZIP with everything (code + data)
Compress-Archive -Path "ai" -DestinationPath "ai.zip" -Force

# Verify
Write-Host "✅ Created: ai.zip"
Write-Host "Size: $((Get-Item ai.zip).Length / 1GB) GB"
```

### 2. Upload to Google Drive

1. Go to https://drive.google.com
2. Create folder: `AI_Cancer_Detection`
3. Upload `ai.zip` to this folder
4. Wait for upload to complete

### 3. Open Google Colab

1. Go to https://colab.research.google.com/
2. File → New notebook
3. **Enable GPU:** Runtime → Change runtime type → **T4 GPU** → Save

---

## 📱 Colab Cells (Copy-Paste in Order)

### ✅ CELL 1: Install Dependencies (Run First!)

```python
print("🔧 Installing compatible dependencies...\n")

# Fix numpy/pandas compatibility
!pip uninstall -y numpy pandas scikit-learn -q
!pip install -q numpy==1.24.3 pandas==2.0.3 scikit-learn==1.3.2

# Install other packages
!pip install -q albumentations==1.3.1 timm==0.9.12 pyyaml==6.0.1 \
             matplotlib==3.8.2 seaborn==0.13.0 tensorboard

print("\n✅ Dependencies installed!")
```

---

### ✅ CELL 2: Check GPU

```python
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("❌ No GPU! Runtime → Change runtime type → T4 GPU")
```

---

### ✅ CELL 3: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
print("✅ Drive mounted")
```

---

### ✅ CELL 4: Extract Project Files

```python
import os, zipfile, glob

# Find ai.zip
print("🔍 Searching for ai.zip...")
zips = glob.glob('/content/drive/MyDrive/**/ai.zip', recursive=True)

if not zips:
    print("❌ ai.zip not found!")
    print("Upload to: My Drive/AI_Cancer_Detection/ai.zip")
    raise FileNotFoundError("ai.zip not found")

zip_path = zips[0]
print(f"✅ Found: {zip_path}")
print(f"   Size: {os.path.getsize(zip_path)/(1024**3):.2f} GB")

# Extract
print("\n📦 Extracting...")
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall('/content/')
print("✅ Extracted")

# Find project root
project_root = '/content/ai' if os.path.exists('/content/ai') else '/content'

if not os.path.exists(f"{project_root}/main.py"):
    print("❌ main.py not found!")
    raise FileNotFoundError("Invalid project structure")

os.chdir(project_root)
print(f"\n✅ Working directory: {os.getcwd()}")

# Verify structure
print("\n📂 Verification:")
for item in ['main.py', 'src', 'configs', 'data']:
    status = "✅" if os.path.exists(item) else "❌"
    print(f"  {status} {item}")
```

---

### ✅ CELL 5: Verify Dataset

```python
import pandas as pd
import os

# Load labels
labels_file = f"{project_root}/data/raw/ChestXray_Binary_Labels.csv"

if not os.path.exists(labels_file):
    print(f"❌ Labels not found: {labels_file}")
    raise FileNotFoundError("Labels CSV not found")

df = pd.read_csv(labels_file)

print("📊 Dataset Statistics:")
print(f"  Total images: {len(df)}")
print(f"\n  Label distribution:")
for label, count in df['BinaryLabel'].value_counts().items():
    pct = (count / len(df)) * 100
    print(f"    {label}: {count} ({pct:.1f}%)")

# Check images
img_dir = f"{project_root}/data/raw/train_data/train"
if os.path.exists(img_dir):
    imgs = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"\n✅ Found {len(imgs)} images")
else:
    print(f"\n❌ Images not found: {img_dir}")
```

---

### ✅ CELL 6: Optimize Config for GPU

```python
import yaml

config_path = f"{project_root}/configs/config.yaml"

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# GPU optimizations
config['data']['batch_size'] = 32
config['data']['num_workers'] = 2
config['training']['epochs'] = 50

with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print("✅ Config optimized for GPU:")
print(f"  Model: {config['model']['architecture']}")
print(f"  Batch size: 32")
print(f"  Epochs: 50")
print(f"  Learning rate: {config['training']['learning_rate']}")
```

---

### ✅ CELL 7: Start Training (2-4 hours)

```python
main_script = f"{project_root}/main.py"
config_file = f"{project_root}/configs/config.yaml"

print("🚀 Starting training with balanced sampling...")
print(f"  Device: cuda")
print(f"  Experiment: colab_resnet50_balanced")
print(f"\n⏱️ Estimated time: 2-4 hours for 50 epochs")
print(f"💡 Tip: Close tab - training continues in background\n")
print("="*80 + "\n")

!python {main_script} train \
    --config {config_file} \
    --experiment-name colab_resnet50_balanced \
    --device cuda \
    --seed 42
```

---

### ✅ CELL 8: Check Progress (Run Anytime)

```python
import glob, os, torch

exp_dirs = glob.glob(f"{project_root}/experiments/colab_resnet50_balanced*")

if not exp_dirs:
    print("❌ No experiment found")
else:
    latest = max(exp_dirs, key=os.path.getctime)
    ckpt_path = f"{latest}/checkpoints/best_model.pth"
    
    print(f"📁 Experiment: {os.path.basename(latest)}\n")
    
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        print("✅ Checkpoint found!")
        print(f"  Epoch: {ckpt.get('epoch', 'N/A')}")
        print(f"  Best AUC: {ckpt.get('best_auc', 0):.4f}")
        print(f"  Size: {os.path.getsize(ckpt_path)/(1024**2):.1f} MB")
    else:
        print("⏳ Training in progress...")
```

---

### ✅ CELL 9: Evaluate Model

```python
import glob, os

exp_dirs = glob.glob(f"{project_root}/experiments/colab_resnet50_balanced*")

if exp_dirs:
    latest = max(exp_dirs, key=os.path.getctime)
    ckpt_path = f"{latest}/checkpoints/best_model.pth"
    
    if os.path.exists(ckpt_path):
        print("🔍 Evaluating model...\n")
        
        !python {main_script} evaluate \
            --config {config_file} \
            --checkpoint {ckpt_path} \
            --device cuda
    else:
        print("❌ No checkpoint found")
else:
    print("❌ No experiment found")
```

---

### ✅ CELL 10: View Results

```python
import json, glob, os
from IPython.display import Image, display

exp_dirs = glob.glob(f"{project_root}/experiments/colab_resnet50_balanced*")

if exp_dirs:
    latest = max(exp_dirs, key=os.path.getctime)
    results_dir = f"{latest}/results"
    
    # Load metrics
    eval_files = glob.glob(f"{results_dir}/**/evaluation_results.json", recursive=True)
    if eval_files:
        with open(eval_files[0], 'r') as f:
            results = json.load(f)
        
        print("📊 Model Performance:\n")
        print(f"  AUC-ROC:   {results.get('auc', 0):.4f}")
        print(f"  Accuracy:  {results.get('accuracy', 0):.4f}")
        print(f"  Precision: {results.get('precision', 0):.4f}")
        print(f"  Recall:    {results.get('recall', 0):.4f}")
        print(f"  F1 Score:  {results.get('f1', 0):.4f}")
        
        auc = results.get('auc', 0)
        if auc >= 0.90:
            print("\n✅ Excellent!")
        elif auc >= 0.80:
            print("\n✅ Good")
        elif auc >= 0.70:
            print("\n⚠️ Moderate")
        else:
            print("\n❌ Poor")
    
    # Show plots
    cm_files = glob.glob(f"{results_dir}/**/confusion_matrix.png", recursive=True)
    if cm_files:
        print("\n📈 Confusion Matrix:")
        display(Image(filename=cm_files[0]))
    
    roc_files = glob.glob(f"{results_dir}/**/roc_curve.png", recursive=True)
    if roc_files:
        print("\n📈 ROC Curve:")
        display(Image(filename=roc_files[0]))
```

---

### ✅ CELL 11: Save to Google Drive

```python
import shutil, glob, os

exp_dirs = glob.glob(f"{project_root}/experiments/colab_resnet50_balanced*")

if exp_dirs:
    latest = max(exp_dirs, key=os.path.getctime)
    ckpt_path = f"{latest}/checkpoints/best_model.pth"
    
    if os.path.exists(ckpt_path):
        # Save to Drive
        drive_dest = '/content/drive/MyDrive/AI_Cancer_Detection/trained_models/'
        os.makedirs(drive_dest, exist_ok=True)
        
        # Copy checkpoint
        dest_file = f"{drive_dest}/best_model_colab.pth"
        shutil.copy2(ckpt_path, dest_file)
        
        print(f"✅ Checkpoint saved:")
        print(f"   {dest_file}")
        print(f"   Size: {os.path.getsize(dest_file)/(1024**2):.1f} MB")
        
        # Copy full experiment
        exp_name = os.path.basename(latest)
        shutil.copytree(latest, f"{drive_dest}/{exp_name}", dirs_exist_ok=True)
        
        print(f"\n✅ Full experiment saved:")
        print(f"   {drive_dest}/{exp_name}")
    else:
        print("❌ No checkpoint found")
```

---

### ✅ CELL 12: Download to PC (Optional)

```python
from google.colab import files
import glob, os

exp_dirs = glob.glob(f"{project_root}/experiments/colab_resnet50_balanced*")

if exp_dirs:
    latest = max(exp_dirs, key=os.path.getctime)
    ckpt_path = f"{latest}/checkpoints/best_model.pth"
    
    if os.path.exists(ckpt_path):
        print(f"📥 Downloading: {os.path.basename(ckpt_path)}")
        print(f"   Size: {os.path.getsize(ckpt_path)/(1024**2):.1f} MB")
        files.download(ckpt_path)
    else:
        print("❌ No checkpoint found")
```

---

## 🎯 What You Get

### Automatic Class Balancing
The `WeightedRandomSampler` in `src/data_loader.py`:
- Samples cancer images ~6× more frequently
- Creates balanced batches (~50/50 cancer/non-cancer)
- No manual data duplication needed

**Training output:**
```
Using WeightedRandomSampler for balanced training
Class weights: Cancer=6.6667, No Cancer=1.1111
```

### Expected Performance
- **Excellent:** AUC > 0.90, Recall > 90%
- **Good:** AUC 0.80-0.90, Recall > 85%
- **Moderate:** AUC 0.70-0.80
- **Poor:** AUC < 0.70 (needs more data)

---

## 🔧 Troubleshooting

### "ai.zip not found"
- Check: `My Drive/AI_Cancer_Detection/ai.zip` exists
- Re-upload if needed

### "No GPU"
- Runtime → Change runtime type → T4 GPU → Save
- Restart runtime

### "Out of memory"
- Cell 6: Change `batch_size: 32` to `16`
- Restart runtime and continue from Cell 1

### "Session disconnected"
- Free tier: 12-hour limit
- Training auto-saves every epoch
- Re-run cells to resume

### "Numpy error"
- You skipped Cell 1 (dependencies)
- Restart runtime
- Run Cell 1 FIRST, then continue

---

## 📥 Use Trained Model Locally

1. **Download from Drive:**
   - Google Drive → `AI_Cancer_Detection/trained_models/`
   - Download `best_model_colab.pth`

2. **Place in project:**
   ```
   ai/experiments/colab_resnet50_balanced_YYYYMMDD/checkpoints/best_model.pth
   ```

3. **Update backend `.env`:**
   ```env
   CHECKPOINT_PATH=../ai/experiments/colab_resnet50_balanced_YYYYMMDD/checkpoints/best_model.pth
   ```

4. **Restart backend:**
   ```powershell
   cd backend
   node server.js
   ```

---

## ⚡ Performance

| Device | Batch Size | Time/Epoch | Total (50 epochs) |
|--------|------------|------------|-------------------|
| CPU | 16 | ~45 min | ~37 hours |
| Colab T4 GPU | 32 | ~3 min | **~2.5 hours** |

---

## ✅ Success Checklist

Before training:
- [ ] GPU enabled (Cell 2 shows GPU name)
- [ ] ai.zip uploaded to Drive
- [ ] All cells run without errors up to Cell 6

After training:
- [ ] Best checkpoint exists
- [ ] AUC > 0.70 (preferably > 0.85)
- [ ] Model saved to Google Drive
- [ ] Checkpoint downloaded locally

---

**Happy Training! 🚀**
