# 🎓 Complete Training Guide - Real Model with Training Data

## 📊 Overview

To train a real lung cancer detection model, you need:
1. ✅ Chest X-ray images (training data)
2. ✅ Labels/annotations (which images have cancer)
3. ✅ Python environment with dependencies
4. ✅ Sufficient compute resources (GPU recommended)

---

## 📥 Step 1: Download Training Data

### **Option A: NIH Chest X-ray Dataset** (Recommended)

This is the most comprehensive public dataset for chest X-rays.

#### **Dataset Information:**
- **Name**: NIH Chest X-ray14 Dataset
- **Images**: 112,120 frontal-view X-ray images
- **Patients**: 30,805 unique patients
- **Size**: ~42 GB
- **Labels**: 14 disease categories including lung cancer indicators
- **Format**: PNG images + CSV labels

#### **Download Steps:**

**Method 1: Kaggle (Easiest)**

1. **Create Kaggle Account**:
   - Go to https://www.kaggle.com
   - Sign up for free account

2. **Download Dataset**:
   - Visit: https://www.kaggle.com/datasets/nih-chest-xrays/data
   - Click "Download" button
   - Or use Kaggle API:
   ```bash
   # Install Kaggle CLI
   pip install kaggle
   
   # Setup API credentials (get from kaggle.com/account)
   # Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\Users\<username>\.kaggle\ (Windows)
   
   # Download dataset
   kaggle datasets download -d nih-chest-xrays/data
   
   # Extract
   unzip data.zip -d ai/data/raw/
   ```

**Method 2: Direct Download from NIH**

1. Visit: https://nihcc.app.box.com/v/ChestXray-NIHCC
2. Download all image archives (images_001.tar.gz to images_012.tar.gz)
3. Download Data_Entry_2017_v2020.csv

#### **Extract and Organize:**

```bash
# Navigate to project
cd AI_Cancer_Detection

# Create data directory
mkdir -p ai/data/raw/train_data

# Extract all archives to train_data
# Windows (using 7-Zip):
7z x images_001.tar.gz -o"ai\data\raw\train_data"
7z x images_002.tar.gz -o"ai\data\raw\train_data"
# ... repeat for all archives

# Linux/Mac:
tar -xzf images_001.tar.gz -C ai/data/raw/train_data/
tar -xzf images_002.tar.gz -C ai/data/raw/train_data/
# ... repeat for all archives

# Copy labels file
copy Data_Entry_2017_v2020.csv ai\data\raw\
```

---

### **Option B: Smaller Datasets (For Quick Testing)**

#### **1. COVID-19 Radiography Database**
- **Size**: ~3 GB (21,165 images)
- **Link**: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
- **Good for**: Quick testing, smaller compute requirements

#### **2. ChestX-ray8 (Subset)**
- **Size**: ~10 GB (subset of NIH dataset)
- **Link**: https://nihcc.app.box.com/v/ChestXray-NIHCC
- **Good for**: Medium-scale training

---

## 📁 Step 2: Organize Your Data

After downloading, your directory structure should look like:

```
AI_Cancer_Detection/
├── ai/
│   ├── data/
│   │   └── raw/
│   │       ├── train_data/
│   │       │   ├── 00000001_000.png
│   │       │   ├── 00000002_000.png
│   │       │   ├── 00000003_001.png
│   │       │   └── ... (all X-ray images)
│   │       └── Data_Entry_2017_v2020.csv  ← Labels file
│   ├── configs/
│   │   └── config.yaml
│   └── main.py
```

### **Verify Data:**

```bash
cd ai
python -c "import os; print(f'Images found: {len(os.listdir(\"data/raw/train_data\"))}')"
```

---

## ⚙️ Step 3: Configure Training

### **Edit Configuration File:**

Open `ai/configs/config.yaml` and adjust settings:

```yaml
# Data configuration
data:
  dataset_path: "data/raw/train_data"
  labels_file: "data/raw/Data_Entry_2017_v2020.csv"
  image_size: 224
  batch_size: 32          # Reduce to 16 or 8 if low on GPU memory
  num_workers: 4          # Adjust based on CPU cores
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15

# Training parameters
training:
  epochs: 50              # Start with 50, increase if needed
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: "adam"
  scheduler: "cosine"
  early_stopping_patience: 10
  save_best_only: true

# Model parameters
model:
  architecture: "resnet50"  # Options: resnet50, densenet121, custom_cnn
  pretrained: true          # Use ImageNet pretrained weights
  num_classes: 1
  dropout: 0.5
```

### **Quick Test Configuration:**

For faster initial testing, use `config_quick_test.yaml`:

```yaml
training:
  epochs: 5               # Just 5 epochs for quick test
  batch_size: 16
```

---

## 🚀 Step 4: Train the Model

### **Basic Training:**

```bash
cd ai

# Full training
python main.py train --config configs/config.yaml

# Quick test (5 epochs)
python main.py train --config configs/config_quick_test.yaml
```

### **Advanced Training Options:**

```bash
# Specify device
python main.py train --config configs/config.yaml --device cuda

# Use CPU only
python main.py train --config configs/config.yaml --device cpu

# Resume from checkpoint
python main.py train --config configs/config.yaml --checkpoint experiments/[name]/checkpoints/checkpoint.pth
```

---

## 📊 Step 5: Monitor Training

### **Training Output:**

You'll see output like:

```
============================================================
STARTING TRAINING
============================================================
Device: cuda
Model: ResNet50
Total parameters: 23,528,522
Trainable parameters: 23,528,522

Epoch 1/50
----------
Train Loss: 0.6234 | Train Acc: 65.23% | Train AUC: 0.7123
Val Loss: 0.5891 | Val Acc: 68.45% | Val AUC: 0.7456
Best model saved!

Epoch 2/50
----------
Train Loss: 0.5678 | Train Acc: 70.12% | Train AUC: 0.7689
Val Loss: 0.5234 | Val Acc: 72.34% | Val AUC: 0.7891
Best model saved!
...
```

### **Check Training Logs:**

```bash
# View logs
cat ai/experiments/[experiment_name]/logs/training.log

# Monitor in real-time
tail -f ai/experiments/[experiment_name]/logs/training.log
```

### **Training Artifacts:**

After training, you'll find:

```
ai/experiments/[experiment_name]/
├── checkpoints/
│   ├── best_model.pth      ← Best performing model
│   └── checkpoint.pth      ← Latest checkpoint
├── logs/
│   └── training.log        ← Training logs
└── results/
    ├── training_history.png
    ├── confusion_matrix.png
    └── metrics.json
```

---

## ⏱️ Training Time Estimates

### **With GPU (NVIDIA RTX 3060 or better):**
- Quick test (5 epochs): 10-30 minutes
- Full training (50 epochs): 2-4 hours
- Extended training (100 epochs): 4-8 hours

### **With CPU Only:**
- Quick test (5 epochs): 1-2 hours
- Full training (50 epochs): 10-20 hours
- Extended training (100 epochs): 20-40 hours

### **Recommendations:**
- Use GPU if available
- Start with quick test (5 epochs) to verify everything works
- Then run full training overnight

---

## 🎯 Step 6: Use Your Trained Model

### **Update Backend Configuration:**

After training completes, note the experiment directory name (e.g., `experiment_20250119_123456`).

**Option 1: Update server.js**

Edit `backend/server.js` line 139:

```javascript
const checkpointPath = process.env.CHECKPOINT_PATH || 
  path.join(__dirname, '..', 'ai', 'experiments', 
    'experiment_20250119_123456', 'checkpoints', 'best_model.pth');
```

**Option 2: Use Environment Variable**

```powershell
$env:CHECKPOINT_PATH = "C:\...\AI_Cancer_Detection\ai\experiments\experiment_20250119_123456\checkpoints\best_model.pth"
cd backend
node server.js
```

### **Test Your Model:**

```bash
# Test inference
cd ai
python main.py inference \
  --checkpoint experiments/[name]/checkpoints/best_model.pth \
  --image path/to/test_xray.png

# Evaluate on test set
python main.py evaluate \
  --checkpoint experiments/[name]/checkpoints/best_model.pth \
  --config configs/config.yaml
```

---

## 🔧 Troubleshooting

### **Issue: Out of Memory (OOM)**

```yaml
# Reduce batch size in config.yaml
data:
  batch_size: 8  # or even 4

# Or use CPU
python main.py train --config configs/config.yaml --device cpu
```

### **Issue: Training Too Slow**

```yaml
# Reduce image size
data:
  image_size: 128  # instead of 224

# Reduce epochs for testing
training:
  epochs: 10
```

### **Issue: Poor Performance**

```yaml
# Try different architecture
model:
  architecture: "densenet121"  # or "resnet50"

# Increase epochs
training:
  epochs: 100

# Adjust learning rate
training:
  learning_rate: 0.0001  # smaller = more stable
```

### **Issue: Dataset Not Found**

```bash
# Verify paths
ls ai/data/raw/train_data/
ls ai/data/raw/Data_Entry_2017_v2020.csv

# Check config.yaml paths match actual file locations
```

---

## 📈 Expected Performance

### **Good Model Metrics:**
- **Accuracy**: 75-85%
- **AUC-ROC**: 0.80-0.90
- **Precision**: 70-80%
- **Recall**: 75-85%

### **Excellent Model Metrics:**
- **Accuracy**: 85-92%
- **AUC-ROC**: 0.90-0.95
- **Precision**: 80-90%
- **Recall**: 85-92%

---

## 🎓 Training Tips

### **1. Start Small:**
```bash
# Test with 5 epochs first
python main.py train --config configs/config_quick_test.yaml
```

### **2. Use Pretrained Weights:**
```yaml
model:
  pretrained: true  # Always use this for better results
```

### **3. Monitor Validation Loss:**
- If validation loss stops decreasing, training is done
- Early stopping will handle this automatically

### **4. Save Checkpoints:**
```yaml
training:
  save_best_only: true  # Saves only the best model
```

### **5. Use Data Augmentation:**
```yaml
augmentation:
  horizontal_flip: 0.5
  rotation_range: 15
  brightness: 0.2
  contrast: 0.2
```

---

## 📚 Alternative: Use Google Colab (Free GPU)

If you don't have a GPU:

### **1. Upload to Google Drive:**
- Upload your code to Google Drive
- Upload dataset (or download in Colab)

### **2. Create Colab Notebook:**

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Navigate to project
%cd /content/drive/MyDrive/AI_Cancer_Detection/ai

# Install dependencies
!pip install -r requirements.txt

# Train model
!python main.py train --config configs/config.yaml --device cuda
```

### **3. Download Trained Model:**
- After training, download `best_model.pth`
- Place in your local project
- Update backend configuration

---

## 🎯 Complete Training Workflow

```bash
# 1. Download dataset
kaggle datasets download -d nih-chest-xrays/data

# 2. Extract to correct location
unzip data.zip -d ai/data/raw/

# 3. Verify data
python check_setup.py

# 4. Quick test training
cd ai
python main.py train --config configs/config_quick_test.yaml

# 5. Full training
python main.py train --config configs/config.yaml

# 6. Evaluate model
python main.py evaluate --checkpoint experiments/[name]/checkpoints/best_model.pth

# 7. Update backend with new checkpoint path

# 8. Start servers and test
cd ../backend && node server.js
cd ../frontend && npm run dev
```

---

## 📞 Need Help?

**Common Issues:**
- Dataset download: Check Kaggle credentials
- Training errors: Check `ai/experiments/[name]/logs/training.log`
- OOM errors: Reduce batch size
- Slow training: Use GPU or reduce image size

**Resources:**
- NIH Dataset: https://nihcc.app.box.com/v/ChestXray-NIHCC
- Kaggle: https://www.kaggle.com/datasets/nih-chest-xrays/data
- PyTorch Docs: https://pytorch.org/docs/

---

**Time to Complete:** 3-6 hours (including download)  
**Difficulty:** Intermediate  
**Requirements:** 50GB+ disk space, GPU recommended  
**Result:** Production-ready lung cancer detection model
