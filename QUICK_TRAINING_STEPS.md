# ⚡ Quick Training Steps - TL;DR Version

## 🎯 Get Training Data & Train Model in 5 Steps

### **Step 1: Download Dataset (Choose One)**

#### **Option A: Kaggle (Easiest)**
```bash
# Install Kaggle CLI
pip install kaggle

# Setup credentials: Get API key from kaggle.com/account
# Place kaggle.json in C:\Users\<username>\.kaggle\

# Download NIH Chest X-ray dataset
kaggle datasets download -d nih-chest-xrays/data

# Extract
unzip data.zip -d ai/data/raw/
```

#### **Option B: Manual Download**
1. Go to: https://www.kaggle.com/datasets/nih-chest-xrays/data
2. Click "Download" (requires free Kaggle account)
3. Extract to `ai/data/raw/train_data/`
4. Ensure `Data_Entry_2017_v2020.csv` is in `ai/data/raw/`

---

### **Step 2: Verify Data**
```bash
cd ai
python check_setup.py
```

Should show:
```
✅ Training images: XXXX images found
```

---

### **Step 3: Quick Test Training (10-30 mins)**
```bash
cd ai
python main.py train --config configs/config_quick_test.yaml
```

This trains for just 5 epochs to verify everything works.

---

### **Step 4: Full Training (2-4 hours with GPU)**
```bash
cd ai
python main.py train --config configs/config.yaml
```

---

### **Step 5: Use Your Model**

After training completes, note the experiment folder name (e.g., `experiment_20250119_123456`).

Update `backend/server.js` line 139:
```javascript
const checkpointPath = process.env.CHECKPOINT_PATH || 
  path.join(__dirname, '..', 'ai', 'experiments', 
    'experiment_20250119_123456', 'checkpoints', 'best_model.pth');
```

Then restart backend:
```bash
cd backend
node server.js
```

---

## 📊 Dataset Options

| Dataset | Size | Images | Time to Download | Best For |
|---------|------|--------|------------------|----------|
| **NIH Chest X-ray14** | 42 GB | 112,120 | 30-60 mins | Production model |
| **COVID-19 Radiography** | 3 GB | 21,165 | 5-10 mins | Quick testing |
| **ChestX-ray8 (subset)** | 10 GB | ~30,000 | 15-30 mins | Medium-scale |

---

## ⏱️ Time Estimates

### **With GPU (RTX 3060+):**
- Download: 30-60 minutes
- Quick test (5 epochs): 10-30 minutes
- Full training (50 epochs): 2-4 hours

### **With CPU Only:**
- Download: 30-60 minutes
- Quick test (5 epochs): 1-2 hours
- Full training (50 epochs): 10-20 hours

---

## 🚀 Fastest Path (Using Google Colab - Free GPU)

### **1. Create Colab Notebook:**
Go to: https://colab.research.google.com

### **2. Run This Code:**
```python
# Clone your project or upload files
!git clone https://github.com/your-repo/AI_Cancer_Detection.git
%cd AI_Cancer_Detection/ai

# Install dependencies
!pip install -r requirements.txt

# Download dataset (if using Kaggle)
!pip install kaggle
# Upload your kaggle.json to Colab
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d nih-chest-xrays/data
!unzip data.zip -d data/raw/

# Train model (uses free GPU)
!python main.py train --config configs/config.yaml --device cuda

# Download trained model
from google.colab import files
files.download('experiments/[experiment_name]/checkpoints/best_model.pth')
```

### **3. Use Downloaded Model:**
Place `best_model.pth` in your local project and update backend path.

---

## 🎯 Recommended Workflow

### **For Testing/Learning:**
```bash
# 1. Download small dataset (COVID-19 Radiography - 3GB)
# 2. Quick test training (5 epochs)
python main.py train --config configs/config_quick_test.yaml
# 3. Test the model
```

### **For Production:**
```bash
# 1. Download full NIH dataset (42GB)
# 2. Full training (50-100 epochs)
python main.py train --config configs/config.yaml
# 3. Evaluate thoroughly
python main.py evaluate --checkpoint experiments/[name]/checkpoints/best_model.pth
```

---

## 🔧 Common Issues & Fixes

### **"Out of Memory" Error:**
```yaml
# Edit config.yaml
data:
  batch_size: 8  # Reduce from 32
```

### **"Dataset not found":**
```bash
# Check paths
ls ai/data/raw/train_data/
ls ai/data/raw/Data_Entry_2017_v2020.csv
```

### **Training too slow:**
```bash
# Use GPU
python main.py train --config configs/config.yaml --device cuda

# Or use Google Colab (free GPU)
```

---

## 📝 After Training Checklist

- [ ] Training completed without errors
- [ ] `best_model.pth` exists in experiments folder
- [ ] Model achieves reasonable accuracy (>75%)
- [ ] Updated backend checkpoint path
- [ ] Tested inference with sample image
- [ ] Backend returns predictions successfully

---

## 🎓 Next Steps After Training

1. **Evaluate Model:**
   ```bash
   python main.py evaluate --checkpoint experiments/[name]/checkpoints/best_model.pth
   ```

2. **Test Inference:**
   ```bash
   python main.py inference --checkpoint experiments/[name]/checkpoints/best_model.pth --image test.png
   ```

3. **Update Backend:**
   - Edit `backend/server.js` with new checkpoint path
   - Restart backend server

4. **Test Full Application:**
   - Start backend and frontend
   - Upload X-ray image
   - Verify predictions are working

---

**For detailed instructions, see:** `TRAINING_GUIDE.md`

**Time Required:** 3-6 hours total  
**Difficulty:** Intermediate  
**Result:** Production-ready model
