# 🚀 Quick Start Guide - AI Lung Cancer Detection

## ⚠️ Model Checkpoint Not Found - Quick Fix

You're seeing this error because the AI model hasn't been trained yet. Here's how to fix it:

---

## 🎯 Solution Options

### **Option 1: Train the Model (Recommended)**

This will create a working model checkpoint:

```bash
# 1. Navigate to the AI directory
cd AI_Cancer_Detection/ai

# 2. Activate virtual environment (if you have one)
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3. Install dependencies (if not already installed)
pip install -r requirements.txt

# 4. Train the model
python main.py train --config configs/config.yaml
```

**What happens during training:**
- ✅ Creates `experiments/[timestamp]/checkpoints/` directory
- ✅ Saves `best_model.pth` (the checkpoint file)
- ✅ Generates training logs and metrics
- ⏱️ Takes 1-3 hours depending on your hardware and dataset size

**After training:**
1. Note the experiment directory name (e.g., `experiment_20250119_123456`)
2. Update `backend/server.js` line 139 with your new checkpoint path
3. Restart the backend server

---

### **Option 2: Use Environment Variable (Quick Test)**

If you already have a trained model file:

**Windows PowerShell:**
```powershell
# Set the checkpoint path
$env:CHECKPOINT_PATH = "C:\full\path\to\your\model.pth"

# Start the backend
cd backend
node server.js
```

**Windows CMD:**
```cmd
set CHECKPOINT_PATH=C:\full\path\to\your\model.pth
cd backend
node server.js
```

---

### **Option 3: Download Pre-trained Model**

If you have access to a pre-trained model:

1. **Download** the `.pth` model file
2. **Create directory:**
   ```bash
   mkdir -p ai/experiments/pretrained/checkpoints
   ```
3. **Place model:**
   ```bash
   # Copy your downloaded model
   cp downloaded_model.pth ai/experiments/pretrained/checkpoints/best_model.pth
   ```
4. **Update backend/server.js** (line 139):
   ```javascript
   const checkpointPath = process.env.CHECKPOINT_PATH || 
     path.join(__dirname, '..', 'ai', 'experiments', 'pretrained', 'checkpoints', 'best_model.pth');
   ```

---

## 📊 Check Current Status

Test if the model is available:

```bash
# Start the backend server
cd backend
node server.js

# In another terminal, check status:
curl http://localhost:3000/status
```

**Expected Response (Model Available):**
```json
{
  "status": "running",
  "modelAvailable": true,
  "checkpointPath": "path/to/model.pth"
}
```

**Expected Response (Model NOT Available):**
```json
{
  "status": "running",
  "modelAvailable": false,
  "checkpointPath": "path/to/model.pth"
}
```

---

## 🗂️ Required Directory Structure

```
AI_Cancer_Detection/
├── ai/
│   ├── configs/
│   │   └── config.yaml                     ✅ Exists
│   ├── experiments/
│   │   └── [your_experiment_name]/
│   │       └── checkpoints/
│   │           └── best_model.pth          ❌ MISSING - Need to create
│   ├── data/
│   │   └── raw/
│   │       └── train_data/                 ⚠️ Need training images
│   └── main.py                             ✅ Exists
├── backend/
│   └── server.js                           ✅ Exists
└── frontend/
```

---

## 📝 Training Configuration

Before training, check `ai/configs/config.yaml`:

```yaml
# Data paths
data:
  dataset_path: "data/raw/train_data"       # Your X-ray images folder
  labels_file: "data/raw/Data_Entry_2017_v2020.csv"  # Labels CSV
  batch_size: 32
  
# Training
training:
  epochs: 100                                # Reduce to 10-20 for quick test
  learning_rate: 0.001
  
# Model
model:
  architecture: "custom_cnn"                 # or "resnet50", "densenet121"
  pretrained: true
```

**For Quick Testing:** Reduce epochs to 10-20 in the config file.

---

## 🧪 Test the Complete System

After setting up the model:

### **1. Start Backend**
```bash
cd backend
node server.js
```

### **2. Start Frontend** (in new terminal)
```bash
cd frontend
npm start
```

### **3. Test Prediction**
Upload a chest X-ray image through the web interface or use curl:
```bash
curl -X POST -F "image=@test_xray.png" http://localhost:3000/predict
```

---

## ⚡ Quick Training Tips

### **Minimal Training (For Testing)**
Edit `ai/configs/config.yaml`:
```yaml
training:
  epochs: 5                    # Just 5 epochs for quick test
  batch_size: 16               # Smaller batch if low memory
```

Then train:
```bash
cd ai
python main.py train --config configs/config.yaml
```

### **Full Training (For Production)**
Use default config (100 epochs) for best results.

---

## 🔍 Troubleshooting

### **"No module named 'torch'"**
```bash
pip install torch torchvision
```

### **"Dataset not found"**
- Check `ai/data/raw/train_data/` has images
- Verify `labels_file` path in config.yaml

### **"CUDA out of memory"**
- Reduce `batch_size` in config.yaml
- Or use CPU: `python main.py train --device cpu`

### **"Permission denied"**
- Run terminal as administrator (Windows)
- Or use `sudo` (Linux/Mac)

---

## 📚 Additional Resources

- **Full Setup Guide:** See `MODEL_SETUP.md`
- **Training Logs:** Check `ai/experiments/[name]/logs/`
- **Model Metrics:** Check `ai/experiments/[name]/results/`

---

## 🎓 Recommended Workflow

```bash
# 1. Install everything
cd ai
pip install -r requirements.txt

# 2. Quick test training (5 epochs)
# Edit config.yaml: set epochs to 5
python main.py train --config configs/config.yaml

# 3. Check the created experiment folder
ls experiments/

# 4. Update backend with new checkpoint path
# Edit backend/server.js line 139

# 5. Start backend
cd ../backend
node server.js

# 6. Test status
curl http://localhost:3000/status

# 7. Start frontend
cd ../frontend
npm start
```

---

**Need Help?** Check `MODEL_SETUP.md` for detailed instructions.

**Last Updated:** January 19, 2025
