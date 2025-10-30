# AI Lung Cancer Detection - Model Setup Guide

## 🚨 Model Checkpoint Not Found Error - SOLUTION

This guide will help you resolve the "Model checkpoint not found" error and get your AI model running.

---

## 📋 Quick Fix Options

### **Option 1: Train a New Model (Recommended for Learning)**

If you have the training dataset, you can train your own model:

```bash
# Navigate to the AI directory
cd ai

# Activate your virtual environment (if using one)
# On Windows:
.venv\Scripts\activate

# Train the model
python main.py train --config configs/config.yaml
```

**Training will:**
- Create the checkpoint directory automatically
- Save the best model to `experiments/[timestamp]/checkpoints/best_model.pth`
- Generate training logs and metrics

**Note:** After training, update the checkpoint path in `backend/server.js` to point to your new model.

---

### **Option 2: Use a Pre-trained Model**

If you have a pre-trained model checkpoint file (`.pth` or `.pt`):

1. **Place the checkpoint file** in the expected directory:
   ```
   AI_Cancer_Detection/
   └── ai/
       └── experiments/
           └── test_run_fixed_20250928_224821/
               └── checkpoints/
                   └── best_model.pth  ← Place your model here
   ```

2. **Or create a new experiment folder:**
   ```bash
   # Create directory
   mkdir -p ai/experiments/my_model/checkpoints
   
   # Copy your model
   cp path/to/your/model.pth ai/experiments/my_model/checkpoints/best_model.pth
   ```

3. **Update the backend server** to point to your model location:
   - Edit `backend/server.js`
   - Update lines 138-139 and 202-203 with your checkpoint path

---

### **Option 3: Set Environment Variable**

Set the `CHECKPOINT_PATH` environment variable to point to your model:

**On Windows (PowerShell):**
```powershell
$env:CHECKPOINT_PATH = "C:\path\to\your\model.pth"
node backend/server.js
```

**On Windows (Command Prompt):**
```cmd
set CHECKPOINT_PATH=C:\path\to\your\model.pth
node backend/server.js
```

**On Linux/Mac:**
```bash
export CHECKPOINT_PATH=/path/to/your/model.pth
node backend/server.js
```

---

## 📁 Expected Directory Structure

```
AI_Cancer_Detection/
├── ai/
│   ├── configs/
│   │   └── config.yaml
│   ├── experiments/
│   │   └── test_run_fixed_20250928_224821/  (or your experiment name)
│   │       ├── checkpoints/
│   │       │   ├── best_model.pth          ← Your trained model
│   │       │   └── checkpoint.pth          ← Latest checkpoint
│   │       ├── logs/
│   │       │   └── training.log
│   │       └── results/
│   │           └── metrics.json
│   ├── data/
│   │   └── raw/
│   │       └── train_data/                  ← Your training images
│   ├── src/
│   └── main.py
├── backend/
│   └── server.js
└── frontend/
```

---

## 🔧 Training Requirements

Before training, ensure you have:

1. **Dataset**: Chest X-ray images in `ai/data/raw/train_data/`
2. **Labels**: CSV file with image labels
3. **Dependencies**: All Python packages installed
   ```bash
   pip install -r ai/requirements.txt
   ```

---

## 🎯 Model Configuration

The model configuration is in `ai/configs/config.yaml`:

```yaml
# Model parameters
model:
  architecture: "custom_cnn"  # Options: resnet50, densenet121, custom_cnn
  pretrained: true
  num_classes: 1
  dropout: 0.5

# Training parameters
training:
  epochs: 100
  learning_rate: 0.001
  batch_size: 32
  
# Paths
paths:
  checkpoints: "experiments/checkpoints"
  logs: "experiments/logs"
```

---

## 🧪 Testing Your Setup

After setting up the model, test it:

### **1. Check Model Status**
```bash
curl http://localhost:3000/status
```

Expected response:
```json
{
  "status": "running",
  "modelAvailable": true,
  "checkpointPath": "path/to/your/model.pth"
}
```

### **2. Run Inference on a Test Image**
```bash
cd ai
python main.py inference --checkpoint path/to/model.pth --image path/to/test.png
```

### **3. Test via API**
```bash
curl -X POST -F "image=@test_image.png" http://localhost:3000/predict
```

---

## 📊 Model Information

**Architecture Options:**
- `custom_cnn`: Custom CNN architecture for lung cancer detection
- `resnet50`: ResNet-50 with transfer learning
- `densenet121`: DenseNet-121 with transfer learning

**Input Requirements:**
- Image format: PNG, JPG, JPEG
- Image size: 224x224 pixels (auto-resized)
- Color: RGB or Grayscale

**Output:**
- Prediction: "Cancer" or "No Cancer"
- Probability: 0.0 to 1.0
- Confidence: Percentage (0-100%)

---

## ❓ Troubleshooting

### **Error: "Checkpoint not found"**
- ✅ Verify the checkpoint file exists at the specified path
- ✅ Check file permissions
- ✅ Ensure the path in `server.js` matches your actual file location

### **Error: "CUDA out of memory"**
- Reduce batch size in `config.yaml`
- Use CPU instead: `--device cpu`

### **Error: "Module not found"**
- Install dependencies: `pip install -r requirements.txt`
- Activate virtual environment

### **Poor Model Performance**
- Train for more epochs
- Adjust learning rate
- Use data augmentation
- Try different architectures

---

## 📞 Support

For additional help:
1. Check the main README.md
2. Review training logs in `experiments/logs/`
3. Verify your dataset is properly formatted

---

## 🎓 Quick Start (Complete Workflow)

```bash
# 1. Install dependencies
cd ai
pip install -r requirements.txt

# 2. Prepare your dataset
# Place images in ai/data/raw/train_data/

# 3. Train the model
python main.py train --config configs/config.yaml

# 4. Note the experiment directory created (e.g., experiment_20250119_001234)

# 5. Update backend/server.js with the new checkpoint path

# 6. Start the backend server
cd ../backend
npm install
node server.js

# 7. Test the API
curl http://localhost:3000/status
```

---

**Last Updated:** January 19, 2025  
**Version:** 1.0.0
