# 💻 Complete Terminal Commands Reference

All commands to setup, run, and maintain the Cancer Detection System.

---

## 🚀 QUICK START (Copy-Paste Ready)

### First Time Setup

```bash
# 1. Navigate to project
cd "C:\Users\SNEH GUPTA\OneDrive\Desktop\cancer_detection"

# 2. Setup Python environment
python -m venv .venv
.venv\Scripts\activate
cd ai
pip install --upgrade pip
pip install -r requirements.txt
cd ..

# 3. Setup Backend
cd backend
npm install
copy .env.example .env
cd ..

# 4. Setup Frontend
cd frontend
npm install
copy .env.example .env
cd ..
```

### Run the System

```bash
# Option 1: Automated (Windows)
start-dev.bat

# Option 2: Manual (3 terminals)
# Terminal 1:
cd backend && npm start

# Terminal 2:
cd frontend && npm run dev

# Terminal 3:
cd ai && .venv\Scripts\activate
```

---

## 📦 INSTALLATION COMMANDS

### Python Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
cd ai
pip install -r requirements.txt

# Verify installation
pip list

# Deactivate
deactivate
```

### Backend Setup

```bash
cd backend

# Install dependencies
npm install

# Create environment file
copy .env.example .env

# Verify installation
npm list

# Check for vulnerabilities
npm audit
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Create environment file
copy .env.example .env

# Verify installation
npm list
```

---

## 🎮 RUNNING COMMANDS

### Start Backend Server

```bash
cd backend

# Production mode
npm start

# Development mode (with auto-reload)
npm run dev

# Custom port
set PORT=3001 && npm start
```

**Expected Output:**
```
🚀 Cancer Detection API Server
📍 Server running on http://localhost:3000
```

### Start Frontend Server

```bash
cd frontend

# Development mode
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

**Expected Output:**
```
VITE v5.0.0  ready in 500 ms
➜  Local:   http://localhost:5173/
```

### Activate Python Environment

```bash
cd ai

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

# Verify activation (should show .venv path)
where python
```

---

## 🧪 TESTING COMMANDS

### Test AI Pipeline

```bash
cd ai
.venv\Scripts\activate
python test_project.py
```

**Expected Output:**
```
============================================================
CANCER DETECTION PROJECT TEST
============================================================
✓ Configuration loaded
✓ Data loaders created successfully
✓ Model created
✓ Forward pass successful
✓ Training completed successfully
✓ Inference successful
✓ ALL TESTS PASSED!
```

### Test Backend API

```bash
# Health check
curl http://localhost:3000/health

# Status check
curl http://localhost:3000/status

# API info
curl http://localhost:3000/
```

### Test Frontend

```bash
# Just open in browser
start http://localhost:5173
```

---

## 🎓 TRAINING COMMANDS

### Quick Test Training (2 epochs, CPU)

```bash
cd ai
.venv\Scripts\activate

python main.py train ^
  --config configs/test_config.yaml ^
  --experiment-name quick_test ^
  --device cpu
```

### Full Production Training

```bash
cd ai
.venv\Scripts\activate

python main.py train ^
  --config configs/config.yaml ^
  --experiment-name production_model ^
  --device auto ^
  --seed 42
```

### Training with Custom Parameters

```bash
python main.py train ^
  --config configs/config.yaml ^
  --experiment-name custom_training ^
  --device cuda ^
  --seed 42
```

### Monitor Training (TensorBoard)

```bash
cd ai
.venv\Scripts\activate

tensorboard --logdir experiments/logs
# Open http://localhost:6006
```

---

## 📊 EVALUATION COMMANDS

### Evaluate Trained Model

```bash
cd ai
.venv\Scripts\activate

python main.py evaluate ^
  --config configs/config.yaml ^
  --checkpoint experiments/real_data_training_20250928_234215/checkpoints/best_model.pth
```

### Evaluate with Visualizations

```bash
python main.py evaluate ^
  --config configs/config.yaml ^
  --checkpoint experiments/real_data_training_20250928_234215/checkpoints/best_model.pth ^
  --visualize
```

---

## 🔮 INFERENCE COMMANDS

### Single Image Inference

```bash
cd ai
.venv\Scripts\activate

python main.py inference ^
  --config configs/config.yaml ^
  --checkpoint experiments/real_data_training_20250928_234215/checkpoints/best_model.pth ^
  --image path\to\xray.png ^
  --device auto
```

### Batch Inference

```bash
python main.py inference ^
  --config configs/config.yaml ^
  --checkpoint experiments/real_data_training_20250928_234215/checkpoints/best_model.pth ^
  --batch-images path\to\images\folder ^
  --output results\batch_predictions.json
```

### Inference with Grad-CAM

```bash
python main.py inference ^
  --config configs/config.yaml ^
  --checkpoint experiments/real_data_training_20250928_234215/checkpoints/best_model.pth ^
  --image path\to\xray.png ^
  --visualize ^
  --output results\prediction_with_gradcam.png
```

---

## 🌐 API TESTING COMMANDS

### Using curl

```bash
# Health check
curl http://localhost:3000/health

# Status check
curl http://localhost:3000/status

# Single prediction
curl -X POST http://localhost:3000/predict ^
  -F "image=@path\to\xray.png"

# Batch prediction
curl -X POST http://localhost:3000/predict-batch ^
  -F "images=@image1.png" ^
  -F "images=@image2.png" ^
  -F "images=@image3.png"
```

### Using PowerShell

```powershell
# Health check
Invoke-RestMethod -Uri http://localhost:3000/health

# Single prediction
$form = @{
    image = Get-Item -Path "path\to\xray.png"
}
Invoke-RestMethod -Uri http://localhost:3000/predict -Method Post -Form $form
```

---

## 🔧 MAINTENANCE COMMANDS

### Update Dependencies

```bash
# Python
cd ai
.venv\Scripts\activate
pip install --upgrade -r requirements.txt

# Backend
cd backend
npm update

# Frontend
cd frontend
npm update
```

### Check for Security Issues

```bash
# Python
cd ai
pip check

# Backend
cd backend
npm audit
npm audit fix

# Frontend
cd frontend
npm audit
npm audit fix
```

### Clean Build Artifacts

```bash
# Python cache
cd ai
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Node modules (if reinstalling)
cd backend
rmdir /s /q node_modules
npm install

cd frontend
rmdir /s /q node_modules
npm install
```

---

## 🗄️ DATA MANAGEMENT COMMANDS

### Regenerate CSV Labels

```bash
cd "C:\Users\SNEH GUPTA\OneDrive\Desktop\cancer_detection"
python create_csv.py
```

### Check Dataset

```bash
cd ai\data\raw

# Count images
dir train_data\train\*.png /s | find /c ".png"

# Check CSV
type Data_Entry_2017_v2020.csv | more
```

### Verify Data Integrity

```bash
cd ai
.venv\Scripts\activate

python -c "
from src.data_loader import DataManager
import yaml

with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

dm = DataManager(config)
image_paths, labels = dm.prepare_labels()
print(f'Total images: {len(image_paths)}')
print(f'Cancer cases: {sum(labels)}')
print(f'Normal cases: {len(labels) - sum(labels)}')
"
```

---

## 🐛 DEBUGGING COMMANDS

### Check Python Environment

```bash
cd ai
.venv\Scripts\activate

# Python version
python --version

# Installed packages
pip list

# Check specific package
pip show torch

# Python path
python -c "import sys; print('\n'.join(sys.path))"
```

### Check Node Environment

```bash
# Node version
node --version

# npm version
npm --version

# Global packages
npm list -g --depth=0

# Check specific package
npm list express
```

### Check Ports

```bash
# Check if port is in use (Windows)
netstat -ano | findstr :3000
netstat -ano | findstr :5173

# Kill process by PID
taskkill /PID <PID> /F
```

### View Logs

```bash
# Backend logs (if using PM2 or similar)
cd backend
npm start > logs.txt 2>&1

# AI training logs
cd ai\experiments\<experiment_name>\logs
type training.log
```

---

## 📈 MONITORING COMMANDS

### System Resources

```bash
# CPU and Memory usage
tasklist | findstr "node.exe python.exe"

# Detailed info
wmic process where name="python.exe" get ProcessId,WorkingSetSize,CommandLine
```

### Model Performance

```bash
cd ai
.venv\Scripts\activate

# Quick evaluation
python main.py evaluate ^
  --config configs/config.yaml ^
  --checkpoint experiments/real_data_training_20250928_234215/checkpoints/best_model.pth
```

---

## 🔄 GIT COMMANDS (If Using Version Control)

```bash
# Initialize repo (if not done)
git init

# Check status
git status

# Add files
git add .

# Commit
git commit -m "Optimized cancer detection system"

# Create .gitignore (already exists)
# View ignored files
git status --ignored

# Push to remote
git remote add origin <repository-url>
git push -u origin main
```

---

## 🎯 COMPLETE WORKFLOW EXAMPLES

### Daily Development Workflow

```bash
# 1. Start work
cd "C:\Users\SNEH GUPTA\OneDrive\Desktop\cancer_detection"

# 2. Update code (if needed)
git pull

# 3. Start servers
start-dev.bat

# 4. Work on AI (if needed)
cd ai
.venv\Scripts\activate
# ... do work ...
deactivate

# 5. End of day
# Press Ctrl+C in each terminal
git add .
git commit -m "Daily progress"
git push
```

### Training a New Model Workflow

```bash
# 1. Activate environment
cd ai
.venv\Scripts\activate

# 2. Test configuration
python test_project.py

# 3. Start training
python main.py train ^
  --config configs/config.yaml ^
  --experiment-name new_model_v1 ^
  --device auto

# 4. Monitor (in another terminal)
tensorboard --logdir experiments/logs

# 5. After training, evaluate
python main.py evaluate ^
  --config configs/config.yaml ^
  --checkpoint experiments/new_model_v1/checkpoints/best_model.pth

# 6. Update backend to use new model
# Edit backend/.env:
# CHECKPOINT_PATH=../ai/experiments/new_model_v1/checkpoints/best_model.pth

# 7. Restart backend
cd ..\backend
npm start
```

### Deploying to Production Workflow

```bash
# 1. Build frontend
cd frontend
npm run build

# 2. Test production build
npm run preview

# 3. Set environment to production
cd ..\backend
# Edit .env: NODE_ENV=production

# 4. Start backend in production mode
npm start

# 5. Serve frontend (use nginx or similar)
# Configure nginx to serve frontend/dist
```

---

## 📝 NOTES

- Always activate Python virtual environment before running AI commands
- Use `^` for line continuation in Windows CMD
- Use `\` for line continuation in PowerShell/Linux
- Replace `<PID>` with actual process ID
- Replace `<experiment_name>` with your experiment name
- Replace `path\to\` with actual file paths

---

## ✅ VERIFICATION COMMANDS

Run these to verify everything is working:

```bash
# 1. Check Python
cd ai
.venv\Scripts\activate
python --version
pip list | findstr torch

# 2. Check Backend
cd ..\backend
npm list | findstr express
curl http://localhost:3000/health

# 3. Check Frontend
cd ..\frontend
npm list | findstr react
start http://localhost:5173

# 4. Test AI
cd ..\ai
python test_project.py

# 5. Test API
curl -X POST http://localhost:3000/predict -F "image=@test_image.png"
```

---

**Last Updated:** October 1, 2025  
**For:** Cancer Detection System v1.0.0
