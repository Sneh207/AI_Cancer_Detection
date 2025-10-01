# 🚀 Complete Setup and Run Guide

## Step-by-Step Installation and Execution

### ✅ Prerequisites Check

Before starting, ensure you have:
- [ ] Python 3.8+ installed (`python --version`)
- [ ] Node.js 14+ installed (`node --version`)
- [ ] npm 6+ installed (`npm --version`)
- [ ] Git installed (`git --version`)

---

## 📦 Complete Installation (First Time Setup)

### Step 1: Python Environment Setup

```bash
# Navigate to project root
cd C:\Users\SNEH GUPTA\OneDrive\Desktop\cancer_detection

# Create Python virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Install AI dependencies
cd ai
pip install --upgrade pip
pip install -r requirements.txt
cd ..
```

### Step 2: Backend Setup

```bash
# Install backend dependencies
cd backend
npm install
cd ..
```

### Step 3: Frontend Setup

```bash
# Install frontend dependencies
cd frontend
npm install
cd ..
```

### Step 4: Environment Configuration

```bash
# Backend environment
cd backend
copy .env.example .env
cd ..

# Frontend environment
cd frontend
copy .env.example .env
cd ..
```

---

## 🎯 Running the Complete System

### Method 1: Automated Startup (Recommended for Windows)

```bash
# Double-click or run:
start-dev.bat
```

This will:
1. Start the backend server on http://localhost:3000
2. Start the frontend dev server on http://localhost:5173
3. Open two terminal windows for monitoring

### Method 2: Manual Startup (Cross-Platform)

**Terminal 1 - Backend Server:**
```bash
cd backend
npm start
```
Wait for: `🚀 Cancer Detection API Server running on http://localhost:3000`

**Terminal 2 - Frontend Server:**
```bash
cd frontend
npm run dev
```
Wait for: `Local: http://localhost:5173/`

**Terminal 3 - Python/AI (when needed):**
```bash
cd ai
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

---

## 🧪 Testing the System

### 1. Test the AI Pipeline

```bash
cd ai
.venv\Scripts\activate
python test_project.py
```

**Expected Output:**
```
✓ Configuration loaded
✓ Data loaders created successfully
✓ Model created
✓ Forward pass successful
✓ Training completed successfully
✓ Inference successful
✓ ALL TESTS PASSED!
```

### 2. Test the Backend API

Open browser or use curl:
```bash
# Health check
curl http://localhost:3000/health

# Status check
curl http://localhost:3000/status
```

### 3. Test the Frontend

1. Open http://localhost:5173 in your browser
2. You should see the "Cancer Detection AI" interface
3. Try uploading a test image

---

## 🎓 Training Your Own Model

### Quick Training Test (2 epochs)

```bash
cd ai
.venv\Scripts\activate
python main.py train --config configs/test_config.yaml --experiment-name quick_test --device cpu
```

### Full Production Training

```bash
cd ai
.venv\Scripts\activate
python main.py train --config configs/config.yaml --experiment-name production_model --device auto
```

**Training will:**
- Use your 3,578 chest X-ray images
- Train for configured epochs (default: 50)
- Save checkpoints to `experiments/production_model/checkpoints/`
- Log metrics to `experiments/production_model/logs/`
- Save best model automatically

**Monitor Training:**
```bash
# In another terminal
cd ai
tensorboard --logdir experiments/logs
# Open http://localhost:6006
```

---

## 🔄 Running Inference

### Via Web Interface (Easiest)

1. Ensure backend and frontend are running
2. Go to http://localhost:5173
3. Upload an X-ray image
4. Click "Analyze X-ray"
5. View results

### Via Command Line

```bash
cd ai
.venv\Scripts\activate

# Single image inference
python main.py inference \
  --config configs/config.yaml \
  --checkpoint experiments/real_data_training_20250928_234215/checkpoints/best_model.pth \
  --image path/to/your/xray.png \
  --device auto
```

### Via API (curl)

```bash
curl -X POST http://localhost:3000/predict \
  -F "image=@path/to/your/xray.png"
```

---

## 📊 Evaluating Model Performance

```bash
cd ai
.venv\Scripts\activate

python main.py evaluate \
  --config configs/config.yaml \
  --checkpoint experiments/real_data_training_20250928_234215/checkpoints/best_model.pth
```

**Output includes:**
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC curve
- Confusion matrix
- Per-class metrics
- Saved visualizations in `experiments/results/`

---

## 🛠️ Common Commands Reference

### Python/AI Commands

```bash
# Activate environment
cd ai
.venv\Scripts\activate

# Run tests
python test_project.py

# Train model
python main.py train --config configs/config.yaml --experiment-name my_model

# Evaluate model
python main.py evaluate --config configs/config.yaml --checkpoint path/to/checkpoint.pth

# Inference
python main.py inference --config configs/config.yaml --checkpoint path/to/checkpoint.pth --image path/to/image.png

# Deactivate environment
deactivate
```

### Backend Commands

```bash
cd backend

# Start server
npm start

# Start with auto-reload (development)
npm run dev

# Check for issues
npm test
```

### Frontend Commands

```bash
cd frontend

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

---

## 🐛 Troubleshooting

### Issue: "Module not found" errors

**Solution:**
```bash
cd ai
.venv\Scripts\activate
pip install -r requirements.txt
```

### Issue: "Port already in use"

**Backend (port 3000):**
```bash
# Windows: Find and kill process
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# Or change port in backend/.env
PORT=3001
```

**Frontend (port 5173):**
```bash
# Vite will automatically try next available port
# Or specify in frontend/vite.config.js
```

### Issue: "CUDA out of memory"

**Solution:**
```bash
# Edit ai/configs/config.yaml
# Reduce batch_size from 32 to 16 or 8

# Or force CPU usage
python main.py train --config configs/config.yaml --device cpu
```

### Issue: "Cannot find checkpoint"

**Solution:**
```bash
# Check if model exists
dir ai\experiments\real_data_training_20250928_234215\checkpoints\

# If not, train a new model
cd ai
python main.py train --config configs/config.yaml --experiment-name new_model
```

### Issue: Frontend can't connect to backend

**Solution:**
1. Verify backend is running: http://localhost:3000/health
2. Check frontend/.env has correct API_URL
3. Check browser console for CORS errors
4. Restart both servers

---

## 📝 Daily Development Workflow

### Starting Work

```bash
# 1. Start backend
cd backend
npm start

# 2. Start frontend (new terminal)
cd frontend
npm run dev

# 3. Activate Python env (new terminal, if needed)
cd ai
.venv\Scripts\activate
```

### During Development

- Backend auto-reloads on file changes (if using `npm run dev`)
- Frontend hot-reloads automatically
- Python changes require manual restart

### Ending Work

- Press `Ctrl+C` in each terminal to stop servers
- Deactivate Python environment: `deactivate`

---

## 🎯 Quick Reference: Full System Start

```bash
# Terminal 1
cd C:\Users\SNEH GUPTA\OneDrive\Desktop\cancer_detection\backend
npm start

# Terminal 2
cd C:\Users\SNEH GUPTA\OneDrive\Desktop\cancer_detection\frontend
npm run dev

# Terminal 3 (optional, for AI work)
cd C:\Users\SNEH GUPTA\OneDrive\Desktop\cancer_detection\ai
.venv\Scripts\activate

# Access application at: http://localhost:5173
```

---

## ✅ Verification Checklist

After setup, verify:
- [ ] Backend responds at http://localhost:3000/health
- [ ] Frontend loads at http://localhost:5173
- [ ] Can upload and analyze an image
- [ ] Python tests pass (`python test_project.py`)
- [ ] Model checkpoint exists
- [ ] No console errors in browser

---

## 📞 Getting Help

If you encounter issues:
1. Check this troubleshooting guide
2. Review error messages carefully
3. Check the main README.md
4. Verify all prerequisites are installed
5. Ensure all dependencies are up to date

---

**Last Updated:** October 1, 2025
