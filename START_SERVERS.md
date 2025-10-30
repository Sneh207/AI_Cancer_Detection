# 🚀 How to Start the Application

## ⚠️ IMPORTANT: Port Configuration

- **Backend**: Runs on port **5000** (http://localhost:5000)
- **Frontend**: Runs on port **3000** (http://localhost:3000)

---

## 📝 Quick Start Steps

### **Step 1: Create Frontend .env File**

In the `frontend` directory, create a `.env` file:

```bash
cd frontend
echo VITE_API_URL=http://localhost:5000 > .env
```

Or manually create `frontend/.env` with this content:
```
VITE_API_URL=http://localhost:5000
```

---

### **Step 2: Start Backend Server**

Open **Terminal 1** (PowerShell):

```powershell
cd AI_Cancer_Detection\backend
node server.js
```

You should see:
```
🚀 Cancer Detection API Server
📍 Server running on http://localhost:5000
```

---

### **Step 3: Start Frontend**

Open **Terminal 2** (PowerShell):

```powershell
cd AI_Cancer_Detection\frontend
npm run dev
```

You should see:
```
VITE ready in XXX ms
➜  Local:   http://localhost:3000/
```

---

### **Step 4: Verify Backend Connection**

Open a browser or use curl:

```bash
# Check backend status
curl http://localhost:5000/status

# Or open in browser
http://localhost:5000/status
```

Expected response:
```json
{
  "status": "running",
  "modelAvailable": false,  ← Will be false until you train/add model
  "checkpointPath": "..."
}
```

---

## 🔧 If You Get "Model checkpoint not found" Error

This is **EXPECTED** until you train or add a model. You have 3 options:

### **Option 1: Train a Model** (Recommended)

```bash
cd ai
python main.py train --config configs/config.yaml
```

This will create: `experiments/[timestamp]/checkpoints/best_model.pth`

### **Option 2: Use Environment Variable**

```powershell
$env:CHECKPOINT_PATH = "C:\path\to\your\model.pth"
cd backend
node server.js
```

### **Option 3: Place Pre-trained Model**

Place your `.pth` file at:
```
ai/experiments/test_run_fixed_20250928_224821/checkpoints/best_model.pth
```

---

## 🐛 Troubleshooting

### **"Port 3000 already in use"**
- Something else is using port 3000
- Kill the process: `Get-Process -Id (Get-NetTCPConnection -LocalPort 3000).OwningProcess | Stop-Process`
- Or change frontend port in `vite.config.js`

### **"Port 5000 already in use"**
- Kill the process: `Get-Process -Id (Get-NetTCPConnection -LocalPort 5000).OwningProcess | Stop-Process`
- Or change backend port in `server.js` line 9

### **Frontend shows "Model checkpoint not found"**
- ✅ This is NORMAL - you need to train/add a model first
- Backend is running correctly
- See "Option 1, 2, or 3" above to add the model

### **"Cannot connect to backend"**
- Verify backend is running on port 5000
- Check `frontend/.env` has `VITE_API_URL=http://localhost:5000`
- Restart frontend after creating `.env` file

---

## ✅ Complete Startup Checklist

- [ ] Backend running on port 5000
- [ ] Frontend running on port 3000
- [ ] Frontend `.env` file created with correct API URL
- [ ] Backend status endpoint returns `"status": "running"`
- [ ] Model checkpoint trained or placed (optional for testing connection)

---

## 🎯 Full Working Setup

Once you have a trained model:

1. ✅ Backend running: `http://localhost:5000`
2. ✅ Frontend running: `http://localhost:3000`
3. ✅ Model checkpoint exists
4. ✅ Upload X-ray image and get predictions!

---

**Need Help?** See:
- `QUICK_START.md` - Quick setup guide
- `MODEL_SETUP.md` - Model training guide
- `CHECKPOINT_SETUP_COMPLETE.md` - Checkpoint setup guide
