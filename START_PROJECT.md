# 🚀 Cancer Detection Project - Quick Start Guide

## ✅ **Frontend-Backend Connection - FIXED**

Your frontend and backend are now properly connected! Here's how to run the complete system.

---

## 📋 **Prerequisites**

Before starting, ensure you have:
- ✅ **Node.js** (v14 or higher) - for backend and frontend
- ✅ **Python** (3.8 or higher) - for AI model
- ✅ **pip** - Python package manager
- ✅ **Trained AI model** - checkpoint file at `ai/experiments/best_model/checkpoints/best_model.pth`

---

## 🔧 **One-Time Setup**

### 1. Install Backend Dependencies
```powershell
cd backend
npm install
cd ..
```

### 2. Install Frontend Dependencies
```powershell
cd frontend
npm install
cd ..
```

### 3. Install Python AI Dependencies
```powershell
cd ai
pip install -r requirements.txt
cd ..
```

---

## 🎯 **Running the Complete System**

You need to run **TWO** terminals simultaneously:

### Terminal 1: Start Backend Server
```powershell
cd backend
npm start
```
✅ Backend will run on **http://localhost:3000**

### Terminal 2: Start Frontend Dev Server
```powershell
cd frontend
npm run dev
```
✅ Frontend will run on **http://localhost:5173**

---

## 🌐 **Access the Application**

Once both servers are running:

1. Open your browser
2. Go to: **http://localhost:5173**
3. Upload a chest X-ray image
4. Click "Analyze X-ray"
5. View the AI prediction results!

---

## 🔍 **How the Connection Works**

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│   Frontend      │         │   Backend       │         │   AI Engine     │
│   (React)       │────────▶│   (Node.js)     │────────▶│   (Python)      │
│   Port 5173     │  Proxy  │   Port 3000     │  Spawn  │   main.py       │
└─────────────────┘         └─────────────────┘         └─────────────────┘
```

1. **Frontend** (React + Vite) runs on port **5173**
2. **Backend** (Node.js + Express) runs on port **3000**
3. Frontend sends requests to `/predict` which Vite proxies to backend
4. Backend receives image, calls Python AI script
5. Python analyzes image and returns prediction
6. Backend sends results back to frontend
7. Frontend displays results beautifully!

---

## 🔧 **Configuration Files**

### Backend Configuration (`backend/.env`)
```
PORT=3000
CHECKPOINT_PATH=../ai/experiments/best_model/checkpoints/best_model.pth
CONFIG_PATH=../ai/configs/config.yaml
```

### Frontend Proxy (`frontend/vite.config.js`)
```javascript
proxy: {
  '/predict': 'http://localhost:3000',
  '/predict-batch': 'http://localhost:3000',
  '/health': 'http://localhost:3000',
  '/status': 'http://localhost:3000'
}
```

---

## 🐛 **Troubleshooting**

### Issue: "Failed to connect to server"
**Solution**: Make sure backend is running on port 3000
```powershell
cd backend
npm start
```

### Issue: "Model checkpoint not found"
**Solution**: Train a model first or update the checkpoint path in `backend/.env`
```powershell
cd ai
python main.py train --config configs/config.yaml --experiment-name my_model --device auto
```

### Issue: "Port 3000 already in use"
**Solution**: Kill the process using port 3000
```powershell
# Find process using port 3000
netstat -ano | findstr :3000

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F
```

### Issue: "Python not found"
**Solution**: Make sure Python is in your PATH
```powershell
python --version
```

### Issue: Frontend shows blank page
**Solution**: Check browser console for errors, ensure frontend is running
```powershell
cd frontend
npm run dev
```

---

## 📊 **API Endpoints**

Your backend exposes these endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/predict` | POST | Single image prediction |
| `/predict-batch` | POST | Batch prediction (up to 10 images) |
| `/status` | GET | Server and model status |

---

## 🎨 **Features**

✅ **Modern UI** - Beautiful gradient design with Tailwind CSS
✅ **Drag & Drop** - Easy image upload
✅ **Real-time Analysis** - Instant AI predictions
✅ **Confidence Scores** - Probability and confidence metrics
✅ **Medical Recommendations** - Actionable advice based on results
✅ **Responsive Design** - Works on all screen sizes

---

## 📝 **Testing the Connection**

### Quick Health Check
Open browser and visit:
- Backend: http://localhost:3000/health
- Frontend: http://localhost:5173

### Test API Directly
```powershell
# Check backend status
curl http://localhost:3000/status

# Check backend health
curl http://localhost:3000/health
```

---

## 🚀 **Production Deployment**

### Build Frontend for Production
```powershell
cd frontend
npm run build
```
This creates optimized files in `frontend/dist/`

### Serve Frontend with Backend
Update `backend/server.js` to serve the built frontend:
```javascript
app.use(express.static(path.join(__dirname, '../frontend/dist')));
```

Then run only the backend:
```powershell
cd backend
npm start
```

---

## 📞 **Need Help?**

If you encounter issues:
1. Check both terminal windows for error messages
2. Verify all dependencies are installed
3. Ensure ports 3000 and 5173 are available
4. Check that the AI model checkpoint exists
5. Review the troubleshooting section above

---

## ✨ **What Was Fixed**

### 1. **Frontend API URL** ✅
- **Before**: `fetch('http://localhost:3000/predict', ...)`
- **After**: `fetch('/predict', ...)` (uses Vite proxy)
- **Benefit**: No CORS issues, cleaner code

### 2. **Backend Python Path** ✅
- **Before**: `path.join(__dirname, 'ai', 'main.py')`
- **After**: `path.join(__dirname, '..', 'ai', 'main.py')`
- **Benefit**: Backend can now find and execute Python AI script

### 3. **Vite Proxy Configuration** ✅
- **Before**: Simple string proxy
- **After**: Full proxy config with `changeOrigin: true`
- **Benefit**: Proper proxy handling for all endpoints

---

## 🎉 **You're All Set!**

Your frontend and backend are now properly connected. Start both servers and enjoy your AI-powered cancer detection system!

**Developed by**: Sneh Gupta and Arpit Bhardwaj
**Course**: CSET211 - Statistical Machine Learning
