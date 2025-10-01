# 🔌 Frontend-Backend Connection Documentation

## ✅ **Connection Status: FULLY CONFIGURED**

Your frontend and backend are now properly connected and ready to work together!

---

## 🏗️ **Architecture Overview**

```
┌──────────────────────────────────────────────────────────────────┐
│                        USER BROWSER                              │
│                    http://localhost:5173                         │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             │ HTTP Request: POST /predict
                             │ (with image file)
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                    VITE DEV SERVER (Proxy)                       │
│                       Port 5173                                  │
│  - Serves React frontend                                         │
│  - Proxies API calls to backend                                  │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             │ Proxied to: http://localhost:3000/predict
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                   NODE.JS BACKEND SERVER                         │
│                       Port 3000                                  │
│  - Receives image upload                                         │
│  - Validates file                                                │
│  - Saves to temporary location                                   │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             │ Spawns Python process
                             │ python ai/main.py inference --image ...
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                     PYTHON AI ENGINE                             │
│                    ai/main.py                                    │
│  - Loads trained model checkpoint                                │
│  - Preprocesses image                                            │
│  - Runs inference                                                │
│  - Returns prediction + confidence                               │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             │ Returns JSON result
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                   NODE.JS BACKEND SERVER                         │
│  - Parses Python output                                          │
│  - Formats response                                              │
│  - Cleans up temp files                                          │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             │ Returns JSON: {success, result, timestamp}
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                    REACT FRONTEND                                │
│  - Displays prediction                                           │
│  - Shows confidence scores                                       │
│  - Renders medical recommendation                                │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🔧 **Configuration Details**

### 1. Frontend Configuration (`frontend/vite.config.js`)

```javascript
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/predict': {
        target: 'http://localhost:3000',
        changeOrigin: true
      },
      '/predict-batch': {
        target: 'http://localhost:3000',
        changeOrigin: true
      },
      '/health': {
        target: 'http://localhost:3000',
        changeOrigin: true
      },
      '/status': {
        target: 'http://localhost:3000',
        changeOrigin: true
      }
    }
  }
})
```

**What this does:**
- Runs frontend on port **5173**
- Proxies all `/predict`, `/predict-batch`, `/health`, `/status` requests to backend
- `changeOrigin: true` ensures proper CORS handling

### 2. Frontend API Call (`frontend/src/App.jsx`)

```javascript
const response = await fetch('/predict', {
  method: 'POST',
  body: formData,  // Contains the image file
});
```

**Key points:**
- Uses relative URL `/predict` (not `http://localhost:3000/predict`)
- Vite proxy automatically forwards to backend
- No CORS issues because proxy handles it

### 3. Backend Configuration (`backend/.env`)

```
PORT=3000
CHECKPOINT_PATH=../ai/experiments/best_model/checkpoints/best_model.pth
CONFIG_PATH=../ai/configs/config.yaml
```

**Key points:**
- Backend listens on port **3000**
- Checkpoint path points to trained AI model
- Config path points to AI configuration

### 4. Backend Server (`backend/server.js`)

```javascript
const PORT = process.env.PORT || 3000;

// CORS enabled for all origins
app.use(cors());

// File upload endpoint
app.post('/predict', upload.single('image'), async (req, res) => {
  // ... handles image upload and calls Python AI
});

// Python script path (FIXED)
const pythonScript = path.join(__dirname, '..', 'ai', 'main.py');
```

**Key points:**
- CORS enabled (allows frontend to communicate)
- Multer handles file uploads (max 10MB)
- Correct path to Python AI script (`..` goes up one level)

---

## 📡 **API Request/Response Flow**

### Request from Frontend

```javascript
// Frontend sends FormData with image
const formData = new FormData();
formData.append('image', selectedFile);

fetch('/predict', {
  method: 'POST',
  body: formData
});
```

### Backend Processes Request

```javascript
// Backend receives file
app.post('/predict', upload.single('image'), async (req, res) => {
  const imagePath = req.file.path;
  
  // Calls Python AI
  const output = await runPythonInference(imagePath, checkpointPath, configPath);
  
  // Parses output
  const result = parsePythonOutput(output);
  
  // Returns JSON
  res.json({
    success: true,
    result: {
      prediction: 'Cancer' or 'No Cancer',
      probability: 0.85,
      confidence: 0.92,
      message: 'Medical recommendation...'
    },
    timestamp: '2025-10-01T11:26:51+05:30'
  });
});
```

### Frontend Receives Response

```javascript
const data = await response.json();

if (data.success) {
  setResult(data.result);
  // Displays:
  // - Prediction badge (Cancer / No Cancer)
  // - Probability bar (85%)
  // - Confidence bar (92%)
  // - Medical recommendation
}
```

---

## 🚀 **Starting the System**

### Option 1: Manual Start (Recommended for Development)

**Terminal 1 - Backend:**
```powershell
cd backend
npm start
```

**Terminal 2 - Frontend:**
```powershell
cd frontend
npm run dev
```

### Option 2: Automated Start (Quick)

**Double-click:**
```
start-dev.bat
```

This opens two terminal windows automatically.

---

## 🧪 **Testing the Connection**

### 1. Test Backend Health
```powershell
# Open browser or use curl
curl http://localhost:3000/health

# Expected response:
# {"status":"ok","timestamp":"2025-10-01T11:26:51.000Z"}
```

### 2. Test Backend Status
```powershell
curl http://localhost:3000/status

# Expected response:
# {
#   "status": "running",
#   "modelAvailable": true,
#   "checkpointPath": "...",
#   "timestamp": "..."
# }
```

### 3. Test Frontend Proxy
1. Open browser: http://localhost:5173
2. Open Developer Tools (F12)
3. Go to Network tab
4. Upload an image and click "Analyze"
5. Check the network request:
   - Request URL should be: `http://localhost:5173/predict`
   - But it's actually sent to: `http://localhost:3000/predict`
   - Status should be: `200 OK`

---

## 🐛 **Common Issues & Solutions**

### Issue 1: "Failed to connect to server"

**Symptoms:**
- Frontend shows error message
- Network tab shows failed request

**Solutions:**
1. Check if backend is running:
   ```powershell
   curl http://localhost:3000/health
   ```
2. Restart backend:
   ```powershell
   cd backend
   npm start
   ```

### Issue 2: "Model checkpoint not found"

**Symptoms:**
- Backend returns 500 error
- Backend console shows "Model checkpoint not found"

**Solutions:**
1. Train a model first:
   ```powershell
   cd ai
   python main.py train --config configs/config.yaml --experiment-name my_model
   ```
2. Or update checkpoint path in `backend/.env`

### Issue 3: CORS Errors

**Symptoms:**
- Browser console shows CORS error
- Request blocked by CORS policy

**Solutions:**
1. Ensure backend has `app.use(cors())` enabled
2. Use relative URLs in frontend (`/predict` not `http://localhost:3000/predict`)
3. Restart both servers

### Issue 4: Port Already in Use

**Symptoms:**
- "EADDRINUSE: address already in use :::3000"

**Solutions:**
```powershell
# Find process using port 3000
netstat -ano | findstr :3000

# Kill the process
taskkill /PID <PID> /F
```

---

## 📊 **Performance Considerations**

### File Upload Limits
- **Max file size**: 10MB (configured in `backend/server.js`)
- **Allowed formats**: JPG, JPEG, PNG
- **Batch limit**: 10 images per request

### Inference Time
- **Typical**: 2-5 seconds per image
- **Depends on**: Model size, CPU/GPU, image resolution

### Optimization Tips
1. **Use GPU**: Set `--device auto` in Python inference
2. **Reduce image size**: Resize large images before upload
3. **Cache model**: Keep model loaded in memory (requires backend modification)

---

## 🔒 **Security Notes**

### Current Setup (Development)
- ✅ CORS enabled for all origins
- ✅ File type validation (images only)
- ✅ File size limit (10MB)
- ✅ Temporary file cleanup

### Production Recommendations
- 🔒 Restrict CORS to specific domains
- 🔒 Add authentication/authorization
- 🔒 Implement rate limiting
- 🔒 Add request validation
- 🔒 Use HTTPS
- 🔒 Sanitize file names
- 🔒 Scan uploaded files for malware

---

## 📝 **Environment Variables**

### Backend (`.env`)
```
PORT=3000                                                    # Backend port
CHECKPOINT_PATH=../ai/experiments/best_model/checkpoints/best_model.pth
CONFIG_PATH=../ai/configs/config.yaml
```

### Frontend (Vite automatically uses)
```
VITE_API_URL=http://localhost:3000   # Optional, for production
```

---

## 🎯 **Next Steps**

1. ✅ **Connection is working** - Frontend and backend are connected
2. 🔄 **Train a model** - Ensure you have a trained checkpoint
3. 🧪 **Test with real images** - Upload chest X-rays and verify predictions
4. 🎨 **Customize UI** - Modify frontend design as needed
5. 🚀 **Deploy** - Build for production when ready

---

## 📞 **Support**

If you encounter issues:
1. Check both terminal windows for errors
2. Review the troubleshooting section
3. Verify all dependencies are installed
4. Ensure ports 3000 and 5173 are available

---

**✨ Your frontend and backend are now fully connected and ready to use!**

**Developed by**: Sneh Gupta and Arpit Bhardwaj
**Course**: CSET211 - Statistical Machine Learning
