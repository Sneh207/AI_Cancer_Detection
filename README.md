# 🏥 AI-Based Cancer Detection from Chest X-rays

A full-stack deep learning application for detecting cancer from chest X-ray images using PyTorch, React, and Node.js.

**Authors:** Sneh Gupta and Arpit Bhardwaj  
**Course:** CSET211 - Statistical Machine Learning

---

## 📋 Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Model Training](#model-training)
- [API Documentation](#api-documentation)
- [Troubleshooting](#troubleshooting)

---

## ✨ Features

- **Deep Learning Model**: ResNet50-based architecture for binary cancer classification
- **Real-time Inference**: Fast prediction API with confidence scores
- **Modern Web Interface**: React-based frontend with beautiful UI
- **Grad-CAM Visualization**: Visual explanations of model predictions
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Batch Processing**: Support for multiple image analysis
- **Production Ready**: Full error handling and logging

---

## 📁 Project Structure

```
cancer_detection/
├── ai/                          # AI/ML Backend
│   ├── src/                     # Source code
│   │   ├── data_loader.py      # Data loading and preprocessing
│   │   ├── models.py           # Model architectures
│   │   ├── train.py            # Training logic
│   │   ├── evaluate.py         # Evaluation metrics
│   │   ├── inference.py        # Inference pipeline
│   │   ├── gradcam.py          # Grad-CAM visualization
│   │   └── utils.py            # Utility functions
│   ├── configs/                # Configuration files
│   │   ├── config.yaml         # Main config
│   │   └── test_config.yaml    # Test config
│   ├── data/                   # Dataset directory
│   │   └── raw/
│   │       ├── train_data/     # Training images
│   │       └── Data_Entry_2017_v2020.csv  # Labels
│   ├── experiments/            # Training outputs
│   ├── main.py                 # CLI entry point
│   ├── test_project.py         # End-to-end test
│   └── requirements.txt        # Python dependencies
│
├── backend/                    # Node.js API Server
│   ├── server.js              # Express server
│   ├── package.json           # Node dependencies
│   └── .env.example           # Environment template
│
├── frontend/                  # React Web App
│   ├── src/
│   │   ├── App.jsx           # Main component
│   │   ├── index.jsx         # Entry point
│   │   └── index.css         # Styles
│   ├── package.json          # Frontend dependencies
│   └── .env.example          # Environment template
│
├── start-dev.bat             # Windows startup script
└── README.md                 # This file
```

---

## 🔧 Prerequisites

- **Python**: 3.8 or higher
- **Node.js**: 14.0 or higher
- **npm**: 6.0 or higher
- **Git**: For version control
- **CUDA** (optional): For GPU acceleration

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd cancer_detection
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install Python dependencies
cd ai
pip install -r requirements.txt
cd ..
```

### 3. Set Up Backend

```bash
cd backend
npm install
cd ..
```

### 4. Set Up Frontend

```bash
cd frontend
npm install
cd ..
```

### 5. Configure Environment Variables

```bash
# Backend
cd backend
copy .env.example .env
# Edit .env if needed

# Frontend
cd ../frontend
copy .env.example .env
# Edit .env if needed
cd ..
```

---

## 🚀 Quick Start

### Option 1: Use the Startup Script (Windows)

```bash
start-dev.bat
```

This will automatically start both backend and frontend servers.

### Option 2: Manual Start

**Terminal 1 - Backend:**
```bash
cd backend
npm start
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

**Terminal 3 - Python (for training/inference):**
```bash
cd ai
python -m venv .venv
.venv\Scripts\activate
```

### Access the Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:3000
- **API Docs**: http://localhost:3000

---

## 💻 Usage

### Web Interface

1. Open http://localhost:5173 in your browser
2. Upload a chest X-ray image (PNG or JPG)
3. Click "Analyze X-ray"
4. View results with confidence scores

### Command Line Interface

#### Test the Project
```bash
cd ai
python test_project.py
```

#### Train a Model
```bash
python main.py train --config configs/config.yaml --experiment-name my_model --device auto
```

#### Evaluate a Model
```bash
python main.py evaluate --config configs/config.yaml --checkpoint experiments/my_model/checkpoints/best_model.pth
```

#### Run Inference
```bash
python main.py inference --config configs/config.yaml --checkpoint experiments/my_model/checkpoints/best_model.pth --image path/to/xray.png
```

---

## 🎓 Model Training

### Dataset Setup

Your dataset is already configured with **3,578 chest X-ray images**:
- 536 cancer cases (15.0%)
- 3,042 normal cases (85.0%)

Location: `ai/data/raw/train_data/train/`

### Training Configuration

Edit `ai/configs/config.yaml` to customize:
- Model architecture (ResNet50, ResNet101, DenseNet, etc.)
- Batch size and learning rate
- Data augmentation parameters
- Training epochs and early stopping

### Start Training

```bash
cd ai
python main.py train --config configs/config.yaml --experiment-name production_model --device auto
```

### Monitor Training

- **TensorBoard**: `tensorboard --logdir ai/experiments/logs`
- **Logs**: Check `ai/experiments/<experiment_name>/logs/`
- **Checkpoints**: Saved in `ai/experiments/<experiment_name>/checkpoints/`

---

## 📡 API Documentation

### Endpoints

#### `GET /`
Get API information
```json
{
  "message": "Cancer Detection API",
  "version": "1.0.0",
  "endpoints": { ... }
}
```

#### `GET /health`
Health check
```json
{
  "status": "ok",
  "timestamp": "2025-10-01T19:12:27.000Z"
}
```

#### `POST /predict`
Analyze single image

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `image` (file)

**Response:**
```json
{
  "success": true,
  "result": {
    "prediction": "Cancer" | "No Cancer",
    "probability": 0.85,
    "confidence": 0.85,
    "message": "..."
  },
  "timestamp": "2025-10-01T19:12:27.000Z"
}
```

#### `POST /predict-batch`
Analyze multiple images (up to 10)

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `images[]` (files)

#### `GET /status`
Server and model status
```json
{
  "status": "running",
  "modelAvailable": true,
  "checkpointPath": "...",
  "timestamp": "2025-10-01T19:12:27.000Z"
}
```

---

## 🔍 Troubleshooting

### Backend Issues

**Problem:** Model checkpoint not found
```
Solution: Train a model first or set CHECKPOINT_PATH in backend/.env
```

**Problem:** Port 3000 already in use
```
Solution: Change PORT in backend/.env or kill the process using port 3000
```

### Frontend Issues

**Problem:** Cannot connect to backend
```
Solution: Ensure backend is running on http://localhost:3000
Check VITE_API_URL in frontend/.env
```

### Python/AI Issues

**Problem:** CUDA out of memory
```
Solution: Reduce batch_size in configs/config.yaml
Or use --device cpu flag
```

**Problem:** Module not found errors
```
Solution: Ensure virtual environment is activated
pip install -r ai/requirements.txt
```

**Problem:** Data loading errors
```
Solution: Verify ai/data/raw/train_data/train/ contains images
Check ai/data/raw/Data_Entry_2017_v2020.csv exists
```

### Windows-Specific Issues

**Problem:** num_workers error in DataLoader
```
Solution: Already fixed - num_workers set to 0 in config.yaml
```

---

## 📊 Model Performance

Current model trained on 3,578 chest X-rays:
- **Architecture**: ResNet50 (pretrained on ImageNet)
- **Input Size**: 224x224 pixels
- **Classes**: Binary (Cancer / No Cancer)
- **Training**: 70% train, 15% validation, 15% test split

---

## 🛠️ Development

### Running Tests

```bash
# Python tests
cd ai
python test_project.py

# Backend tests
cd backend
npm test

# Frontend tests
cd frontend
npm test
```

### Code Style

- **Python**: Follow PEP 8
- **JavaScript**: ESLint configuration included
- **React**: Functional components with hooks

---

## 📝 License

MIT License - See LICENSE file for details

---

## 👥 Authors

- **Sneh Gupta**
- **Arpit Bhardwaj**

**Course**: CSET211 - Statistical Machine Learning

---

## 🙏 Acknowledgments

- NIH Chest X-ray Dataset
- PyTorch and torchvision teams
- React and Vite communities
- All open-source contributors

---

## 📞 Support

For issues and questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review existing documentation
3. Contact the development team

---

**⚠️ Medical Disclaimer**: This system is intended for research and educational purposes only. It is NOT a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical advice.
