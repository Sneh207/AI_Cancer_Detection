# AI-Based Lung Cancer Detection from Chest X-rays
## Comprehensive Project Overview for Faculty Presentation

**Authors:** Sneh Gupta and Arpit Bhardwaj  
**Course:** CSET211 - Statistical Machine Learning  
**Institution:** Bennett University

---

## 1. EXECUTIVE SUMMARY

This is a **full-stack deep learning application** designed to detect lung cancer from chest X-ray images. The system combines:
- **AI/ML Backend**: PyTorch-based deep learning models
- **API Layer**: Node.js/Express REST API
- **Web Frontend**: React/Vite user interface

The application processes radiographic images through a trained neural network, provides predictions with confidence scores, and generates visual explanations (Grad-CAM heatmaps) to improve clinical interpretability.

---

## 2. METHODOLOGY

### 2.1 Three-Layer Modular Architecture

```
┌─────────────────────────────────────────────────────┐
│           React/Vite Frontend (UI Layer)            │
│  - Image upload & drag-drop interface               │
│  - Real-time prediction display                     │
│  - Grad-CAM visualization overlay                   │
│  - Batch processing support                         │
└──────────────────┬──────────────────────────────────┘
                   │ (HTTP/REST)
┌──────────────────▼──────────────────────────────────┐
│      Node.js/Express Backend (API Layer)            │
│  - /predict endpoint (single image)                 │
│  - /predict-batch endpoint (multiple images)        │
│  - /status endpoint (health monitoring)             │
│  - Multer file upload handling                      │
│  - Python inference orchestration                   │
└──────────────────┬──────────────────────────────────┘
                   │ (Child Process)
┌──────────────────▼──────────────────────────────────┐
│   Python ML Layer (Computational Core)              │
│  - Data preprocessing & normalization               │
│  - Model inference (ResNet50/DenseNet/EfficientNet) │
│  - Grad-CAM heatmap generation                      │
│  - Prediction confidence scoring                    │
└─────────────────────────────────────────────────────┘
```

### 2.2 Python-Based Machine Learning Layer

**Core Responsibilities:**

#### Data Preprocessing
- Load raw chest X-ray images (JPEG/PNG)
- Normalize pixel values to ImageNet standards (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Resize images to 224×224 pixels
- Apply domain-relevant transformations

#### Data Augmentation & Class Balancing
- **Augmentation Strategy**: Applied only during training to prevent data leakage
  - Random horizontal/vertical flips (50% probability)
  - Rotation range: ±15 degrees
  - Brightness adjustments: ±20%
  - Contrast adjustments: ±20%
  - Zoom range: ±10%

- **Oversampling for Class Imbalance**:
  - Dataset composition: ~17% cancer (positive), ~83% normal (negative)
  - Uses `WeightedRandomSampler` to balance class representation
  - Ensures minority-class samples are sufficiently represented during training

#### Model Training & Validation
- **Architecture**: Transfer learning from ImageNet pretrained weights
- **Supported Models**:
  - ResNet50 (primary)
  - ResNet101
  - DenseNet121
  - DenseNet169
  - EfficientNet variants
  
- **Training Configuration**:
  - Optimizer: Adam (learning rate: 0.0001, weight decay: 0.0001)
  - Loss Function: Binary Cross-Entropy with Logits
  - Class Weight (pos_weight): 4.4 (to counter class imbalance)
  - Learning Rate Scheduler: Cosine Annealing
  - Batch Size: 16
  - Max Epochs: 50
  - Early Stopping Patience: 15 epochs

- **Monitoring**:
  - TensorBoard integration for real-time training visualization
  - Automatic checkpoint saving
  - Best model selection based on validation metrics

#### Inference & Interpretability
- **Single & Batch Prediction**: Supports processing one or multiple images
- **Grad-CAM Visualization**: 
  - Generates pixel-wise attention maps
  - Highlights regions influencing the model's decision
  - Improves clinical trust and interpretability
  - Overlaid on original X-ray for visual comparison

### 2.3 Node.js/Express Backend (API Layer)

**Purpose**: Mediates between AI model and web UI

**REST Endpoints**:
```
POST /predict
  Input: Single chest X-ray image (multipart/form-data)
  Output: {
    prediction: 0 or 1 (0=normal, 1=cancer),
    confidence: 0.0-1.0,
    gradcam_path: "path/to/heatmap.png"
  }

POST /predict-batch
  Input: Multiple chest X-ray images
  Output: Array of predictions with confidence scores

GET /status
  Output: Server health status and model availability
```

**Key Features**:
- Multer middleware for secure file uploads (10MB limit, image format validation)
- CORS enabled for cross-origin requests
- Child process spawning to run Python inference scripts
- Asynchronous request handling
- Error handling and logging

**Deployment Flexibility**:
- Can be deployed on-premise or in cloud environments
- Suitable for integration with clinical systems (PACS/HIS pipelines)
- RESTful design enables easy integration with external applications

### 2.4 React/Vite Frontend Layer

**User Interface Design**:
- **Modern Stack**: React + Vite for fast development and production builds
- **Styling**: TailwindCSS for responsive, utility-first design
- **Animations**: Framer Motion for smooth UI transitions
- **Icons**: Lucide React for consistent iconography
- **Components**: shadcn/ui for accessible, pre-built components

**User Features**:
- Intuitive image upload with drag-and-drop support
- Real-time prediction display with confidence percentage
- Predicted class visualization (Cancer/Normal)
- Integrated Grad-CAM heatmap overlay on original X-ray
- Batch processing interface for analyzing multiple radiographs
- Dark mode toggle for user preference
- Responsive design for desktop and tablet use

**Communication**:
- Axios-based HTTP client for backend communication
- Proper error handling and user feedback
- Loading states during inference

### 2.5 Training Workflow

**Standardized Procedure**:

```
1. Dataset Loading
   └─ Read CSV metadata (ChestXray_Binary_Labels.csv)
   └─ Load image paths and corresponding labels

2. Data Splitting
   └─ Stratified train/val/test split (70%/15%/15%)
   └─ Ensures class distribution across all sets

3. Class Balancing
   └─ Apply WeightedRandomSampler
   └─ Oversample minority class (cancer) during training

4. Batch Creation
   └─ DataLoader with configurable batch size (16)
   └─ Augmentations applied to training batches only

5. Model Fine-tuning
   └─ Transfer learning from ImageNet weights
   └─ GPU acceleration for faster training
   └─ Binary classification head (1 output neuron)

6. Performance Tracking
   └─ TensorBoard logging of loss, accuracy, metrics
   └─ Real-time visualization during training

7. Checkpoint Management
   └─ Periodic saving of model weights
   └─ Best model selection based on validation metrics
   └─ Early stopping to prevent overfitting
```

**Benefits**:
- ✅ Reproducibility: Configuration-driven parameters
- ✅ Adaptability: Easy model/architecture switching
- ✅ Traceability: Complete training history and checkpoints
- ✅ Clinical Compliance: Meets validation requirements

---

## 3. DATA STRUCTURE AND ALGORITHMS

### 3.1 Dataset Structure

**Source**: ChestXray_Binary_Labels.csv metadata file

**Contents**:
- Image filenames (indexed references)
- Pathology labels (binary: cancer/normal)
- Study identifiers for tracking

**Image Storage**:
- Format: JPEG/PNG
- Location: `data/raw/train_data/train/`
- Total Images: 5,641 chest X-rays
- Class Distribution: ~17% cancer, ~83% normal

### 3.2 DataLoader and Sampling Strategy

**PyTorch DataLoader Implementation**:

```python
# Stratified split ensures balanced class distribution
train_split: 70% (3,948 images)
val_split:   15% (846 images)
test_split:  15% (847 images)

# WeightedRandomSampler for class balancing
- Calculates inverse class frequencies
- Assigns higher sampling probability to minority class
- Generates balanced batches during training
```

**Augmentation Strategy**:
- ✅ Applied to training data only (prevents data leakage)
- ❌ NOT applied to validation/test data (maintains integrity)

**Oversampling Details**:
- Identifies minority-class (cancer) indices
- Generates k augmented versions per sample
- Promotes balanced batch composition
- Stabilizes training and prevents class bias

### 3.3 Model Architecture

**Primary Model**: ResNet50 (50-layer Residual Network)

**Alternative Architectures**:
- ResNet101 (deeper variant)
- DenseNet121 (dense connections)
- DenseNet169 (larger dense variant)
- EfficientNet (parameter-efficient)

**Key Architectural Features**:
```
ImageNet Pretrained Weights
         ↓
   Feature Extraction Layers
   (Convolutional Blocks)
         ↓
   Global Average Pooling
         ↓
   Fully Connected Layers
   (with Dropout: 0.5)
         ↓
   Binary Classification Head
   (1 output neuron + Sigmoid)
         ↓
   Prediction: 0 (Normal) or 1 (Cancer)
```

**Transfer Learning Benefits**:
- Faster convergence (leverages learned features)
- Better generalization (pretrained on 1M+ ImageNet images)
- Reduced training time and computational cost
- Improved performance on limited medical imaging data

### 3.4 Algorithms Used

#### 1. Transfer Learning
- **Purpose**: Leverage pretrained ImageNet weights for medical imaging
- **Implementation**: Fine-tune final layers while keeping early layers frozen
- **Benefit**: Dramatically faster convergence and better performance

#### 2. Binary Cross-Entropy Loss with Class Weights
```
Loss = -[y*log(p) + (1-y)*log(1-p)] * weight
where:
  y = true label (0 or 1)
  p = predicted probability
  weight = pos_weight (4.4 for cancer class)
```
- **Purpose**: Mitigate class imbalance by penalizing minority-class misclassifications
- **Benefit**: Prevents model from defaulting to "normal" predictions

#### 3. Grad-CAM (Gradient-weighted Class Activation Mapping)
```
Grad-CAM = ReLU(Σ αc * Ac)
where:
  αc = gradient of class score w.r.t. feature maps
  Ac = activation maps from final conv layer
```
- **Purpose**: Generate pixel-wise attention maps showing decision regions
- **Benefit**: Improves interpretability and clinical trust
- **Output**: Heatmap overlay highlighting suspected cancer areas

#### 4. Weighted Sampling / Oversampling
- **Purpose**: Balance training batches despite class imbalance
- **Implementation**: WeightedRandomSampler assigns higher probability to minority class
- **Benefit**: Stabilizes training and prevents model bias

#### 5. Cosine Annealing Learning Rate Scheduler
```
lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T))
```
- **Purpose**: Gradually reduce learning rate during training
- **Benefit**: Helps model converge to better local minima

---

## 4. CURRENT PROJECT STATUS

### 4.1 Implemented Components

✅ **ML Pipeline**
- Data loading and preprocessing
- Multiple model architectures (ResNet50, DenseNet, EfficientNet)
- Training with augmentation and class balancing
- Inference pipeline with Grad-CAM
- TensorBoard monitoring
- Checkpoint management

✅ **Backend API**
- Express.js REST server
- Image upload handling (Multer)
- Single and batch prediction endpoints
- Health status endpoint
- Python process orchestration

✅ **Frontend UI**
- React/Vite application
- Image upload with drag-and-drop
- Real-time prediction display
- Grad-CAM visualization
- Batch processing interface
- Dark mode support
- Responsive design

✅ **Configuration System**
- YAML-based configuration (config.yaml)
- Flexible model selection
- Adjustable training parameters
- Environment variable support

### 4.2 Configuration Details

**Main Config File**: `ai/configs/config.yaml`

```yaml
Data Configuration:
  - Dataset path: data/raw/train_data/train
  - Labels file: ChestXray_Binary_Labels.csv
  - Image size: 224×224 pixels
  - Batch size: 16
  - Train/Val/Test split: 70%/15%/15%

Augmentation:
  - Horizontal flip: 50% probability
  - Rotation: ±15 degrees
  - Brightness: ±20%
  - Contrast: ±20%
  - Zoom: ±10%

Training:
  - Epochs: 50
  - Learning rate: 0.0001
  - Optimizer: Adam
  - Scheduler: Cosine Annealing
  - Early stopping patience: 15 epochs

Model:
  - Architecture: ResNet50 (configurable)
  - Pretrained: Yes (ImageNet weights)
  - Dropout: 0.5
  - Binary classification (1 output)

Loss Function:
  - Type: Binary Cross-Entropy
  - pos_weight: 4.4 (for class imbalance)

Evaluation Metrics:
  - Accuracy, Precision, Recall, F1, AUC-ROC
```

### 4.3 Key Metrics & Performance Tracking

**Monitored During Training**:
- Training loss and validation loss
- Accuracy (overall correctness)
- Precision (true positives / predicted positives)
- Recall (true positives / actual positives)
- F1-Score (harmonic mean of precision and recall)
- AUC-ROC (area under receiver operating characteristic curve)

**Visualization**:
- TensorBoard dashboard for real-time monitoring
- Confusion matrix for detailed error analysis
- Loss curves for convergence analysis

---

## 5. PROJECT STRUCTURE

```
AI_Cancer_Detection/
├── ai/                                    # ML/AI Backend
│   ├── main.py                           # Training/evaluation entry point
│   ├── requirements.txt                  # Python dependencies
│   ├── configs/
│   │   └── config.yaml                   # Main configuration
│   ├── src/
│   │   ├── data_loader.py               # Dataset & DataLoader
│   │   ├── models.py                    # Model architectures
│   │   ├── train.py                     # Training logic
│   │   ├── evaluate.py                  # Evaluation metrics
│   │   ├── inference.py                 # Inference pipeline
│   │   ├── gradcam.py                   # Grad-CAM visualization
│   │   └── utils.py                     # Utility functions
│   ├── data/
│   │   └── raw/
│   │       ├── train_data/train/        # 5,641 X-ray images
│   │       └── ChestXray_Binary_Labels.csv
│   ├── notebooks/
│   │   ├── baseline_model.ipynb         # Baseline experiments
│   │   └── data_exploration.ipynb       # Data analysis
│   └── experiments/                     # Training outputs
│       └── checkpoints/                 # Model checkpoints
│
├── backend/                              # Node.js API Server
│   ├── server.js                        # Express server
│   ├── package.json                     # Node dependencies
│   ├── .env                             # Environment variables
│   └── uploads/                         # Temporary uploads
│
├── frontend/                             # React/Vite UI
│   ├── src/
│   │   ├── App.jsx                      # Main app component
│   │   ├── components/
│   │   │   ├── Navbar.jsx
│   │   │   ├── HeroSection.jsx
│   │   │   ├── UploadSection.jsx
│   │   │   ├── ResultsSection.jsx
│   │   │   └── Footer.jsx
│   │   └── index.css                    # TailwindCSS styles
│   ├── package.json                     # Node dependencies
│   ├── vite.config.js                   # Vite configuration
│   └── tailwind.config.js               # TailwindCSS config
│
└── README.md                             # Project documentation
```

---

## 6. WORKFLOW DIAGRAM

```
User Interface (React)
        ↓
   [Upload X-ray]
        ↓
Backend API (Express)
        ↓
   [Validate Image]
        ↓
Python ML Pipeline
        ├─ [Preprocess Image]
        ├─ [Load Model]
        ├─ [Run Inference]
        └─ [Generate Grad-CAM]
        ↓
Backend API
        ↓
   [Format Response]
        ↓
User Interface
        ├─ [Display Prediction]
        ├─ [Show Confidence %]
        └─ [Overlay Grad-CAM]
```

---

## 7. TECHNICAL STACK

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | React 18 | UI framework |
| | Vite | Build tool & dev server |
| | TailwindCSS | Styling |
| | Framer Motion | Animations |
| | Axios | HTTP client |
| **Backend** | Node.js | Runtime |
| | Express.js | Web framework |
| | Multer | File uploads |
| | CORS | Cross-origin requests |
| **ML/AI** | PyTorch | Deep learning framework |
| | Torchvision | Computer vision utilities |
| | Albumentations | Image augmentation |
| | scikit-learn | Metrics & utilities |
| | TensorBoard | Training visualization |
| **Data** | Pandas | Data manipulation |
| | NumPy | Numerical computing |
| | PIL/Pillow | Image processing |

---

## 8. KEY FEATURES & ADVANTAGES

### Clinical Relevance
- ✅ Binary classification (Cancer/Normal) for clear decision support
- ✅ Confidence scores for clinical confidence assessment
- ✅ Grad-CAM visualization for radiologist trust and verification
- ✅ Batch processing for high-throughput screening

### Technical Robustness
- ✅ Transfer learning for better generalization
- ✅ Automatic class balancing to prevent bias
- ✅ Early stopping to prevent overfitting
- ✅ Comprehensive evaluation metrics
- ✅ Checkpoint management for reproducibility

### Scalability & Deployment
- ✅ Modular architecture for independent scaling
- ✅ RESTful API for easy integration
- ✅ Support for on-premise or cloud deployment
- ✅ Configuration-driven parameters for flexibility

### User Experience
- ✅ Intuitive web interface
- ✅ Real-time predictions
- ✅ Visual explanations (Grad-CAM)
- ✅ Responsive design
- ✅ Dark mode support

---

## 9. FUTURE ENHANCEMENTS

- 🔄 Multi-class classification (expand beyond binary)
- 🔄 Ensemble methods combining multiple architectures
- 🔄 DICOM image support for clinical integration
- 🔄 Database integration for patient record management
- 🔄 Advanced visualization (3D reconstructions)
- 🔄 Model explainability improvements (LIME, SHAP)
- 🔄 Performance optimization for edge deployment
- 🔄 Federated learning for privacy-preserving training

---

## 10. CONCLUSION

This project demonstrates a **production-ready AI system** for medical image analysis, combining:
- State-of-the-art deep learning techniques
- Robust software engineering practices
- Clinical interpretability through Grad-CAM
- User-friendly interface for medical professionals

The three-layer modular architecture ensures **scalability, maintainability, and easy integration** with existing clinical workflows.

---

**For Questions or Clarifications**: Refer to individual component documentation or contact the development team.
