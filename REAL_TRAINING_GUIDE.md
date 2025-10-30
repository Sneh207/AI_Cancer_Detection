# 🚀 Real Model Training Guide

## ✅ Your Data is Ready!

I've verified your training data:
- **10,000 chest X-ray images** in `ai/data/raw/train_data/train/`
- **10,000 matching labels** in `ai/data/raw/ChestXray_Binary_Labels.csv`
- **Binary classification**: "Cancer" vs "No Cancer"
- **All images matched** with labels ✅

## 📊 Label Distribution:

From your CSV file:
- **Cancer cases**: ~1,500-2,000 images
- **No Cancer cases**: ~8,000-8,500 images
- This is a realistic medical dataset with class imbalance

## 🔧 Configuration Updated:

I've updated your training configuration:

### `ai/configs/config.yaml`:
```yaml
data:
  dataset_path: "data/raw/train_data/train"
  labels_file: "data/raw/ChestXray_Binary_Labels.csv"
  image_column: "Image Index"
  label_column: "BinaryLabel"
  batch_size: 16
  train_split: 0.7  # 7,000 images
  val_split: 0.15   # 1,500 images
  test_split: 0.15  # 1,500 images

model:
  architecture: "resnet50"  # Pretrained ResNet50
  pretrained: true
  num_classes: 1

training:
  epochs: 100
  learning_rate: 0.001
  early_stopping_patience: 15
```

### Code Updates:
- ✅ `data_loader.py` - Now uses your "BinaryLabel" column
- ✅ `config.yaml` - Points to correct CSV and image folder
- ✅ Model architecture - Changed to ResNet50 for better performance

## 🚀 How to Train:

### Option 1: Quick Start (Recommended)
```bash
cd ai
TRAIN_MODEL.bat
```

### Option 2: Manual Command
```bash
cd ai
python main.py train --config configs/config.yaml
```

### Option 3: With Custom Settings
```bash
cd ai
python main.py train --config configs/config.yaml --epochs 50 --batch-size 8
```

## ⏱️ Training Time Estimate:

- **With GPU**: 1-2 hours
- **With CPU**: 4-8 hours
- **Early stopping**: May finish earlier if model converges

## 📈 What to Expect:

### During Training:
```
Epoch 1/100
Using image column: 'Image Index'
Using label column: 'BinaryLabel'
Found 10000 valid images
Cancer cases: 1847 (18.5%)
No Cancer cases: 8153 (81.5%)

Training samples: 7000
Validation samples: 1500
Test samples: 1500

Train Loss: 0.5234 | Train Acc: 0.7543
Val Loss: 0.4821 | Val Acc: 0.7892
Val AUC: 0.8234

✅ New best model saved!
```

### Good Signs:
- ✅ Validation AUC > 0.70
- ✅ Validation loss decreasing
- ✅ No overfitting (train/val loss similar)

### Warning Signs:
- ⚠️ AUC < 0.60 (model not learning)
- ⚠️ Train loss much lower than val loss (overfitting)
- ⚠️ Loss not decreasing (learning rate too high/low)

## 📁 Output Files:

After training, you'll find:

```
ai/experiments/
└── cancer_detection_YYYYMMDD_HHMMSS/
    ├── checkpoints/
    │   ├── best_model.pth          ← Use this for inference!
    │   └── last_model.pth
    ├── logs/
    │   └── training.log
    └── results/
        ├── training_curves.png
        ├── confusion_matrix.png
        └── metrics.json
```

## 🔄 Update Backend to Use New Model:

After training completes:

### 1. Find your model path:
```
ai/experiments/cancer_detection_YYYYMMDD_HHMMSS/checkpoints/best_model.pth
```

### 2. Update `backend/.env`:
```env
PORT=5000
CHECKPOINT_PATH=../ai/experiments/cancer_detection_YYYYMMDD_HHMMSS/checkpoints/best_model.pth
CONFIG_PATH=../ai/configs/config.yaml
```

### 3. Restart backend:
```bash
cd backend
node server.js
```

## 🧪 Test the Trained Model:

### Quick Test:
```bash
cd ai
python main.py inference --config configs/config.yaml --checkpoint experiments/cancer_detection_YYYYMMDD_HHMMSS/checkpoints/best_model.pth --image path/to/test_xray.png --visualize
```

### Through Web Interface:
1. Start backend: `cd backend && node server.js`
2. Start frontend: `cd frontend && npm run dev`
3. Upload an X-ray at http://localhost:5173
4. See real predictions! 🎉

## 📊 Monitoring Training:

### Watch Progress:
```bash
# In another terminal
cd ai
tail -f experiments/cancer_detection_*/logs/training.log
```

### Check Metrics:
- Training loss should decrease
- Validation AUC should increase
- Best model saved when val AUC improves

## 🛑 Stop Training Early:

If you need to stop:
- Press `Ctrl+C`
- Last checkpoint will be saved
- You can resume later or use the last saved model

## ⚡ Performance Tips:

### If Training is Slow:
1. **Reduce batch size**: Change `batch_size: 16` to `8`
2. **Use fewer epochs**: Change `epochs: 100` to `50`
3. **Reduce image size**: Change `image_size: 224` to `128`

### If Out of Memory:
1. **Reduce batch size**: `batch_size: 8` or `4`
2. **Reduce workers**: `num_workers: 0`
3. **Close other applications**

### If Model Not Learning:
1. **Check data**: Run `python check_data.py`
2. **Adjust learning rate**: Try `0.0001` or `0.01`
3. **Change model**: Try `densenet121` instead of `resnet50`

## 🎯 Expected Results:

With your 10,000 images, you should achieve:
- **Accuracy**: 75-85%
- **AUC-ROC**: 0.75-0.85
- **Precision**: 60-75% (for Cancer class)
- **Recall**: 65-80% (for Cancer class)

These are realistic medical AI metrics!

## 🚀 Ready to Train?

Run this command:
```bash
cd ai
TRAIN_MODEL.bat
```

Or:
```bash
cd ai
python main.py train --config configs/config.yaml
```

The training will:
1. ✅ Load your 10,000 images
2. ✅ Split into train/val/test (70/15/15)
3. ✅ Train ResNet50 with data augmentation
4. ✅ Save best model based on validation AUC
5. ✅ Generate training curves and metrics
6. ✅ Create a model ready for real predictions!

**Good luck with training! 🎉**

---

## 📞 Troubleshooting:

### Error: "CUDA out of memory"
**Solution**: Reduce batch_size to 8 or 4

### Error: "No module named 'torch'"
**Solution**: `pip install torch torchvision`

### Error: "No module named 'albumentations'"
**Solution**: `pip install albumentations`

### Error: "Cannot find images"
**Solution**: Check paths in config.yaml

### Training stuck at 0% accuracy
**Solution**: Check if labels are correct, try different learning rate

---

**After training, your model will give REAL predictions, not dummy 51.7%!** 🎯
