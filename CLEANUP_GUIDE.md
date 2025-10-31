# 🧹 Project Cleanup Guide

Files to remove before creating ai.zip for Colab training.

---

## ❌ Files to DELETE

### In `ai/` folder:

```powershell
# Navigate to ai folder
cd C:\AI_Lung_Cancer\AI_Cancer_Detection\ai

# Delete unnecessary files
Remove-Item -Path @(
    "colab_training.ipynb",              # Old notebook (replaced by COLAB_TRAINING_GUIDE.md)
    "check_checkpoint_architecture.py",   # Debug script
    "check_checkpoint_metrics.py",        # Debug script
    "check_data.py",                      # Debug script
    "check_model_type.py",                # Debug script
    "check_training_data.py",             # Debug script
    "create_balanced_labels.py",          # One-time script
    "create_dummy_checkpoint.py",         # Test script
    "fix_data_structure.py",              # One-time script
    "read_tensorboard_metrics.py",        # Debug script
    "test_model.py",                      # Local test only
    "test_project.py",                    # Local test only
    "MONITOR_TRAINING.bat",               # Windows only
    "TRAIN_MODEL.bat"                     # Windows only
) -Force -ErrorAction SilentlyContinue

# Delete virtual environment (huge and not needed in Colab)
Remove-Item -Recurse -Force ".venv" -ErrorAction SilentlyContinue

# Delete experiments folder (old training runs)
Remove-Item -Recurse -Force "experiments" -ErrorAction SilentlyContinue

# Delete data.zip if it exists (data is already in data/ folder)
Remove-Item "data.zip" -Force -ErrorAction SilentlyContinue

Write-Host "✅ Cleanup complete!"
```

---

## ✅ Files to KEEP

### Essential Files:
- `main.py` - Entry point
- `requirements.txt` - Dependencies
- `environment.yml` - Conda config
- `README.md` - Documentation
- `COLAB_TRAINING_GUIDE.md` - Colab instructions

### Essential Folders:
- `src/` - Source code
- `configs/` - Configuration files
- `data/` - Dataset
- `notebooks/` - Jupyter notebooks
- `scripts/` - Utility scripts
- `tests/` - Test files
- `utils/` - Utility functions

---

## 📦 Create Clean ZIP

After cleanup, create the ZIP:

```powershell
# Navigate to project root
cd C:\AI_Lung_Cancer\AI_Cancer_Detection

# Create clean ZIP
Compress-Archive -Path "ai" -DestinationPath "ai_clean.zip" -Force

# Check size
Write-Host "`nZIP Size: $((Get-Item ai_clean.zip).Length / 1GB) GB"

# Verify contents
Add-Type -Assembly System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead("$PWD\ai_clean.zip")
Write-Host "`nTotal files in ZIP: $($zip.Entries.Count)"
$zip.Dispose()
```

---

## 🎯 Expected Results

### Before Cleanup:
- Size: ~4.8 GB (with data.zip, .venv, experiments)
- Files: 1000+ files

### After Cleanup:
- Size: ~4.5 GB (just code + data)
- Files: ~3600 files (mostly images)

---

## 🚀 Upload to Google Drive

After creating `ai_clean.zip`:

1. Delete old `ai.zip` from Google Drive
2. Upload `ai_clean.zip` to `My Drive/AI_Cancer_Detection/`
3. Rename to `ai.zip` in Drive (or update Colab cell to use `ai_clean.zip`)

---

## ⚠️ Important Notes

### DO NOT Delete:
- `data/` folder - Contains your dataset
- `src/` folder - Contains source code
- `configs/` folder - Contains configuration
- `main.py` - Entry point for training

### Safe to Delete:
- `.venv/` - Virtual environment (recreated in Colab)
- `experiments/` - Old training runs (new ones created in Colab)
- `*.bat` files - Windows scripts (not needed in Colab)
- Debug/test scripts - Only needed for local development

---

## 📋 Quick Cleanup Script

Run this single command to clean everything:

```powershell
cd C:\AI_Lung_Cancer\AI_Cancer_Detection\ai

# One-line cleanup
Remove-Item -Force -ErrorAction SilentlyContinue @("colab_training.ipynb", "check_*.py", "create_*.py", "fix_*.py", "read_*.py", "test_*.py", "*.bat", "data.zip"); Remove-Item -Recurse -Force -ErrorAction SilentlyContinue @(".venv", "experiments")

Write-Host "✅ Cleanup done! Now create ZIP."
```

Then create ZIP:

```powershell
cd ..
Compress-Archive -Path "ai" -DestinationPath "ai_clean.zip" -Force
Write-Host "✅ ZIP created: ai_clean.zip"
```

---

**Ready to upload to Google Drive and start training on GPU! 🚀**
