#!/usr/bin/env python3
"""
Setup Verification Script for AI Lung Cancer Detection
Checks if all required components are properly configured
"""

import os
import sys
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def print_status(item, status, details=""):
    """Print status with color coding"""
    symbols = {
        'ok': '✅',
        'warning': '⚠️',
        'error': '❌',
        'info': 'ℹ️'
    }
    symbol = symbols.get(status, '•')
    print(f"{symbol} {item}")
    if details:
        print(f"   → {details}")

def check_python_packages():
    """Check if required Python packages are installed"""
    print_header("Python Dependencies")
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'scikit-learn': 'Scikit-learn',
        'Pillow': 'Pillow (PIL)',
        'yaml': 'PyYAML'
    }
    
    all_installed = True
    for package, name in required_packages.items():
        try:
            __import__(package)
            print_status(f"{name}", 'ok', f"'{package}' installed")
        except ImportError:
            print_status(f"{name}", 'error', f"'{package}' NOT installed")
            all_installed = False
    
    if not all_installed:
        print("\n💡 Install missing packages:")
        print("   pip install -r ai/requirements.txt")
    
    return all_installed

def check_directory_structure():
    """Check if required directories exist"""
    print_header("Directory Structure")
    
    base_dir = Path(__file__).parent
    
    required_dirs = {
        'ai': 'AI module directory',
        'ai/configs': 'Configuration files',
        'ai/src': 'Source code',
        'backend': 'Backend server',
        'frontend': 'Frontend application'
    }
    
    all_exist = True
    for dir_path, description in required_dirs.items():
        full_path = base_dir / dir_path
        if full_path.exists():
            print_status(description, 'ok', str(full_path))
        else:
            print_status(description, 'error', f"NOT FOUND: {full_path}")
            all_exist = False
    
    return all_exist

def check_config_files():
    """Check if configuration files exist"""
    print_header("Configuration Files")
    
    base_dir = Path(__file__).parent
    
    config_files = {
        'ai/configs/config.yaml': 'Main configuration',
        'backend/server.js': 'Backend server',
        'backend/package.json': 'Backend dependencies'
    }
    
    all_exist = True
    for file_path, description in config_files.items():
        full_path = base_dir / file_path
        if full_path.exists():
            print_status(description, 'ok', str(full_path))
        else:
            print_status(description, 'error', f"NOT FOUND: {full_path}")
            all_exist = False
    
    return all_exist

def check_model_checkpoint():
    """Check if model checkpoint exists"""
    print_header("Model Checkpoint")
    
    base_dir = Path(__file__).parent
    
    # Check for any checkpoint files
    experiments_dir = base_dir / 'ai' / 'experiments'
    
    if not experiments_dir.exists():
        print_status("Experiments directory", 'warning', "Directory not found")
        print("\n💡 Create experiments directory:")
        print("   mkdir -p ai/experiments")
        return False
    
    # Look for checkpoint files
    checkpoint_files = list(experiments_dir.rglob('*.pth')) + list(experiments_dir.rglob('*.pt'))
    
    if checkpoint_files:
        print_status("Model checkpoints found", 'ok', f"{len(checkpoint_files)} checkpoint(s)")
        for ckpt in checkpoint_files[:5]:  # Show first 5
            print(f"   • {ckpt.relative_to(base_dir)}")
        if len(checkpoint_files) > 5:
            print(f"   ... and {len(checkpoint_files) - 5} more")
        return True
    else:
        print_status("Model checkpoint", 'error', "NO CHECKPOINT FOUND")
        print("\n💡 You need to train a model or download a pre-trained checkpoint:")
        print("   Option 1: Train a model")
        print("      cd ai")
        print("      python main.py train --config configs/config.yaml")
        print("\n   Option 2: Download pre-trained model")
        print("      Place .pth file in: ai/experiments/[name]/checkpoints/")
        return False

def check_dataset():
    """Check if dataset exists"""
    print_header("Dataset")
    
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'ai' / 'data' / 'raw' / 'train_data'
    
    if not data_dir.exists():
        print_status("Training data directory", 'warning', "NOT FOUND")
        print("\n💡 Create data directory and add training images:")
        print("   mkdir -p ai/data/raw/train_data")
        print("   # Then add your chest X-ray images")
        return False
    
    # Count image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(data_dir.rglob(f'*{ext}'))
    
    if image_files:
        print_status("Training images", 'ok', f"{len(image_files)} images found")
        return True
    else:
        print_status("Training images", 'warning', "No images found in data directory")
        print(f"   → Directory exists but is empty: {data_dir}")
        return False

def check_cuda():
    """Check CUDA availability"""
    print_header("GPU/CUDA Support")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print_status("CUDA Available", 'ok', f"{device_count} GPU(s) detected")
            print_status("GPU Device", 'info', device_name)
            
            # Check memory
            props = torch.cuda.get_device_properties(0)
            total_memory = props.total_memory / (1024**3)  # Convert to GB
            print_status("GPU Memory", 'info', f"{total_memory:.1f} GB")
            return True
        else:
            print_status("CUDA Available", 'warning', "No GPU detected - will use CPU")
            print("   → Training will be slower on CPU")
            print("   → Consider using Google Colab or cloud GPU for faster training")
            return False
    except ImportError:
        print_status("PyTorch", 'error', "Not installed - cannot check CUDA")
        return False

def print_summary(checks):
    """Print summary of all checks"""
    print_header("Summary")
    
    total = len(checks)
    passed = sum(checks.values())
    
    print(f"\nChecks passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All checks passed! Your setup is ready.")
        print("\n📝 Next steps:")
        print("   1. If you don't have a model checkpoint, train one:")
        print("      cd ai && python main.py train --config configs/config.yaml")
        print("   2. Start the backend server:")
        print("      cd backend && node server.js")
        print("   3. Start the frontend:")
        print("      cd frontend && npm start")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\n📚 For detailed setup instructions, see:")
        print("   • QUICK_START.md - Quick setup guide")
        print("   • MODEL_SETUP.md - Detailed model setup")

def main():
    """Main function"""
    print("\n" + "🔍 AI Lung Cancer Detection - Setup Verification".center(60))
    
    checks = {
        'Python Packages': check_python_packages(),
        'Directory Structure': check_directory_structure(),
        'Configuration Files': check_config_files(),
        'Model Checkpoint': check_model_checkpoint(),
        'Dataset': check_dataset(),
        'CUDA Support': check_cuda()
    }
    
    print_summary(checks)
    
    # Return exit code
    if all(checks.values()):
        return 0
    else:
        return 1

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup check interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during setup check: {e}")
        sys.exit(1)
