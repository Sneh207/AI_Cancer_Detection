#!/usr/bin/env python3
"""
Create a dummy checkpoint for testing the application without training data.
This creates a minimal working checkpoint that can be loaded by the inference system.
"""

import torch
import torch.nn as nn
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import get_model
from src.utils import load_config

def create_dummy_checkpoint():
    """Create a dummy checkpoint file for testing"""
    
    print("=" * 60)
    print("Creating Dummy Model Checkpoint")
    print("=" * 60)
    
    # Load config
    config_path = PROJECT_ROOT / 'configs' / 'config.yaml'
    config = load_config(str(config_path))
    
    # Create model
    print("\n1. Initializing model...")
    model = get_model(config)
    print(f"   Model architecture: {config['model']['architecture']}")
    
    # Create checkpoint directory
    checkpoint_dir = PROJECT_ROOT / 'experiments' / 'test_run_fixed_20250928_224821' / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n2. Created checkpoint directory: {checkpoint_dir}")
    
    # Create checkpoint state
    checkpoint_state = {
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': None,
        'scheduler_state_dict': None,
        'train_loss': 0.5,
        'val_loss': 0.5,
        'val_acc': 0.75,
        'val_auc': 0.80,
        'threshold': 0.5,
        'config': config,
        'note': 'This is a DUMMY checkpoint for testing purposes only. Train a real model for actual predictions.'
    }
    
    # Save checkpoint
    checkpoint_path = checkpoint_dir / 'best_model.pth'
    torch.save(checkpoint_state, checkpoint_path)
    print(f"\n3. Saved checkpoint to: {checkpoint_path}")
    
    # Verify checkpoint can be loaded
    print("\n4. Verifying checkpoint...")
    try:
        loaded = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print("   ✓ Checkpoint loads successfully")
        print(f"   ✓ Contains model state dict: {len(loaded['model_state_dict'])} parameters")
        print(f"   ✓ Epoch: {loaded['epoch']}")
        print(f"   ✓ Threshold: {loaded['threshold']}")
    except Exception as e:
        print(f"   ✗ Error loading checkpoint: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ DUMMY CHECKPOINT CREATED SUCCESSFULLY!")
    print("=" * 60)
    print("\n⚠️  IMPORTANT NOTES:")
    print("   • This is a DUMMY model with random weights")
    print("   • It will NOT provide accurate predictions")
    print("   • Use this ONLY for testing the application flow")
    print("   • Train a real model with actual data for production use")
    print("\n📝 Next Steps:")
    print("   1. Start backend: cd backend && node server.js")
    print("   2. Start frontend: cd frontend && npm run dev")
    print("   3. Upload an X-ray image to test the system")
    print("   4. Train a real model when you have training data")
    print("\n" + "=" * 60)
    
    return True

if __name__ == '__main__':
    try:
        success = create_dummy_checkpoint()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error creating checkpoint: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
