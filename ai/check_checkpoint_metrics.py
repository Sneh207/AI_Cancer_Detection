#!/usr/bin/env python3
"""
Quick script to check training metrics from checkpoint
"""
import torch
import sys

checkpoint_path = "experiments/resnet50_training_20251022_092057/checkpoints/best_model.pth"

try:
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print("\n" + "="*60)
    print("CHECKPOINT INFORMATION")
    print("="*60)
    
    # Basic info
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    
    # Validation metrics
    if 'val_loss' in checkpoint:
        print(f"Validation Loss: {checkpoint['val_loss']:.4f}")
    
    if 'val_auc' in checkpoint:
        print(f"Validation AUC: {checkpoint['val_auc']:.4f}")
    
    # Check for additional metrics
    print("\nAvailable keys in checkpoint:")
    for key in checkpoint.keys():
        if key not in ['model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'config']:
            print(f"  - {key}: {checkpoint[key]}")
    
    # Config info
    if 'config' in checkpoint:
        config = checkpoint['config']
        print("\nTraining Configuration:")
        if 'model' in config:
            print(f"  Model: {config['model'].get('architecture', 'Unknown')}")
        if 'training' in config:
            print(f"  Max Epochs: {config['training'].get('epochs', 'Unknown')}")
            print(f"  Learning Rate: {config['training'].get('learning_rate', 'Unknown')}")
            print(f"  Batch Size: {config['data'].get('batch_size', 'Unknown')}")
    
    print("="*60)
    
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
