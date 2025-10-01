#!/usr/bin/env python3
"""
Simple test script to verify the cancer detection project works end-to-end
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import yaml
from src.data_loader import DataManager
from src.models import get_model
from src.train import Trainer
from src.utils import print_device_info, seed_everything

def test_project():
    """Test the entire project pipeline"""
    
    print("=" * 60)
    print("CANCER DETECTION PROJECT TEST")
    print("=" * 60)
    
    # Set random seed for reproducibility
    seed_everything(42)
    
    # Print device info
    print_device_info()
    
    # Load configuration
    config_path = 'ai/configs/test_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\n✓ Configuration loaded from {config_path}")
    
    # Setup device
    device = torch.device('cpu')  # Force CPU for testing
    print(f"✓ Using device: {device}")
    
    # Create data manager and load data
    print("\n1. Testing Data Loading...")
    data_manager = DataManager(config)
    train_loader, val_loader, test_loader = data_manager.get_data_loaders()
    print(f"✓ Data loaders created successfully")
    print(f"  - Training batches: {len(train_loader)}")
    print(f"  - Validation batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")
    
    # Test data batch
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"✓ First batch loaded: data shape {data.shape}, target shape {target.shape}")
        break
    
    # Create model
    print("\n2. Testing Model Creation...")
    model = get_model(config)
    model = model.to(device)
    print(f"✓ Model created: {model.__class__.__name__}")
    
    # Test forward pass
    with torch.no_grad():
        output = model(data)
        print(f"✓ Forward pass successful: output shape {output.shape}")
    
    # Test training for 1 epoch
    print("\n3. Testing Training Loop...")
    
    # Update config for minimal training
    config['training']['epochs'] = 1
    config['paths']['checkpoints'] = 'experiments/test_checkpoints'
    config['paths']['logs'] = 'experiments/test_logs'
    config['paths']['results'] = 'experiments/test_results'
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config, device)
    
    # Train for 1 epoch
    try:
        best_auc = trainer.train()
        print(f"✓ Training completed successfully! Best AUC: {best_auc:.4f}")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return False
    
    # Test inference on a single sample
    print("\n4. Testing Inference...")
    model.eval()
    with torch.no_grad():
        sample_data, sample_target = next(iter(test_loader))
        sample_data = sample_data.to(device)
        
        # Get prediction
        output = model(sample_data[:1])  # Just first sample
        prediction = torch.sigmoid(output).item()
        
        print(f"✓ Inference successful")
        print(f"  - Sample prediction: {prediction:.4f}")
        print(f"  - Actual label: {sample_target[0].item()}")
        print(f"  - Predicted class: {'Cancer' if prediction > 0.5 else 'No Cancer'}")
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED! The project is working correctly.")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = test_project()
    if success:
        print("\n🎉 Project validation completed successfully!")
        print("\nYou can now use the following commands:")
        print("1. Training: python ai/main.py train --config ai/configs/config.yaml --experiment-name my_experiment")
        print("2. Evaluation: python ai/main.py evaluate --config ai/configs/config.yaml --checkpoint <path_to_checkpoint>")
        print("3. Inference: python ai/main.py inference --config ai/configs/config.yaml --checkpoint <path_to_checkpoint> --image <path_to_image>")
    else:
        print("\n❌ Project validation failed. Please check the errors above.")
        sys.exit(1)
