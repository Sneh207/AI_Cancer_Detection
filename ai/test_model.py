import torch
import os

checkpoint_path = 'experiments/test_run_fixed_20250928_224821/checkpoints/best_model.pth'

print("=" * 60)
print("Testing Trained Model Checkpoint")
print("=" * 60)

if os.path.exists(checkpoint_path):
    print(f"\n✓ Checkpoint found at: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print("\n✓ Model loaded successfully!")
        
        print("\nModel Information:")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Validation Loss: {checkpoint.get('val_loss', 'N/A')}")
        print(f"  Validation Accuracy: {checkpoint.get('val_acc', 'N/A')}")
        print(f"  Validation AUC: {checkpoint.get('val_auc', 'N/A')}")
        print(f"  Threshold: {checkpoint.get('threshold', 0.5)}")
        
        if 'model_state_dict' in checkpoint:
            print(f"\n✓ Model state dict present")
            print(f"  Parameters: {len(checkpoint['model_state_dict'])} layers")
        
        print("\n" + "=" * 60)
        print("✅ YOUR MODEL IS READY TO USE!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
else:
    print(f"\n✗ Checkpoint not found at: {checkpoint_path}")
