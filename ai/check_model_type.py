import torch
import os

checkpoint_path = 'experiments/test_run_fixed_20250928_224821/checkpoints/best_model.pth'

print("=" * 60)
print("Checking Model Type")
print("=" * 60)

# Check file size
file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
print(f"\nFile size: {file_size_mb:.2f} MB")

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# Check if it's a dummy model
if 'note' in checkpoint:
    print(f"\n⚠️  WARNING: {checkpoint['note']}")
    print("\n🚨 THIS IS A DUMMY MODEL!")
    print("   It will give random/fixed predictions.")
else:
    print("\n✓ This appears to be a real trained model (no dummy note)")

print(f"\nModel Details:")
print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"  Train Loss: {checkpoint.get('train_loss', 'N/A')}")
print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A')}")
print(f"  Val Accuracy: {checkpoint.get('val_acc', 'N/A')}")
print(f"  Val AUC: {checkpoint.get('val_auc', 'N/A')}")

# Check model state dict
if 'model_state_dict' in checkpoint:
    num_params = len(checkpoint['model_state_dict'])
    print(f"\n  Model parameters: {num_params} layers")
    
    # Check if weights are initialized (not random)
    first_layer_key = list(checkpoint['model_state_dict'].keys())[0]
    first_layer_weights = checkpoint['model_state_dict'][first_layer_key]
    mean_val = first_layer_weights.mean().item()
    std_val = first_layer_weights.std().item()
    
    print(f"  First layer mean: {mean_val:.6f}")
    print(f"  First layer std: {std_val:.6f}")
    
    if abs(mean_val) < 0.001 and abs(std_val - 1.0) < 0.1:
        print("\n  ⚠️  Weights look like random initialization!")
    else:
        print("\n  ✓ Weights appear to be trained")

print("\n" + "=" * 60)
