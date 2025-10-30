#!/usr/bin/env python3
"""
Check what architecture is in the checkpoint
"""
import torch

checkpoint_path = "experiments/resnet50_training_20251022_092057/checkpoints/best_model.pth"

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("Checkpoint keys:", checkpoint.keys())
print("\nModel state dict keys (first 10):")
for i, key in enumerate(list(checkpoint['model_state_dict'].keys())[:10]):
    print(f"  {key}")

if 'config' in checkpoint:
    print("\nConfig architecture:", checkpoint['config']['model']['architecture'])
else:
    print("\nNo config in checkpoint")

# Check if it's CustomCNN or ResNet
state_keys = list(checkpoint['model_state_dict'].keys())
if 'conv1.weight' in state_keys:
    print("\n✅ This is a CustomCNN model")
elif 'backbone.0.weight' in state_keys:
    print("\n✅ This is a ResNet model")
else:
    print("\n⚠️ Unknown architecture")
