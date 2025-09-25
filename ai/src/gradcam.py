import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import transforms
import os

class GradCAM:
    """
    Grad-CAM implementation for visualizing important regions in chest X-rays
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        
        # Register hooks
        self.gradients = None
        self.activations = None
        
        # Find target layer if not specified
        if target_layer is None:
            target_layer = self._find_target_layer()
        
        self.target_layer = target_layer
        
        # Register hooks
        self._register_hooks()
    
    def _find_target_layer(self):
        """
        Automatically find the last convolutional layer
        """
        target_layer = None
        
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
        
        if target_layer is None:
            # For models with backbone
            for name, module in self.model.named_modules():
                if 'backbone' in name and isinstance(module, torch.nn.Conv2d):
                    target_layer = module
        
        return target_layer
    
    def _register_hooks(self):
        """
        Register forward and backward hooks
        """
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_gradcam(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input tensor (1, C, H, W)
            class_idx: Target class index (for multi-class)
        
        Returns:
            heatmap: Grad-CAM heatmap
        """
        # Forward pass
        output = self.model(input_tensor)
        
        # For binary classification, use the single output
        if class_idx is None:
            score = output.squeeze()
        else:
            score = output[0, class_idx]
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        score.backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients.detach()
        activations = self.activations.detach()
        
        # Calculate weights (global average pooling of gradients)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1).squeeze()
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize to 0-1
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()
    
    def apply_colormap(self, heatmap, colormap=cv2.COLORMAP_JET):
        """
        Apply colormap to heatmap
        """
        heatmap_uint8 = np.uint8(255 * heatmap)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
        return colored_heatmap
    
    def superimpose_heatmap(self, image, heatmap, alpha=0.4):
        """
        Superimpose heatmap on original image
        
        Args:
            image: Original image (PIL Image or numpy array)
            heatmap: Grad-CAM heatmap
            alpha: Transparency of heatmap overlay
        
        Returns:
            superimposed: Image with heatmap overlay
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure image is RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            original_image = image
        else:
            original_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # Apply colormap
        colored_heatmap = self.apply_colormap(heatmap_resized)
        
        # Normalize original image to 0-1
        if original_image.max() > 1:
            original_image = original_image.astype(np.float32) / 255.0
        
        # Normalize colored heatmap to 0-1
        colored_heatmap = colored_heatmap.astype(np.float32) / 255.0
        
        # Superimpose
        superimposed = (1 - alpha) * original_image + alpha * colored_heatmap
        
        return (superimposed * 255).astype(np.uint8)

class GradCAMVisualizer:
    """
    High-level interface for Grad-CAM visualization
    """
    def __init__(self, model, device, target_layer=None):
        self.model = model
        self.device = device
        self.gradcam = GradCAM(model, target_layer)
        
        # Preprocessing transform (should match training)
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def visualize_image(self, image_path, save_path=None, alpha=0.4):
        """
        Visualize Grad-CAM for a single image
        
        Args:
            image_path: Path to input image
            save_path: Path to save visualization
            alpha: Transparency of heatmap overlay
        
        Returns:
            prediction: Model prediction probability
            heatmap: Grad-CAM heatmap
        """
        # Load and preprocess image
        original_image = Image.open(image_path).convert('RGB')
        input_tensor = self.preprocess(original_image).unsqueeze(0).to(self.device)
        
        # Get model prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            prediction = torch.sigmoid(output).item()
        
        # Generate Grad-CAM
        heatmap = self.gradcam.generate_gradcam(input_tensor)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original X-ray')
        axes[0].axis('off')
        
        # Grad-CAM heatmap
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        # Superimposed image
        superimposed = self.gradcam.superimpose_heatmap(
            np.array(original_image), heatmap, alpha=alpha
        )
        axes[2].imshow(superimposed)
        axes[2].set_title(f'Overlay (Prediction: {prediction:.3f})')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
        
        return prediction, heatmap
    
    def visualize_batch(self, data_loader, num_samples=8, save_dir=None):
        """
        Visualize Grad-CAM for a batch of images
        
        Args:
            data_loader: PyTorch DataLoader
            num_samples: Number of samples to visualize
            save_dir: Directory to save visualizations
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        sample_count = 0
        
        for batch_idx, (images, targets) in enumerate(data_loader):
            if sample_count >= num_samples:
                break
            
            batch_size = images.size(0)
            
            for i in range(min(batch_size, num_samples - sample_count)):
                image = images[i].unsqueeze(0).to(self.device)
                target = targets[i].item()
                
                # Get prediction
                with torch.no_grad():
                    output = self.model(image)
                    prediction = torch.sigmoid(output).item()
                
                # Generate Grad-CAM
                heatmap = self.gradcam.generate_gradcam(image)
                
                # Convert image tensor back to PIL
                # Denormalize
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                
                image_denorm = image.cpu().squeeze() * std + mean
                image_denorm = torch.clamp(image_denorm, 0, 1)
                
                # Convert to PIL
                to_pil = transforms.ToPILImage()
                pil_image = to_pil(image_denorm)
                
                # Create visualization
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                
                # Original image
                axes[0].imshow(pil_image, cmap='gray' if pil_image.mode == 'L' else None)
                axes[0].set_title(f'Original (True: {int(target)})')
                axes[0].axis('off')
                
                # Heatmap
                im = axes[1].imshow(heatmap, cmap='jet')
                axes[1].set_title('Grad-CAM')
                axes[1].axis('off')
                plt.colorbar(im, ax=axes[1], fraction=0.046)
                
                # Superimposed
                superimposed = self.gradcam.superimpose_heatmap(
                    np.array(pil_image), heatmap
                )
                axes[2].imshow(superimposed)
                axes[2].set_title(f'Overlay (Pred: {prediction:.3f})')
                axes[2].axis('off')
                
                plt.tight_layout()
                
                if save_dir:
                    save_path = os.path.join(save_dir, f'sample_{sample_count:03d}.png')
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
                plt.show()
                
                sample_count += 1
                
                if sample_count >= num_samples:
                    break
    
    def analyze_prediction_focus(self, image_path, threshold_percentile=90):
        """
        Analyze what regions the model focuses on for prediction
        
        Args:
            image_path: Path to input image
            threshold_percentile: Percentile threshold for important regions
        
        Returns:
            analysis: Dictionary with analysis results
        """
        # Load and preprocess image
        original_image = Image.open(image_path).convert('RGB')
        input_tensor = self.preprocess(original_image).unsqueeze(0).to(self.device)
        
        # Get prediction and heatmap
        with torch.no_grad():
            output = self.model(input_tensor)
            prediction = torch.sigmoid(output).item()
        
        heatmap = self.gradcam.generate_gradcam(input_tensor)
        
        # Analyze heatmap
        threshold = np.percentile(heatmap.flatten(), threshold_percentile)
        important_regions = heatmap > threshold
        
        # Calculate statistics
        total_pixels = heatmap.size
        important_pixels = np.sum(important_regions)
        focus_percentage = (important_pixels / total_pixels) * 100
        
        # Find center of focus
        y_coords, x_coords = np.where(important_regions)
        if len(y_coords) > 0:
            center_y = np.mean(y_coords) / heatmap.shape[0]  # Normalized
            center_x = np.mean(x_coords) / heatmap.shape[1]  # Normalized
        else:
            center_y, center_x = 0.5, 0.5
        
        analysis = {
            'prediction': prediction,
            'focus_percentage': focus_percentage,
            'center_of_focus': (center_x, center_y),
            'max_attention': np.max(heatmap),
            'mean_attention': np.mean(heatmap),
            'attention_std': np.std(heatmap)
        }
        
        return analysis, heatmap