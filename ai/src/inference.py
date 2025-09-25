import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
import yaml
import os
import time

from .models import get_model
from .gradcam import GradCAMVisualizer

class CancerDetectionInference:
    """
    Inference class for cancer detection from chest X-rays
    """
    def __init__(self, model_path, config_path=None, device=None):
        """
        Initialize inference pipeline
        
        Args:
            model_path: Path to saved model checkpoint
            config_path: Path to model configuration file
            device: Device to run inference on
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load config
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = checkpoint.get('config', self._default_config())
        
        # Initialize model
        self.model = get_model(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Get threshold from checkpoint or use default
        self.threshold = checkpoint.get('threshold', 0.5)
        
        # Initialize preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((self.config['data']['image_size'], 
                             self.config['data']['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize Grad-CAM
        self.gradcam_viz = GradCAMVisualizer(self.model, self.device)
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Using threshold: {self.threshold:.3f}")
    
    def _default_config(self):
        """Default configuration if not provided"""
        return {
            'data': {'image_size': 224},
            'model': {
                'architecture': 'resnet50',
                'num_classes': 1,
                'pretrained': True,
                'dropout': 0.5
            }
        }
    
    def preprocess_image(self, image_path):
        """
        Preprocess single image for inference
        
        Args:
            image_path: Path to image file
            
        Returns:
            preprocessed_tensor: Preprocessed image tensor
            original_image: Original PIL image
        """
        try:
            # Load image
            original_image = Image.open(image_path).convert('RGB')
            
            # Preprocess
            preprocessed = self.preprocess(original_image)
            preprocessed_tensor = preprocessed.unsqueeze(0).to(self.device)
            
            return preprocessed_tensor, original_image
            
        except Exception as e:
            raise ValueError(f"Error preprocessing image {image_path}: {str(e)}")
    
    def predict_single(self, image_path, return_gradcam=False):
        """
        Predict cancer probability for a single image
        
        Args:
            image_path: Path to chest X-ray image
            return_gradcam: Whether to return Grad-CAM visualization
            
        Returns:
            result: Dictionary containing prediction results
        """
        start_time = time.time()
        
        # Preprocess image
        input_tensor, original_image = self.preprocess_image(image_path)
        
        # Make prediction
        with torch.no_grad():
            logits = self.model(input_tensor)
            probability = torch.sigmoid(logits).item()
            prediction = int(probability > self.threshold)
        
        inference_time = time.time() - start_time
        
        # Prepare result
        result = {
            'image_path': image_path,
            'probability': probability,
            'prediction': prediction,
            'prediction_label': 'Cancer' if prediction == 1 else 'No Cancer',
            'confidence': max(probability, 1 - probability),
            'threshold': self.threshold,
            'inference_time': inference_time
        }
        
        # Generate Grad-CAM if requested
        if return_gradcam:
            try:
                heatmap = self.gradcam_viz.gradcam.generate_gradcam(input_tensor)
                result['gradcam_heatmap'] = heatmap
            except Exception as e:
                print(f"Warning: Could not generate Grad-CAM: {str(e)}")
                result['gradcam_heatmap'] = None
        
        return result
    
    def predict_batch(self, image_paths, batch_size=16):
        """
        Predict cancer probability for multiple images
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing
            
        Returns:
            results: List of prediction results
        """
        results = []
        
        print(f"Processing {len(image_paths)} images...")
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            valid_paths = []
            
            # Preprocess batch
            for path in batch_paths:
                try:
                    tensor, _ = self.preprocess_image(path)
                    batch_tensors.append(tensor.squeeze(0))
                    valid_paths.append(path)
                except Exception as e:
                    print(f"Warning: Skipping {path} due to error: {str(e)}")
                    continue
            
            if not batch_tensors:
                continue
            
            # Stack tensors
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Make predictions
            with torch.no_grad():
                logits = self.model(batch_tensor)
                probabilities = torch.sigmoid(logits).cpu().numpy().flatten()
                predictions = (probabilities > self.threshold).astype(int)
            
            # Store results
            for j, (path, prob, pred) in enumerate(zip(valid_paths, probabilities, predictions)):
                result = {
                    'image_path': path,
                    'probability': float(prob),
                    'prediction': int(pred),
                    'prediction_label': 'Cancer' if pred == 1 else 'No Cancer',
                    'confidence': float(max(prob, 1 - prob)),
                    'threshold': self.threshold
                }
                results.append(result)
        
        return results
    
    def generate_report(self, image_path, save_path=None):
        """
        Generate comprehensive diagnostic report for a single image
        
        Args:
            image_path: Path to chest X-ray image
            save_path: Path to save the report
            
        Returns:
            report: Comprehensive diagnostic report
        """
        # Get prediction with Grad-CAM
        result = self.predict_single(image_path, return_gradcam=True)
        
        # Analyze prediction focus
        analysis, heatmap = self.gradcam_viz.analyze_prediction_focus(image_path)
        
        # Create comprehensive report
        report = {
            'patient_info': {
                'image_path': image_path,
                'analysis_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'model_version': self.config.get('model', {}).get('architecture', 'Unknown')
            },
            'prediction_results': {
                'probability': result['probability'],
                'prediction': result['prediction_label'],
                'confidence': result['confidence'],
                'threshold_used': result['threshold'],
                'inference_time_seconds': result['inference_time']
            },
            'model_attention_analysis': {
                'focus_percentage': analysis['focus_percentage'],
                'center_of_focus_normalized': analysis['center_of_focus'],
                'max_attention_score': analysis['max_attention'],
                'mean_attention_score': analysis['mean_attention'],
                'attention_variability': analysis['attention_std']
            },
            'clinical_interpretation': self._generate_clinical_interpretation(result, analysis),
            'recommendations': self._generate_recommendations(result)
        }
        
        # Save report if requested
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to: {save_path}")
        
        return report
    
    def _generate_clinical_interpretation(self, result, analysis):
        """Generate clinical interpretation of results"""
        interpretation = []
        
        prob = result['probability']
        confidence = result['confidence']
        focus_pct = analysis['focus_percentage']
        
        # Probability interpretation
        if prob >= 0.8:
            interpretation.append("High probability of malignant findings detected.")
        elif prob >= 0.6:
            interpretation.append("Moderate probability of suspicious findings.")
        elif prob >= 0.4:
            interpretation.append("Low to moderate probability of abnormal findings.")
        else:
            interpretation.append("Low probability of malignant findings.")
        
        # Confidence interpretation
        if confidence >= 0.9:
            interpretation.append("Model prediction shows high confidence.")
        elif confidence >= 0.7:
            interpretation.append("Model prediction shows moderate confidence.")
        else:
            interpretation.append("Model prediction shows low confidence - manual review recommended.")
        
        # Focus analysis
        if focus_pct > 15:
            interpretation.append("Model attention is highly concentrated, suggesting specific regions of interest.")
        elif focus_pct > 5:
            interpretation.append("Model attention is moderately focused on specific regions.")
        else:
            interpretation.append("Model attention is diffuse across the image.")
        
        return interpretation
    
    def _generate_recommendations(self, result):
        """Generate clinical recommendations"""
        recommendations = []
        
        prob = result['probability']
        confidence = result['confidence']
        
        if prob >= 0.7:
            recommendations.extend([
                "Immediate radiologist review recommended",
                "Consider additional imaging (CT scan) if clinically indicated",
                "Follow-up with oncology consultation"
            ])
        elif prob >= 0.5:
            recommendations.extend([
                "Radiologist review within 24-48 hours",
                "Consider repeat imaging in 3-6 months",
                "Correlate with clinical symptoms"
            ])
        else:
            recommendations.extend([
                "Routine radiologist review",
                "Standard follow-up protocols",
                "Continue regular screening as appropriate"
            ])
        
        if confidence < 0.7:
            recommendations.append("Low confidence prediction - prioritize human expert review")
        
        recommendations.append("This AI system is intended as a diagnostic aid only - clinical judgment should always take precedence")
        
        return recommendations
    
    def visualize_prediction(self, image_path, save_path=None):
        """
        Create visualization of prediction with Grad-CAM
        
        Args:
            image_path: Path to input image
            save_path: Path to save visualization
            
        Returns:
            prediction_probability: Prediction probability
        """
        prediction, heatmap = self.gradcam_viz.visualize_image(
            image_path, save_path, alpha=0.4
        )
        
        return prediction