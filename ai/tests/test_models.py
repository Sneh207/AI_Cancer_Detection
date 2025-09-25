"""
Unit tests for model architectures
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
import torch
import torch.nn as nn
from src.models import (
    CustomCNN, ResNetModel, DenseNetModel, EfficientNetModel, 
    get_model, count_parameters
)


class TestModels(unittest.TestCase):
    """Test model architectures"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.input_tensor = torch.randn(2, 3, 224, 224)  # Batch size 2
        self.config_base = {
            'model': {
                'num_classes': 1,
                'pretrained': False,  # Use False for faster testing
                'dropout': 0.5
            }
        }
    
    def test_custom_cnn_forward(self):
        """Test CustomCNN forward pass"""
        model = CustomCNN(num_classes=1, dropout=0.5)
        model.eval()
        
        with torch.no_grad():
            output = model(self.input_tensor)
        
        # Check output shape
        self.assertEqual(output.shape, (2, 1))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_resnet_models(self):
        """Test ResNet model variants"""
        architectures = ['resnet18', 'resnet34', 'resnet50']
        
        for arch in architectures:
            with self.subTest(architecture=arch):
                model = ResNetModel(
                    num_classes=1, 
                    pretrained=False, 
                    dropout=0.5, 
                    architecture=arch
                )
                model.eval()
                
                with torch.no_grad():
                    output = model(self.input_tensor)
                
                self.assertEqual(output.shape, (2, 1))
                self.assertFalse(torch.isnan(output).any())
    
    def test_densenet_models(self):
        """Test DenseNet model variants"""
        architectures = ['densenet121', 'densenet161']
        
        for arch in architectures:
            with self.subTest(architecture=arch):
                model = DenseNetModel(
                    num_classes=1, 
                    pretrained=False, 
                    dropout=0.5, 
                    architecture=arch
                )
                model.eval()
                
                with torch.no_grad():
                    output = model(self.input_tensor)
                
                self.assertEqual(output.shape, (2, 1))
                self.assertFalse(torch.isnan(output).any())
    
    def test_efficientnet_models(self):
        """Test EfficientNet model variants"""
        architectures = ['efficientnet_b0']
        
        for arch in architectures:
            with self.subTest(architecture=arch):
                model = EfficientNetModel(
                    num_classes=1, 
                    pretrained=False, 
                    dropout=0.5, 
                    architecture=arch
                )
                model.eval()
                
                with torch.no_grad():
                    output = model(self.input_tensor)
                
                self.assertEqual(output.shape, (2, 1))
                self.assertFalse(torch.isnan(output).any())
    
    def test_get_model_factory(self):
        """Test model factory function"""
        architectures = [
            'custom_cnn', 'resnet18', 'resnet50', 
            'densenet121', 'efficientnet_b0'
        ]
        
        for arch in architectures:
            with self.subTest(architecture=arch):
                config = self.config_base.copy()
                config['model']['architecture'] = arch
                
                model = get_model(config)
                self.assertIsInstance(model, nn.Module)
                
                # Test forward pass
                model.eval()
                with torch.no_grad():
                    output = model(self.input_tensor)
                
                self.assertEqual(output.shape, (2, 1))
    
    def test_invalid_architecture(self):
        """Test invalid architecture handling"""
        config = self.config_base.copy()
        config['model']['architecture'] = 'invalid_arch'
        
        with self.assertRaises(ValueError):
            get_model(config)
    
    def test_parameter_counting(self):
        """Test parameter counting function"""
        model = CustomCNN(num_classes=1, dropout=0.5)
        param_count = count_parameters(model)
        
        self.assertIsInstance(param_count, int)
        self.assertGreater(param_count, 0)
    
    def test_model_training_mode(self):
        """Test model training/evaluation mode switching"""
        model = CustomCNN(num_classes=1, dropout=0.5)
        
        # Test training mode
        model.train()
        self.assertTrue(model.training)
        
        # Test evaluation mode
        model.eval()
        self.assertFalse(model.training)
    
    def test_gradient_flow(self):
        """Test that gradients flow properly"""
        model = CustomCNN(num_classes=1, dropout=0.5)
        model.train()
        
        output = model(self.input_tensor)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are computed
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertFalse(torch.isnan(param.grad).any())


if __name__ == '__main__':
    unittest.main()