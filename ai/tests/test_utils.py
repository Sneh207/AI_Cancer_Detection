"""
Unit tests for utility functions
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
import tempfile
import shutil
import torch
import numpy as np
import yaml
from src.utils import (
    AverageMeter, EarlyStopping, save_checkpoint, load_checkpoint,
    load_config, save_config, seed_everything, calculate_class_weights,
    count_parameters
)
from src.models import CustomCNN


class TestAverageMeter(unittest.TestCase):
    """Test AverageMeter utility"""
    
    def test_initialization(self):
        """Test AverageMeter initialization"""
        meter = AverageMeter()
        self.assertEqual(meter.val, 0)
        self.assertEqual(meter.avg, 0)
        self.assertEqual(meter.sum, 0)
        self.assertEqual(meter.count, 0)
    
    def test_update(self):
        """Test AverageMeter update"""
        meter = AverageMeter()
        
        # Single update
        meter.update(5.0)
        self.assertEqual(meter.val, 5.0)
        self.assertEqual(meter.avg, 5.0)
        self.assertEqual(meter.sum, 5.0)
        self.assertEqual(meter.count, 1)
        
        # Multiple updates
        meter.update(3.0)
        self.assertEqual(meter.val, 3.0)
        self.assertEqual(meter.avg, 4.0)  # (5 + 3) / 2
        self.assertEqual(meter.sum, 8.0)
        self.assertEqual(meter.count, 2)
    
    def test_update_with_n(self):
        """Test AverageMeter update with n parameter"""
        meter = AverageMeter()
        
        meter.update(2.0, n=3)
        self.assertEqual(meter.val, 2.0)
        self.assertEqual(meter.avg, 2.0)
        self.assertEqual(meter.sum, 6.0)  # 2.0 * 3
        self.assertEqual(meter.count, 3)
    
    def test_reset(self):
        """Test AverageMeter reset"""
        meter = AverageMeter()
        meter.update(5.0)
        meter.reset()
        
        self.assertEqual(meter.val, 0)
        self.assertEqual(meter.avg, 0)
        self.assertEqual(meter.sum, 0)
        self.assertEqual(meter.count, 0)


class TestEarlyStopping(unittest.TestCase):
    """Test EarlyStopping utility"""
    
    def test_initialization(self):
        """Test EarlyStopping initialization"""
        early_stopping = EarlyStopping(patience=5, min_delta=0.01)
        self.assertEqual(early_stopping.patience, 5)
        self.assertEqual(early_stopping.min_delta, 0.01)
        self.assertEqual(early_stopping.counter, 0)
        self.assertEqual(early_stopping.best_loss, float('inf'))
    
    def test_improvement(self):
        """Test early stopping with improvement"""
        early_stopping = EarlyStopping(patience=3, min_delta=0.01)
        
        # First loss (improvement)
        should_stop = early_stopping(1.0)
        self.assertFalse(should_stop)
        self.assertEqual(early_stopping.counter, 0)
        self.assertEqual(early_stopping.best_loss, 1.0)
        
        # Better loss (improvement)
        should_stop = early_stopping(0.8)
        self.assertFalse(should_stop)
        self.assertEqual(early_stopping.counter, 0)
        self.assertEqual(early_stopping.best_loss, 0.8)
    
    def test_no_improvement(self):
        """Test early stopping without improvement"""
        early_stopping = EarlyStopping(patience=2, min_delta=0.01)
        
        # Set initial best loss
        early_stopping(1.0)
        
        # No improvement
        should_stop = early_stopping(1.0)
        self.assertFalse(should_stop)
        self.assertEqual(early_stopping.counter, 1)
        
        # Still no improvement - should trigger stopping
        should_stop = early_stopping(1.1)
        self.assertTrue(should_stop)
        self.assertEqual(early_stopping.counter, 2)


class TestCheckpointUtils(unittest.TestCase):
    """Test checkpoint save/load utilities"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.model = CustomCNN(num_classes=1, dropout=0.5)
        self.optimizer = torch.optim.Adam(self.model.parameters())
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_save_checkpoint(self):
        """Test checkpoint saving"""
        state = {
            'epoch': 10,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': 0.5,
            'val_auc': 0.85
        }
        
        save_checkpoint(state, is_best=True, checkpoint_dir=self.temp_dir)
        
        # Check that files are created
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'checkpoint.pth')))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'best_model.pth')))
    
    def test_load_checkpoint(self):
        """Test checkpoint loading"""
        # First save a checkpoint
        state = {
            'epoch': 15,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': 0.3,
            'val_auc': 0.90
        }
        
        checkpoint_path = os.path.join(self.temp_dir, 'test_checkpoint.pth')
        torch.save(state, checkpoint_path)
        
        # Create new model and optimizer
        new_model = CustomCNN(num_classes=1, dropout=0.5)
        new_optimizer = torch.optim.Adam(new_model.parameters())
        
        # Load checkpoint
        epoch, best_loss = load_checkpoint(
            checkpoint_path, new_model, new_optimizer
        )
        
        self.assertEqual(epoch, 15)
        self.assertEqual(best_loss, 0.3)


class TestConfigUtils(unittest.TestCase):
    """Test configuration utilities"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = {
            'data': {
                'batch_size': 32,
                'image_size': 224
            },
            'model': {
                'architecture': 'resnet50',
                'pretrained': True
            },
            'training': {
                'epochs': 100,
                'learning_rate': 0.001
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_save_load_config(self):
        """Test config save and load"""
        config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        
        # Save config
        save_config(self.test_config, config_path)
        self.assertTrue(os.path.exists(config_path))
        
        # Load config
        loaded_config = load_config(config_path)
        self.assertEqual(loaded_config, self.test_config)
    
    def test_load_nonexistent_config(self):
        """Test loading non-existent config file"""
        with self.assertRaises(FileNotFoundError):
            load_config('nonexistent_config.yaml')


class TestUtilityFunctions(unittest.TestCase):
    """Test miscellaneous utility functions"""
    
    def test_seed_everything(self):
        """Test random seed setting"""
        seed_everything(42)
        
        # Generate some random numbers
        torch_rand1 = torch.rand(5)
        np_rand1 = np.random.rand(5)
        
        # Reset seed and generate again
        seed_everything(42)
        torch_rand2 = torch.rand(5)
        np_rand2 = np.random.rand(5)
        
        # Should be identical
        self.assertTrue(torch.allclose(torch_rand1, torch_rand2))
        self.assertTrue(np.allclose(np_rand1, np_rand2))
    
    def test_calculate_class_weights(self):
        """Test class weight calculation"""
        # Imbalanced dataset: 80% class 0, 20% class 1
        labels = np.array([0] * 80 + [1] * 20)
        
        class_weights = calculate_class_weights(labels)
        
        self.assertIsInstance(class_weights, dict)
        self.assertIn(0, class_weights)
        self.assertIn(1, class_weights)
        
        # Class 1 should have higher weight (minority class)
        self.assertGreater(class_weights[1], class_weights[0])
    
    def test_count_parameters(self):
        """Test parameter counting"""
        model = CustomCNN(num_classes=1, dropout=0.5)
        
        total_params, trainable_params = count_parameters(model)
        
        self.assertIsInstance(total_params, int)
        self.assertIsInstance(trainable_params, int)
        self.assertGreater(total_params, 0)
        self.assertGreater(trainable_params, 0)
        self.assertLessEqual(trainable_params, total_params)
    
    def test_balanced_dataset(self):
        """Test class weights with balanced dataset"""
        # Balanced dataset: 50% each class
        labels = np.array([0] * 50 + [1] * 50)
        
        class_weights = calculate_class_weights(labels)
        
        # Weights should be approximately equal for balanced dataset
        self.assertAlmostEqual(class_weights[0], class_weights[1], places=2)


if __name__ == '__main__':
    unittest.main()