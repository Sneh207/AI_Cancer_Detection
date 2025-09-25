"""
Unit tests for data loading and preprocessing
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
import tempfile
import shutil
import torch
import numpy as np
import pandas as pd
from PIL import Image
from src.data_loader import ChestXrayDataset, DataManager, get_transforms


class TestChestXrayDataset(unittest.TestCase):
    """Test ChestXrayDataset class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directory for test images
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample images
        self.image_paths = []
        self.labels = []
        
        for i in range(5):
            # Create a dummy image
            img = Image.new('RGB', (256, 256), color='black')
            img_path = os.path.join(self.temp_dir, f'test_image_{i}.png')
            img.save(img_path)
            
            self.image_paths.append(img_path)
            self.labels.append(i % 2)  # Alternate between 0 and 1
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_dataset_length(self):
        """Test dataset length"""
        dataset = ChestXrayDataset(
            self.image_paths, 
            self.labels, 
            image_size=224
        )
        self.assertEqual(len(dataset), 5)
    
    def test_dataset_getitem(self):
        """Test dataset item retrieval"""
        transform = get_transforms(224, is_train=False)
        dataset = ChestXrayDataset(
            self.image_paths, 
            self.labels, 
            transform=transform,
            image_size=224
        )
        
        image, label = dataset[0]
        
        # Check types
        self.assertIsInstance(image, torch.Tensor)
        self.assertIsInstance(label, torch.Tensor)
        
        # Check shapes
        self.assertEqual(image.shape, (3, 224, 224))
        self.assertEqual(label.shape, ())
        
        # Check value ranges
        self.assertTrue(torch.all(image >= -3))  # Normalized values
        self.assertTrue(torch.all(image <= 3))
        self.assertIn(label.item(), [0.0, 1.0])
    
    def test_dataset_with_invalid_image(self):
        """Test dataset handling of invalid image paths"""
        invalid_paths = self.image_paths + ['invalid_path.jpg']
        invalid_labels = self.labels + [1]
        
        dataset = ChestXrayDataset(
            invalid_paths, 
            invalid_labels, 
            image_size=224
        )
        
        # Should handle invalid path gracefully
        image, label = dataset[-1]  # Last item with invalid path
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.shape, (3, 224, 224))


class TestTransforms(unittest.TestCase):
    """Test data transformations"""
    
    def setUp(self):
        """Set up test image"""
        self.test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    def test_train_transforms(self):
        """Test training transforms"""
        transform = get_transforms(224, is_train=True)
        
        transformed = transform(image=self.test_image)
        image = transformed['image']
        
        # Check output type and shape
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.shape, (3, 224, 224))
        
        # Check normalization (values should be roughly in range [-2, 2])
        self.assertTrue(torch.all(image >= -3))
        self.assertTrue(torch.all(image <= 3))
    
    def test_val_transforms(self):
        """Test validation transforms"""
        transform = get_transforms(224, is_train=False)
        
        transformed = transform(image=self.test_image)
        image = transformed['image']
        
        # Check output type and shape
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.shape, (3, 224, 224))
    
    def test_different_image_sizes(self):
        """Test transforms with different image sizes"""
        sizes = [128, 256, 512]
        
        for size in sizes:
            with self.subTest(size=size):
                transform = get_transforms(size, is_train=False)
                transformed = transform(image=self.test_image)
                image = transformed['image']
                
                self.assertEqual(image.shape, (3, size, size))


class TestDataManager(unittest.TestCase):
    """Test DataManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directory structure
        self.temp_dir = tempfile.mkdtemp()
        self.images_dir = os.path.join(self.temp_dir, 'images')
        os.makedirs(self.images_dir)
        
        # Create sample CSV file
        self.csv_file = os.path.join(self.temp_dir, 'labels.csv')
        
        # Create sample images and CSV data
        csv_data = []
        for i in range(10):
            img_name = f'test_{i:03d}.png'
            img_path = os.path.join(self.images_dir, img_name)
            
            # Create dummy image
            img = Image.new('RGB', (256, 256), color='black')
            img.save(img_path)
            
            # Create CSV entry
            if i < 3:
                finding = 'Mass'  # Cancer case
            elif i < 6:
                finding = 'Nodule'  # Cancer case
            else:
                finding = 'No Finding'  # Normal case
            
            csv_data.append({
                'Image Index': img_name,
                'Finding Labels': finding
            })
        
        # Save CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(self.csv_file, index=False)
        
        # Create config
        self.config = {
            'data': {
                'dataset_path': self.images_dir,
                'labels_file': self.csv_file,
                'image_size': 224,
                'batch_size': 4,
                'num_workers': 0,  # Set to 0 for testing
                'train_split': 0.6,
                'val_split': 0.2,
                'test_split': 0.2
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_prepare_labels(self):
        """Test label preparation"""
        data_manager = DataManager(self.config)
        image_paths, labels = data_manager.prepare_labels()
        
        # Check that we have the right number of samples
        self.assertEqual(len(image_paths), 10)
        self.assertEqual(len(labels), 10)
        
        # Check that cancer labels are correct
        cancer_count = sum(labels)
        self.assertEqual(cancer_count, 6)  # 3 Mass + 3 Nodule
    
    def test_split_data(self):
        """Test data splitting"""
        data_manager = DataManager(self.config)
        image_paths, labels = data_manager.prepare_labels()
        
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_manager.split_data(
            image_paths, labels
        )
        
        # Check splits add up to total
        total_samples = len(X_train) + len(X_val) + len(X_test)
        self.assertEqual(total_samples, len(image_paths))
        
        # Check label consistency
        total_labels = len(y_train) + len(y_val) + len(y_test)
        self.assertEqual(total_labels, len(labels))
    
    def test_calculate_pos_weight(self):
        """Test positive weight calculation"""
        data_manager = DataManager(self.config)
        pos_weight = data_manager.calculate_pos_weight()
        
        # Should be > 0
        self.assertGreater(pos_weight, 0)
        self.assertIsInstance(pos_weight, float)
    
    def test_get_data_loaders(self):
        """Test data loader creation"""
        data_manager = DataManager(self.config)
        train_loader, val_loader, test_loader = data_manager.get_data_loaders()
        
        # Check that loaders are created
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)
        
        # Test one batch from train loader
        batch = next(iter(train_loader))
        images, labels = batch
        
        self.assertIsInstance(images, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)
        self.assertEqual(len(images.shape), 4)  # Batch dimension
        self.assertEqual(images.shape[1:], (3, 224, 224))


if __name__ == '__main__':
    unittest.main()