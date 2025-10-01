import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

class ChestXrayDataset(Dataset):
    """
    Custom dataset for chest X-ray cancer detection
    """
    def __init__(self, image_paths, labels, transform=None, image_size=224):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.image_size = image_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (self.image_size, self.image_size), color='black')
        
        # Convert PIL to numpy array for albumentations
        image = np.array(image)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Default transform
            transform = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            transformed = transform(image=image)
            image = transformed['image']
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return image, label

def get_transforms(image_size=224, is_train=True):
    """
    Get augmentation transforms for training and validation
    """
    if is_train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.2),
            A.Rotate(limit=15, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
            A.OneOf([
                A.GaussNoise(var_limit=0.01),
                A.GaussianBlur(blur_limit=3),
            ], p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

class DataManager:
    """
    Manages data loading and preprocessing for the cancer detection project
    """
    def __init__(self, config):
        self.config = config
        # Base dir is the project root (ai/), which is parent of this file's directory
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        dataset_path = config['data']['dataset_path']
        labels_file = config['data']['labels_file']
        # Resolve to absolute paths if relative
        self.dataset_path = dataset_path if os.path.isabs(dataset_path) else os.path.join(base_dir, dataset_path)
        self.labels_file = labels_file if os.path.isabs(labels_file) else os.path.join(base_dir, labels_file)
        self.image_size = config['data']['image_size']
        self.batch_size = config['data']['batch_size']
        self.num_workers = config['data']['num_workers']
        
    def prepare_labels(self):
        """
        Process NIH ChestX-ray14 labels to binary cancer/no-cancer classification
        """
        # Load the CSV file
        df = pd.read_csv(self.labels_file)
        
        # Define cancer-related findings
        cancer_labels = ['Mass', 'Nodule']
        
        # Create binary labels
        df['cancer'] = df['Finding Labels'].apply(
            lambda x: 1 if any(label in x for label in cancer_labels) else 0
        )
        
        # Build correct image paths - images are in train/ subdirectory
        train_subdir = os.path.join(self.dataset_path, 'train')
        if os.path.exists(train_subdir):
            # Images are in dataset_path/train/
            df['image_path'] = df['Image Index'].apply(
                lambda x: os.path.join(train_subdir, x)
            )
        else:
            # Images are directly in dataset_path
            df['image_path'] = df['Image Index'].apply(
                lambda x: os.path.join(self.dataset_path, x)
            )
        
        # Filter out images that don't exist
        df = df[df['image_path'].apply(os.path.exists)]
        
        print(f"Found {len(df)} valid images")
        print(f"Cancer cases: {df['cancer'].sum()} ({df['cancer'].mean()*100:.1f}%)")
        
        return df['image_path'].values, df['cancer'].values
    
    def split_data(self, image_paths, labels):
        """
        Split data into train, validation, and test sets
        """
        # Check if we have enough samples for stratified splitting
        unique, counts = np.unique(labels, return_counts=True)
        min_class_count = np.min(counts)
        
        test_size = self.config['data']['test_split']
        val_size = self.config['data']['val_split']
        
        # Calculate minimum samples needed for stratified split
        min_test_samples = int(test_size * len(labels))
        min_val_samples = int(val_size * len(labels))
        
        # Use stratify only if we have enough samples in each class
        use_stratify = min_class_count >= max(2, min_test_samples, min_val_samples)
        
        if not use_stratify:
            print(f"Warning: Using random split instead of stratified (min class count: {min_class_count})")
        
        # First split: train+val, test
        X_temp, X_test, y_temp, y_test = train_test_split(
            image_paths, labels, 
            test_size=test_size,
            stratify=labels if use_stratify else None, 
            random_state=42
        )
        
        # Second split: train, val
        val_size_adjusted = val_size / (1 - test_size)
        
        # Check again for validation split
        if use_stratify:
            unique_temp, counts_temp = np.unique(y_temp, return_counts=True)
            min_class_count_temp = np.min(counts_temp)
            use_stratify_val = min_class_count_temp >= max(2, int(val_size_adjusted * len(y_temp)))
        else:
            use_stratify_val = False
            
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            stratify=y_temp if use_stratify_val else None,
            random_state=42
        )
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def get_data_loaders(self):
        """
        Create PyTorch data loaders for train, validation, and test sets
        """
        # Prepare labels
        image_paths, labels = self.prepare_labels()
        
        # Split data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.split_data(image_paths, labels)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Cancer cases in training: {sum(y_train)} ({sum(y_train)/len(y_train)*100:.1f}%)")
        
        # Create datasets
        train_dataset = ChestXrayDataset(
            X_train, y_train, 
            transform=get_transforms(self.image_size, is_train=True),
            image_size=self.image_size
        )
        
        val_dataset = ChestXrayDataset(
            X_val, y_val,
            transform=get_transforms(self.image_size, is_train=False),
            image_size=self.image_size
        )
        
        test_dataset = ChestXrayDataset(
            X_test, y_test,
            transform=get_transforms(self.image_size, is_train=False),
            image_size=self.image_size
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers,
            pin_memory=True, drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def calculate_pos_weight(self):
        """
        Calculate positive weight for handling class imbalance
        """
        _, labels = self.prepare_labels()
        pos_count = np.sum(labels)
        neg_count = len(labels) - pos_count
        pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        print(f"Positive samples: {pos_count}")
        print(f"Negative samples: {neg_count}")
        print(f"Calculated pos_weight: {pos_weight:.2f}")
        
        return pos_weight