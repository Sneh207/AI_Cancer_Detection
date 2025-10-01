"""
AI-Based Cancer Detection from Chest X-rays
Source package initialization

Authors: Sneh Gupta and Arpit Bhardwaj
Course: CSET211 - Statistical Machine Learning
"""

__version__ = "1.0.0"
__authors__ = ["Sneh Gupta", "Arpit Bhardwaj"]
__course__ = "CSET211 - Statistical Machine Learning"

# Core functionality imports (match actual module layout under src/)
from .data_loader import ChestXrayDataset, DataManager, get_transforms
from .models import (
    get_model,
    CustomCNN,
    ResNetModel,
    DenseNetModel,
    EfficientNetModel,
)
from .train import Trainer
from .evaluate import ModelEvaluator
from .gradcam import GradCAM, GradCAMVisualizer
from .inference import CancerDetectionInference
from .utils import (
    AverageMeter,
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
    load_config,
    save_config,
    seed_everything,
    print_device_info,
)

__all__ = [
    # Data handling
    'ChestXrayDataset',
    'DataManager', 
    'get_transforms',
    
    # Models
    'get_model',
    'CustomCNN',
    'ResNetModel', 
    'DenseNetModel',
    'EfficientNetModel',
    
    # Training and evaluation
    'Trainer',
    'ModelEvaluator',
    
    # Explainability
    'GradCAM',
    'GradCAMVisualizer',
    
    # Inference
    'CancerDetectionInference',
    
    # Utilities
    'AverageMeter',
    'EarlyStopping',
    'save_checkpoint',
    'load_checkpoint',
    'load_config',
    'save_config',
    'seed_everything',
    'print_device_info',
]
# Authors: Sneh Gupta and Arpit Bhardwaj
# Course: CSET211 - Statistical Machine Learning
# """

# __version__ = "1.0.0"
# __authors__ = ["Sneh Gupta", "Arpit Bhardwaj"]
# __course__ = "CSET211 - Statistical Machine Learning"

# # Import main components for easier access
# from .data_loader import ChestXrayDataset, DataManager, get_transforms
# from .models import get_model, CustomCNN, ResNetModel, DenseNetModel, EfficientNetModel
# from .train import Trainer
# from .evaluate import ModelEvaluator
# from .gradcam import GradCAM, GradCAMVisualizer
# from .inference import CancerDetectionInference
# from .utils import (
#     AverageMeter, EarlyStopping, save_checkpoint, load_checkpoint,
#     load_config, save_config, seed_everything, print_device_info
# )

# # Test configuration
# import os
# import sys

# # Add src to path for testing
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# # Test settings
# TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
# TEMP_DIR = os.path.join(os.path.dirname(__file__), 'temp')

# # Ensure test directories exist
# os.makedirs(TEST_DATA_DIR, exist_ok=True)
# os.makedirs(TEMP_DIR, exist_ok=True)

# __all__ = [
#     # Data handling
#     'ChestXrayDataset',
#     'DataManager', 
#     'get_transforms',
    
#     # Models
#     'get_model',
#     'CustomCNN',
#     'ResNetModel', 
#     'DenseNetModel',
#     'EfficientNetModel',
    
#     # Training and evaluation
#     'Trainer',
#     'ModelEvaluator',
    
#     # Explainability
#     'GradCAM',
#     'GradCAMVisualizer',
    
#     # Inference
#     'CancerDetectionInference',
    
#     # Utilities
#     'AverageMeter',
#     'EarlyStopping',
#     'save_checkpoint',
#     'load_checkpoint',
#     'load_config',
#     'save_config',
#     'seed_everything',
#     'print_device_info',
# ]