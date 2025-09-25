import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import yaml
import json
from datetime import datetime

class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class EarlyStopping:
    """
    Early stopping to terminate training when validation loss stops improving
    """
    def __init__(self, patience=7, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

        return self.counter >= self.patience

def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth'):
    """
    Save model checkpoint
    
    Args:
        state: Dictionary containing model state
        is_best: Whether this is the best model so far
        checkpoint_dir: Directory to save checkpoints
        filename: Checkpoint filename
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(state, best_filepath)
        print(f"Best model saved to {best_filepath}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
    
    Returns:
        epoch: Epoch number from checkpoint
        best_loss: Best validation loss from checkpoint
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_loss = checkpoint.get('val_loss', float('inf'))
    
    print(f"Loaded checkpoint from epoch {epoch} with validation loss {best_loss:.4f}")
    
    return epoch, best_loss

def load_config(config_path):
    """
    Load YAML configuration file
    
    Args:
        config_path: Path to config file
    
    Returns:
        config: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config, save_path):
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        save_path: Path to save config
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

def create_experiment_directory(base_dir, experiment_name=None):
    """
    Create experiment directory with timestamp
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Optional experiment name
    
    Returns:
        experiment_dir: Path to created experiment directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if experiment_name:
        dir_name = f"{experiment_name}_{timestamp}"
    else:
        dir_name = f"experiment_{timestamp}"
    
    experiment_dir = os.path.join(base_dir, dir_name)
    
    # Create subdirectories
    subdirs = ['checkpoints', 'logs', 'results', 'configs']
    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)
    
    print(f"Experiment directory created: {experiment_dir}")
    return experiment_dir

def plot_training_history(history, save_path=None):
    """
    Plot training and validation metrics
    
    Args:
        history: Dictionary with training history
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[0, 1].plot(history['train_acc'], label='Train Accuracy', color='blue')
    axes[0, 1].plot(history['val_acc'], label='Validation Accuracy', color='red')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUC plot
    axes[1, 0].plot(history['train_auc'], label='Train AUC', color='blue')
    axes[1, 0].plot(history['val_auc'], label='Validation AUC', color='red')
    axes[1, 0].set_title('Training and Validation AUC')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate plot
    axes[1, 1].plot(history['learning_rate'], color='green')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    plt.show()

def calculate_class_weights(labels):
    """
    Calculate class weights for handling imbalanced datasets
    
    Args:
        labels: Array of binary labels
    
    Returns:
        class_weights: Dictionary with class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    class_weights = dict(zip(classes, weights))
    
    print(f"Class weights: {class_weights}")
    return class_weights

def visualize_data_distribution(labels, save_path=None):
    """
    Visualize class distribution in dataset
    
    Args:
        labels: Array of labels
        save_path: Path to save plot
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['No Cancer', 'Cancer'], counts, color=['lightblue', 'lightcoral'])
    plt.title('Class Distribution in Dataset')
    plt.ylabel('Number of Samples')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                str(count), ha='center', va='bottom')
    
    # Add percentage labels
    total = sum(counts)
    for i, (bar, count) in enumerate(zip(bars, counts)):
        pct = (count / total) * 100
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                f'{pct:.1f}%', ha='center', va='center', fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_summary_report(metrics, config, save_path=None):
    """
    Create comprehensive summary report
    
    Args:
        metrics: Dictionary with evaluation metrics
        config: Model configuration
        save_path: Path to save report
    
    Returns:
        report: Summary report dictionary
    """
    report = {
        'experiment_info': {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_architecture': config.get('model', {}).get('architecture', 'Unknown'),
            'image_size': config.get('data', {}).get('image_size', 224),
            'batch_size': config.get('data', {}).get('batch_size', 32)
        },
        'training_config': {
            'epochs': config.get('training', {}).get('epochs', 'Unknown'),
            'learning_rate': config.get('training', {}).get('learning_rate', 'Unknown'),
            'optimizer': config.get('training', {}).get('optimizer', 'Unknown'),
            'scheduler': config.get('training', {}).get('scheduler', 'Unknown')
        },
        'performance_metrics': metrics,
        'model_analysis': {
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
    }
    
    # Add analysis based on metrics
    if metrics.get('recall', 0) > 0.9:
        report['model_analysis']['strengths'].append('High sensitivity - good at detecting cancer cases')
    elif metrics.get('recall', 0) < 0.7:
        report['model_analysis']['weaknesses'].append('Low sensitivity - may miss cancer cases')
    
    if metrics.get('precision', 0) > 0.8:
        report['model_analysis']['strengths'].append('High precision - few false positives')
    elif metrics.get('precision', 0) < 0.6:
        report['model_analysis']['weaknesses'].append('Low precision - many false positives')
    
    if metrics.get('roc_auc', 0) > 0.9:
        report['model_analysis']['strengths'].append('Excellent discriminatory ability (AUC > 0.9)')
    elif metrics.get('roc_auc', 0) < 0.8:
        report['model_analysis']['weaknesses'].append('Limited discriminatory ability (AUC < 0.8)')
    
    # Add recommendations
    if metrics.get('recall', 0) < metrics.get('precision', 0):
        report['model_analysis']['recommendations'].append('Consider lowering threshold to improve sensitivity')
    elif metrics.get('precision', 0) < metrics.get('recall', 0):
        report['model_analysis']['recommendations'].append('Consider raising threshold to improve precision')
    
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Summary report saved to: {save_path}")
    
    return report

def setup_logging(log_dir, experiment_name="cancer_detection"):
    """
    Setup logging configuration
    
    Args:
        log_dir: Directory to save logs
        experiment_name: Name of experiment
    """
    import logging
    
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{experiment_name}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(experiment_name)
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger

def count_parameters(model):
    """
    Count the number of parameters in a model
    
    Args:
        model: PyTorch model
    
    Returns:
        total_params: Total number of parameters
        trainable_params: Number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params

def print_model_info(model, input_size=(1, 3, 224, 224)):
    """
    Print detailed model information
    
    Args:
        model: PyTorch model
        input_size: Input tensor size for testing
    """
    total_params, trainable_params = count_parameters(model)
    
    print("=" * 60)
    print("MODEL INFORMATION")
    print("=" * 60)
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(input_size)
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Input shape: {list(dummy_input.shape)}")
        print(f"Output shape: {list(output.shape)}")
        print("✓ Forward pass successful")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
    
    print("=" * 60)

def seed_everything(seed=42):
    """
    Seed all random number generators for reproducibility
    
    Args:
        seed: Random seed
    """
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")

def get_device_info():
    """
    Get information about available computing devices
    
    Returns:
        device_info: Dictionary with device information
    """
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    if torch.cuda.is_available():
        device_info['cuda_device_name'] = torch.cuda.get_device_name(0)
        device_info['cuda_memory_total'] = torch.cuda.get_device_properties(0).total_memory
        device_info['cuda_memory_cached'] = torch.cuda.memory_cached(0)
        device_info['cuda_memory_allocated'] = torch.cuda.memory_allocated(0)
    
    return device_info

def print_device_info():
    """Print device information"""
    info = get_device_info()
    
    print("=" * 50)
    print("DEVICE INFORMATION")
    print("=" * 50)
    print(f"CUDA Available: {info['cuda_available']}")
    
    if info['cuda_available']:
        print(f"CUDA Device Count: {info['cuda_device_count']}")
        print(f"CUDA Device Name: {info['cuda_device_name']}")
        print(f"Total Memory: {info['cuda_memory_total'] / 1024**3:.1f} GB")
        print(f"Cached Memory: {info['cuda_memory_cached'] / 1024**3:.1f} GB")
        print(f"Allocated Memory: {info['cuda_memory_allocated'] / 1024**3:.1f} GB")
    
    print(f"Using Device: {info['current_device']}")
    print("=" * 50)