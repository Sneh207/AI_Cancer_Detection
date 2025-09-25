#!/usr/bin/env python3
"""
AI-Based Cancer Detection from Chest X-rays
Main entry point for training, evaluation, and inference
Authors: Sneh Gupta and Arpit Bhardwaj
Course: CSET211 - Statistical Machine Learning
"""

import argparse
import os
import sys
import torch
import warnings
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import DataManager
from src.models import get_model, get_model_summary
from src.train import Trainer
from src.evaluate import ModelEvaluator
from src.inference import CancerDetectionInference
from src.utils import (
    load_config, save_config, seed_everything, 
    print_device_info, create_experiment_directory,
    setup_logging
)

warnings.filterwarnings('ignore', category=UserWarning)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="AI-Based Cancer Detection from Chest X-rays",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('mode', choices=['train', 'evaluate', 'inference', 'demo'],
                       help='Mode to run the application')
    
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint for evaluation/inference')
    
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single image for inference')
    
    parser.add_argument('--batch-images', type=str, default=None,
                       help='Path to directory with images for batch inference')
    
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for results')
    
    parser.add_argument('--device', type=str, default=None,
                       choices=['cpu', 'cuda', 'auto'],
                       help='Device to use (auto will select best available)')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Name for the experiment')
    
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations (Grad-CAM)')
    
    parser.add_argument('--threshold', type=float, default=None,
                       help='Classification threshold (overrides config)')
    
    return parser.parse_args()

def setup_device(device_arg):
    """Setup compute device"""
    if device_arg == 'cpu':
        device = torch.device('cpu')
    elif device_arg == 'cuda':
        if not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:  # auto or None
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return device

def train_model(args):
    """Train the cancer detection model"""
    print("="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up experiment directory
    exp_dir = create_experiment_directory(
        'experiments', 
        args.experiment_name or f"{config['model']['architecture']}_training"
    )
    
    # Update paths in config
    config['paths']['checkpoints'] = os.path.join(exp_dir, 'checkpoints')
    config['paths']['logs'] = os.path.join(exp_dir, 'logs')
    config['paths']['results'] = os.path.join(exp_dir, 'results')
    
    # Save config to experiment directory
    save_config(config, os.path.join(exp_dir, 'configs', 'config.yaml'))
    
    # Setup logging
    logger = setup_logging(config['paths']['logs'], 'cancer_detection_training')
    
    # Setup device
    device = setup_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Set random seed
    seed_everything(args.seed)
    logger.info(f"Random seed set to {args.seed}")
    
    # Print device information
    print_device_info()
    
    # Initialize data manager
    logger.info("Initializing data manager...")
    data_manager = DataManager(config)
    
    # Get data loaders
    train_loader, val_loader, test_loader = data_manager.get_data_loaders()
    
    # Calculate positive weight for class imbalance
    pos_weight = data_manager.calculate_pos_weight()
    config['loss']['pos_weight'] = pos_weight
    
    # Initialize model
    logger.info(f"Initializing {config['model']['architecture']} model...")
    model = get_model(config)
    
    # Print model summary
    get_model_summary(model)
    
    # Initialize trainer
    trainer = Trainer(model, train_loader, val_loader, config, device)
    
    # Start training
    logger.info("Starting training process...")
    best_auc = trainer.train()
    
    logger.info(f"Training completed. Best validation AUC: {best_auc:.4f}")
    print(f"\nExperiment results saved to: {exp_dir}")
    
    return exp_dir

def evaluate_model(args):
    """Evaluate trained model"""
    print("="*80)
    print("STARTING EVALUATION")
    print("="*80)
    
    if not args.checkpoint:
        raise ValueError("Checkpoint path required for evaluation. Use --checkpoint argument.")
    
    # Load configuration
    config = load_config(args.config)
    device = setup_device(args.device)
    
    # Set up results directory
    results_dir = args.output or os.path.join('experiments', 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize data manager and get test loader
    data_manager = DataManager(config)
    _, _, test_loader = data_manager.get_data_loaders()
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = get_model(config)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model, test_loader, device, results_dir)
    
    # Evaluate model
    metrics, optimal_threshold = evaluator.evaluate_model(threshold=args.threshold)
    
    # Threshold analysis
    print("\nPerforming threshold analysis...")
    threshold_results = evaluator.evaluate_at_different_thresholds()
    
    print(f"\nEvaluation results saved to: {results_dir}")
    
    return metrics, optimal_threshold

def run_inference(args):
    """Run inference on single image or batch"""
    print("="*80)
    print("STARTING INFERENCE")
    print("="*80)
    
    if not args.checkpoint:
        raise ValueError("Checkpoint path required for inference. Use --checkpoint argument.")
    
    if not args.image and not args.batch_images:
        raise ValueError("Either --image or --batch-images must be specified for inference.")
    
    # Load configuration and initialize inference pipeline
    config = load_config(args.config)
    device = setup_device(args.device)
    
    inference_pipeline = CancerDetectionInference(
        args.checkpoint, args.config, device
    )
    
    # Set custom threshold if provided
    if args.threshold:
        inference_pipeline.threshold = args.threshold
        print(f"Using custom threshold: {args.threshold}")
    
    results_dir = args.output or 'experiments/inference_results'
    os.makedirs(results_dir, exist_ok=True)
    
    if args.image:
        # Single image inference
        print(f"Processing single image: {args.image}")
        
        result = inference_pipeline.predict_single(
            args.image, 
            return_gradcam=args.visualize
        )
        
        print("\nPrediction Results:")
        print(f"  Image: {result['image_path']}")
        print(f"  Prediction: {result['prediction_label']}")
        print(f"  Probability: {result['probability']:.4f}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Inference Time: {result['inference_time']:.3f}s")
        
        # Generate visualization if requested
        if args.visualize:
            viz_path = os.path.join(results_dir, 'gradcam_visualization.png')
            inference_pipeline.visualize_prediction(args.image, viz_path)
            print(f"  Visualization saved: {viz_path}")
        
        # Generate comprehensive report
        report_path = os.path.join(results_dir, 'diagnostic_report.json')
        report = inference_pipeline.generate_report(args.image, report_path)
        print(f"  Report saved: {report_path}")
        
    else:
        # Batch inference
        print(f"Processing batch from directory: {args.batch_images}")
        
        # Get all image files in directory
        image_extensions = {'.jpg', '.jpeg', '.png', '.dcm', '.dicom'}
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(Path(args.batch_images).glob(f"*{ext}"))
            image_paths.extend(Path(args.batch_images).glob(f"*{ext.upper()}"))
        
        image_paths = [str(p) for p in image_paths]
        
        if not image_paths:
            print(f"No images found in {args.batch_images}")
            return
        
        print(f"Found {len(image_paths)} images")
        
        # Run batch prediction
        results = inference_pipeline.predict_batch(image_paths)
        
        # Save results
        import json
        results_file = os.path.join(results_dir, 'batch_predictions.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        cancer_predictions = sum(1 for r in results if r['prediction'] == 1)
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        
        print(f"\nBatch Inference Summary:")
        print(f"  Total images processed: {len(results)}")
        print(f"  Cancer predictions: {cancer_predictions}")
        print(f"  No-cancer predictions: {len(results) - cancer_predictions}")
        print(f"  Average confidence: {avg_confidence:.4f}")
        print(f"  Results saved: {results_file}")

def run_demo(args):
    """Run demo with sample images"""
    print("="*80)
    print("RUNNING DEMO")
    print("="*80)
    
    if not args.checkpoint:
        print("Demo mode requires a trained model checkpoint.")
        print("Please train a model first or provide a checkpoint with --checkpoint")
        return
    
    # Look for sample images in data directory
    sample_dir = 'data/sample_images'
    if not os.path.exists(sample_dir):
        print(f"Creating sample directory: {sample_dir}")
        os.makedirs(sample_dir, exist_ok=True)
        print("Please place some sample chest X-ray images in the sample_images directory")
        return
    
    # Set demo arguments
    args.batch_images = sample_dir
    args.visualize = True
    args.output = 'experiments/demo_results'
    
    print(f"Running demo with images from: {sample_dir}")
    run_inference(args)

def main():
    """Main function"""
    args = parse_arguments()
    
    try:
        if args.mode == 'train':
            train_model(args)
        
        elif args.mode == 'evaluate':
            evaluate_model(args)
        
        elif args.mode == 'inference':
            run_inference(args)
        
        elif args.mode == 'demo':
            run_demo(args)
        
        print("\n" + "="*80)
        print("EXECUTION COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
        sys.exit(1)
    
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()