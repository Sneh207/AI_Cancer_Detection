#!/usr/bin/env python3
"""
Read TensorBoard metrics from events file
"""
import os
import sys

try:
    from tensorboard.backend.event_processing import event_accumulator
    
    log_dir = "experiments/resnet50_training_20251022_092057/logs"
    
    print("Loading TensorBoard events...")
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    print("\n" + "="*70)
    print("TRAINING METRICS FROM TENSORBOARD")
    print("="*70)
    
    # Get available tags
    print("\nAvailable metrics:")
    print(f"  Scalars: {ea.Tags()['scalars']}")
    
    # Extract metrics
    metrics = {}
    
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        metrics[tag] = [(e.step, e.value) for e in events]
    
    # Display metrics by epoch
    print("\n" + "-"*70)
    print("EPOCH-BY-EPOCH METRICS")
    print("-"*70)
    
    # Get max epoch
    max_epoch = 0
    for tag, values in metrics.items():
        if values:
            max_epoch = max(max_epoch, max(v[0] for v in values))
    
    for epoch in range(1, max_epoch + 1):
        print(f"\nEpoch {epoch}:")
        
        # Training metrics
        if 'Loss/Train' in metrics:
            train_loss = [v[1] for v in metrics['Loss/Train'] if v[0] == epoch]
            if train_loss:
                print(f"  Train Loss: {train_loss[0]:.4f}")
        
        if 'Accuracy/Train' in metrics:
            train_acc = [v[1] for v in metrics['Accuracy/Train'] if v[0] == epoch]
            if train_acc:
                print(f"  Train Accuracy: {train_acc[0]:.4f} ({train_acc[0]*100:.2f}%)")
        
        if 'AUC/Train' in metrics:
            train_auc = [v[1] for v in metrics['AUC/Train'] if v[0] == epoch]
            if train_auc:
                print(f"  Train AUC: {train_auc[0]:.4f}")
        
        # Validation metrics
        if 'Loss/Val' in metrics:
            val_loss = [v[1] for v in metrics['Loss/Val'] if v[0] == epoch]
            if val_loss:
                print(f"  Val Loss: {val_loss[0]:.4f}")
        
        if 'Accuracy/Val' in metrics:
            val_acc = [v[1] for v in metrics['Accuracy/Val'] if v[0] == epoch]
            if val_acc:
                print(f"  Val Accuracy: {val_acc[0]:.4f} ({val_acc[0]*100:.2f}%)")
        
        if 'AUC/Val' in metrics:
            val_auc = [v[1] for v in metrics['AUC/Val'] if v[0] == epoch]
            if val_auc:
                print(f"  Val AUC: {val_auc[0]:.4f}")
        
        if 'Precision/Val' in metrics:
            val_prec = [v[1] for v in metrics['Precision/Val'] if v[0] == epoch]
            if val_prec:
                print(f"  Val Precision: {val_prec[0]:.4f}")
        
        if 'Recall/Val' in metrics:
            val_rec = [v[1] for v in metrics['Recall/Val'] if v[0] == epoch]
            if val_rec:
                print(f"  Val Recall: {val_rec[0]:.4f}")
        
        if 'F1/Val' in metrics:
            val_f1 = [v[1] for v in metrics['F1/Val'] if v[0] == epoch]
            if val_f1:
                print(f"  Val F1: {val_f1[0]:.4f}")
        
        if 'LearningRate' in metrics:
            lr = [v[1] for v in metrics['LearningRate'] if v[0] == epoch]
            if lr:
                print(f"  Learning Rate: {lr[0]:.6f}")
    
    print("\n" + "="*70)
    print(f"SUMMARY: {max_epoch} epochs completed")
    print("="*70)
    
except ImportError:
    print("TensorBoard not installed. Installing...")
    os.system("pip install tensorboard")
    print("\nPlease run this script again after installation.")
    sys.exit(1)
    
except Exception as e:
    print(f"Error reading TensorBoard events: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
