import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from .utils import save_checkpoint, AverageMeter, EarlyStopping

class Trainer:
    """
    Main trainer class for cancer detection model
    """
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Loss function with class weights
        pos_weight = torch.tensor([config['loss'].get('pos_weight', 1.0)]).to(device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Optimizer
        if config['training']['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay']
            )
        elif config['training']['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=config['training']['learning_rate'],
                momentum=0.9,
                weight_decay=config['training']['weight_decay']
            )
        
        # Learning rate scheduler
        if config['training']['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config['training']['epochs']
            )
        elif config['training']['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        elif config['training']['scheduler'] == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=10
            )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config['training']['early_stopping_patience'],
            min_delta=0.001
        )
        
        # TensorBoard writer
        self.writer = SummaryWriter(config['paths']['logs'])
        
        # Best metrics
        self.best_val_loss = float('inf')
        self.best_val_auc = 0.0
        
    def train_epoch(self, epoch):
        """
        Train for one epoch
        """
        self.model.train()
        
        losses = AverageMeter()
        train_preds = []
        train_targets = []
        
        pbar = tqdm(self.train_loader, desc=f'Train Epoch {epoch}')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data).squeeze()
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Record loss
            losses.update(loss.item(), data.size(0))
            
            # Store predictions and targets for metrics
            with torch.no_grad():
                preds = torch.sigmoid(output).cpu().numpy()
                train_preds.extend(preds)
                train_targets.extend(target.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Calculate metrics
        train_preds = np.array(train_preds)
        train_targets = np.array(train_targets)
        
        train_acc = accuracy_score(train_targets, train_preds > 0.5)
        train_auc = roc_auc_score(train_targets, train_preds) if len(np.unique(train_targets)) > 1 else 0.0
        
        return losses.avg, train_acc, train_auc
    
    def validate(self, epoch):
        """
        Validate the model
        """
        self.model.eval()
        
        losses = AverageMeter()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Val Epoch {epoch}')
            
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data).squeeze()
                loss = self.criterion(output, target)
                
                # Record loss
                losses.update(loss.item(), data.size(0))
                
                # Store predictions and targets
                preds = torch.sigmoid(output).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(target.cpu().numpy())
                
                pbar.set_postfix({'Loss': f'{losses.avg:.4f}'})
        
        # Calculate metrics
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        
        val_acc = accuracy_score(val_targets, val_preds > 0.5)
        val_precision = precision_score(val_targets, val_preds > 0.5, zero_division=0)
        val_recall = recall_score(val_targets, val_preds > 0.5, zero_division=0)
        val_f1 = f1_score(val_targets, val_preds > 0.5, zero_division=0)
        val_auc = roc_auc_score(val_targets, val_preds) if len(np.unique(val_targets)) > 1 else 0.0
        
        return losses.avg, val_acc, val_precision, val_recall, val_f1, val_auc
    
    def train(self):
        """
        Main training loop
        """
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        
        start_time = time.time()
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            # Train for one epoch
            train_loss, train_acc, train_auc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_precision, val_recall, val_f1, val_auc = self.validate(epoch)
            
            # Update learning rate scheduler
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # Log metrics to TensorBoard
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Val', val_acc, epoch)
            self.writer.add_scalar('AUC/Train', train_auc, epoch)
            self.writer.add_scalar('AUC/Val', val_auc, epoch)
            self.writer.add_scalar('Precision/Val', val_precision, epoch)
            self.writer.add_scalar('Recall/Val', val_recall, epoch)
            self.writer.add_scalar('F1/Val', val_f1, epoch)
            self.writer.add_scalar('LearningRate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print epoch results
            print(f'Epoch {epoch}/{self.config["training"]["epochs"]}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}')
            print(f'  Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}')
            print(f'  LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            print('-' * 80)
            
            # Save best model
            is_best = val_auc > self.best_val_auc
            if is_best:
                self.best_val_auc = val_auc
                self.best_val_loss = val_loss
            
            if self.config['training']['save_best_only']:
                if is_best:
                    save_checkpoint({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                        'val_loss': val_loss,
                        'val_auc': val_auc,
                        'config': self.config
                    }, is_best, self.config['paths']['checkpoints'])
            else:
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'val_loss': val_loss,
                    'val_auc': val_auc,
                    'config': self.config
                }, is_best, self.config['paths']['checkpoints'])
            
            # Early stopping
            if self.early_stopping(val_loss):
                print(f"Early stopping triggered after epoch {epoch}")
                break
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time//60:.0f}m {total_time%60:.0f}s")
        print(f"Best validation AUC: {self.best_val_auc:.4f}")
        
        self.writer.close()
        return self.best_val_auc