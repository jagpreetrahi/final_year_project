"""
Training Utility Functions

Helper functions for training, validation, and evaluation.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple


def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion,
    optimizer,
    device: torch.device,
    gradient_clip: float = 1.0
) -> Tuple[float, float]:
    """
    Train model for one epoch
    
    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device
        gradient_clip: Gradient clipping value
        
    Returns:
        Average loss, Average accuracy
    """
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch in tqdm(train_loader, desc="Training", leave=False):
        # Move data to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        logits = model(input_ids, attention_mask, images)
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = 100.0 * correct / total
    
    return avg_loss, avg_acc


def validate(
    model: nn.Module,
    val_loader,
    criterion,
    device: torch.device
) -> Tuple[float, float]:
    """
    Validate model
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device
        
    Returns:
        Average loss, Average accuracy
    """
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask, images)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    avg_acc = 100.0 * correct / total
    
    return avg_loss, avg_acc


def test(
    model: nn.Module,
    test_loader,
    device: torch.device
) -> Dict:
    """
    Test model and return detailed metrics
    
    Args:
        model: Model to test
        test_loader: Test data loader
        device: Device
        
    Returns:
        Dictionary with metrics
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask, images)
            _, predicted = torch.max(logits, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    accuracy = 100.0 * np.mean(all_predictions == all_labels)
    
    results = {
        'accuracy': accuracy,
        'predictions': all_predictions,
        'labels': all_labels
    }
    
    return results


def save_checkpoint(
    model: nn.Module,
    optimizer,
    epoch: int,
    metrics: Dict,
    filepath: str
):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Training metrics
        filepath: Where to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    torch.save(checkpoint, filepath)
    print(f"✓ Checkpoint saved: {filepath}")


def load_checkpoint(
    model: nn.Module,
    optimizer,
    filepath: str,
    device: torch.device
) -> Dict:
    """
    Load model checkpoint
    
    Args:
        model: Model to load into
        optimizer: Optimizer to load into
        filepath: Checkpoint file
        device: Device
        
    Returns:
        Metrics from checkpoint
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✓ Checkpoint loaded: {filepath}")
    print(f"  Epoch: {checkpoint['epoch']}")
    
    return checkpoint['metrics']


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        Args:
            patience: How many epochs to wait
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        Check if should stop
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


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


def count_parameters(model: nn.Module) -> int:
    """
    Count total trainable parameters
    
    Args:
        model: Model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module):
    """
    Print model architecture summary
    
    Args:
        model: Model
    """
    print("\n" + "="*70)
    print("MODEL SUMMARY")
    print("="*70)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("="*70 + "\n")


if __name__ == "__main__":
    print("Train Utils module loaded successfully!")
    print("\nAvailable functions:")
    print("  - train_one_epoch()")
    print("  - validate()")
    print("  - test()")
    print("  - save_checkpoint()")
    print("  - load_checkpoint()")
    print("  - EarlyStopping")
    print("  - AverageMeter")
    print("  - count_parameters()")
    print("  - print_model_summary()")