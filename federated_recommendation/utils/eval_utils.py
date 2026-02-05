"""
Evaluation Utilities

Functions for computing metrics and visualizing results.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from typing import Dict, List
import os


def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict:
    """
    Compute classification metrics
    
    Args:
        predictions: Predicted labels
        labels: True labels
        
    Returns:
        Dictionary with metrics
    """
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    
    metrics = {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100
    }
    
    return metrics


def compute_per_class_metrics(
    predictions: np.ndarray, 
    labels: np.ndarray,
    num_classes: int
) -> Dict:
    """
    Compute per-class metrics
    
    Args:
        predictions: Predicted labels
        labels: True labels
        num_classes: Number of classes
        
    Returns:
        Dictionary with per-class metrics
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, labels=range(num_classes)
    )
    
    per_class = {
        f'class_{i}': {
            'precision': precision[i] * 100,
            'recall': recall[i] * 100,
            'f1': f1[i] * 100,
            'support': int(support[i])
        }
        for i in range(num_classes)
    }
    
    return per_class


def plot_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    save_path: str = None,
    class_names: List[str] = None
):
    """
    Plot confusion matrix
    
    Args:
        predictions: Predicted labels
        labels: True labels
        save_path: Where to save plot
        class_names: Names of classes
    """
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names if class_names else range(len(cm)),
        yticklabels=class_names if class_names else range(len(cm))
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved: {save_path}")
    
    plt.close()


def plot_training_history(
    metrics: Dict,
    save_path: str = None
):
    """
    Plot training and validation metrics over time
    
    Args:
        metrics: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Where to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    if 'train_loss' in metrics and 'val_loss' in metrics:
        epochs = range(1, len(metrics['train_loss']) + 1)
        ax1.plot(epochs, metrics['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, metrics['val_loss'], 'r-', label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    if 'train_acc' in metrics and 'val_acc' in metrics:
        epochs = range(1, len(metrics['train_acc']) + 1)
        ax2.plot(epochs, metrics['train_acc'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, metrics['val_acc'], 'r-', label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training history saved: {save_path}")
    
    plt.close()


def plot_federated_metrics(
    metrics: Dict,
    save_path: str = None
):
    """
    Plot federated learning metrics
    
    Args:
        metrics: Dictionary with FL metrics
        save_path: Where to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Test accuracy over rounds
    if 'test_acc' in metrics:
        rounds = range(1, len(metrics['test_acc']) + 1)
        axes[0, 0].plot(rounds, metrics['test_acc'], 'g-', marker='o')
        axes[0, 0].set_xlabel('FL Round')
        axes[0, 0].set_ylabel('Test Accuracy (%)')
        axes[0, 0].set_title('Global Model Test Accuracy')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Training loss over rounds
    if 'train_loss' in metrics:
        rounds = range(1, len(metrics['train_loss']) + 1)
        axes[0, 1].plot(rounds, metrics['train_loss'], 'b-', marker='o')
        axes[0, 1].set_xlabel('FL Round')
        axes[0, 1].set_ylabel('Average Loss')
        axes[0, 1].set_title('Average Client Training Loss')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Test loss over rounds
    if 'test_loss' in metrics:
        rounds = range(1, len(metrics['test_loss']) + 1)
        axes[1, 0].plot(rounds, metrics['test_loss'], 'r-', marker='o')
        axes[1, 0].set_xlabel('FL Round')
        axes[1, 0].set_ylabel('Test Loss')
        axes[1, 0].set_title('Global Model Test Loss')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Client accuracy distribution (if available)
    if 'train_acc' in metrics:
        rounds = range(1, len(metrics['train_acc']) + 1)
        axes[1, 1].plot(rounds, metrics['train_acc'], 'm-', marker='o')
        axes[1, 1].set_xlabel('FL Round')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].set_title('Average Client Training Accuracy')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ FL metrics saved: {save_path}")
    
    plt.close()


def plot_cluster_comparison(
    cluster_metrics: List[float],
    final_accuracy: float,
    save_path: str = None
):
    """
    Plot comparison of cluster accuracies
    
    Args:
        cluster_metrics: List of cluster accuracies
        final_accuracy: Final global model accuracy
        save_path: Where to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cluster_ids = [f'Cluster {i}' for i in range(len(cluster_metrics))]
    cluster_ids.append('Global')
    
    all_metrics = cluster_metrics + [final_accuracy]
    
    colors = ['skyblue'] * len(cluster_metrics) + ['orange']
    
    bars = ax.bar(cluster_ids, all_metrics, color=colors, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.2f}%',
            ha='center',
            va='bottom'
        )
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Cluster vs Global Model Accuracy')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Cluster comparison saved: {save_path}")
    
    plt.close()


def plot_ifca_cluster_evolution(
    cluster_sizes_history: List[List[int]],
    save_path: str = None
):
    """
    Plot how cluster sizes evolve over IFCA rounds
    
    Args:
        cluster_sizes_history: List of cluster sizes per round
        save_path: Where to save plot
    """
    num_clusters = len(cluster_sizes_history[0])
    num_rounds = len(cluster_sizes_history)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rounds = range(1, num_rounds + 1)
    
    for cluster_id in range(num_clusters):
        sizes = [round_sizes[cluster_id] for round_sizes in cluster_sizes_history]
        ax.plot(rounds, sizes, marker='o', label=f'Cluster {cluster_id}')
    
    ax.set_xlabel('IFCA Round')
    ax.set_ylabel('Number of Clients')
    ax.set_title('Cluster Size Evolution (IFCA)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ IFCA evolution saved: {save_path}")
    
    plt.close()


def print_evaluation_report(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: List[str] = None
):
    """
    Print detailed evaluation report
    
    Args:
        predictions: Predicted labels
        labels: True labels
        class_names: Names of classes
    """
    print("\n" + "="*70)
    print("EVALUATION REPORT")
    print("="*70)
    
    # Overall metrics
    metrics = compute_metrics(predictions, labels)
    print(f"\n📊 Overall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"  Precision: {metrics['precision']:.2f}%")
    print(f"  Recall:    {metrics['recall']:.2f}%")
    print(f"  F1-Score:  {metrics['f1']:.2f}%")
    
    # Classification report
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(
        labels, 
        predictions, 
        target_names=class_names,
        digits=2
    ))
    
    print("="*70 + "\n")


def compare_methods(results: Dict, save_path: str = None):
    """
    Compare different methods (Standard FL vs Clustered FL vs IFCA)
    
    Args:
        results: Dictionary with method names and accuracies
        save_path: Where to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(results.keys())
    accuracies = list(results.values())
    
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    bars = ax.bar(methods, accuracies, color=colors, edgecolor='black')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.2f}%',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )
    
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Method Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Method comparison saved: {save_path}")
    
    plt.close()


def create_results_summary(
    metrics: Dict,
    save_path: str
):
    """
    Create a text file with results summary
    
    Args:
        metrics: All metrics
        save_path: Where to save summary
    """
    with open(save_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("FEDERATED RECOMMENDATION SYSTEM - RESULTS SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"✓ Results summary saved: {save_path}")


if __name__ == "__main__":
    print("Evaluation Utils module loaded successfully!")
    print("\nAvailable functions:")
    print("  - compute_metrics()")
    print("  - compute_per_class_metrics()")
    print("  - plot_confusion_matrix()")
    print("  - plot_training_history()")
    print("  - plot_federated_metrics()")
    print("  - plot_cluster_comparison()")
    print("  - plot_ifca_cluster_evolution()")
    print("  - print_evaluation_report()")
    print("  - compare_methods()")
    print("  - create_results_summary()")