"""
MAIN.PY - Main Entry Point for Federated Recommendation System

This is THE file you run to execute your entire project.
It connects all the pieces together.

Usage:
    python main.py --quick              # Quick test (5 mins)
    python main.py --phase all          # Full training (hours)
    python main.py --phase multimodal   # Only Phases 1-4
    python main.py --phase ifca         # Only Phase 7 (your innovation)
    python main.py --compare            # Compare all methods
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import sys
from datetime import datetime
from tqdm import tqdm

# Import your models
from models.text_encoder import TextEncoder, get_text_tokenizer
from models.image_encoder import ImageEncoder, get_image_transforms
from models.multi_modal import MutltiModalModel
from models.federatedLearning import FederatedTrainer
from models.clusteredFederated import IFCAClusteredFederatedLearning

# Import your utilities
from utils.data_utils import SyntheticSocialMediaDataset, create_dataloaders
from utils.train_utils import train_one_epoch, validate, test, save_checkpoint
from utils.eval_utils import (
    plot_confusion_matrix, plot_training_history, 
    plot_federated_metrics, plot_cluster_comparison,
    plot_ifca_cluster_evolution, compare_methods
)

# Import config
from configs.config import Config


def setup_environment():
    """Setup directories and environment"""
    print("\n Setting up environment...")
    
    # Create directories
    os.makedirs('./results', exist_ok=True)
    os.makedirs('./results/checkpoints', exist_ok=True)
    os.makedirs('./results/plots', exist_ok=True)
    os.makedirs('./results/logs', exist_ok=True)
    
    print(" Directories created")


def phase_1_to_4_multimodal():
    """
    PHASES 1-4: Train Multi-Modal Model
    
    Returns:
        model, datasets, tokenizer
    """
    print("\n" + "="*70)
    print("PHASES 1-4: MULTI-MODAL MODEL")
    print("="*70)
    
    device = Config.DEVICE
    print(f"Device: {device}")
    
    # Step 1: Get tokenizer
    print("\n Loading BERT tokenizer...")
    tokenizer = get_text_tokenizer(Config.BERT_MODEL_NAME)
    
    # Step 2: Create datasets
    print("\n Creating synthetic datasets...")
    train_dataset = SyntheticSocialMediaDataset(
        num_samples=Config.SYNTHETIC_NUM_SAMPLES_TRAIN,
        num_classes=Config.NUM_CLASSES,
        tokenizer=tokenizer,
        mode='train'
    )
    
    val_dataset = SyntheticSocialMediaDataset(
        num_samples=Config.SYNTHETIC_NUM_SAMPLES_VAL,
        num_classes=Config.NUM_CLASSES,
        tokenizer=tokenizer,
        mode='val'
    )
    
    test_dataset = SyntheticSocialMediaDataset(
        num_samples=Config.SYNTHETIC_NUM_SAMPLES_TEST,
        num_classes=Config.NUM_CLASSES,
        tokenizer=tokenizer,
        mode='test'
    )
    
    # Step 3: Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, 
        batch_size=Config.BATCH_SIZE
    )
    
    print(f" Train: {len(train_dataset)} samples")
    print(f" Val: {len(val_dataset)} samples")
    print(f" Test: {len(test_dataset)} samples")
    
    # Step 4: Create model
    print("\n Creating multi-modal model...")
    
    text_encoder = TextEncoder(
        model_name=Config.BERT_MODEL_NAME,
        hidden_dim=Config.TEXT_HIDDEN_DIM,
        freeze_bert=Config.FREEZE_BERT
    )
    
    image_encoder = ImageEncoder(
        model_name=Config.RESNET_MODEL,
        hidden_dim=Config.IMAGE_HIDDEN_DIM,
        freeze_backbone=Config.FREEZE_RESNET
    )
    
    model = MutltiModalModel(
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        fusion_type=Config.FUSION_TYPE,
        num_classes=Config.NUM_CLASSES,
        
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" Total parameters: {total_params:,}")
    print(f" Trainable parameters: {trainable_params:,}")
    
    # Step 5: Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Step 6: Training loop
    print(f"\n Training for {Config.NUM_EPOCHS} epochs...")
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{Config.NUM_EPOCHS}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        
        # Store metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = './results/checkpoints/multimodal_best.pth'
            save_checkpoint(model, optimizer, epoch, history, save_path)
    
    # Step 7: Test
    print("\n Testing final model...")
    test_results = test(model, test_loader, device)
    print(f"Test Accuracy: {test_results['accuracy']:.2f}%")
    
    # Step 8: Plot results
    if Config.PLOT_METRICS:
        print("\n Generating plots...")
        plot_training_history(history, save_path='./results/plots/multimodal_training.png')
        plot_confusion_matrix(
            test_results['predictions'],
            test_results['labels'],
            save_path='./results/plots/multimodal_confusion_matrix.png'
        )
    
    print("\n Multi-modal training complete!")
    
    return model, (train_dataset, val_dataset, test_dataset), tokenizer


def phase_6_standard_federated(model, datasets):
    """
    PHASE 6: Standard Federated Learning
    
    Args:
        model: Pre-trained multi-modal model
        datasets: Tuple of (train, val, test) datasets
        
    Returns:
        global_model, metrics
    """
    print("\n" + "="*70)
    print("PHASE 6: STANDARD FEDERATED LEARNING")
    print("="*70)
    
    train_dataset, val_dataset, test_dataset = datasets
    device = Config.DEVICE
    
    # Create FL trainer
    print(f"\n Setting up {Config.NUM_CLIENTS} clients...")
    trainer = FederatedTrainer(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        num_clients=Config.NUM_CLIENTS,
        clients_per_round=Config.CLIENTS_PER_ROUND,
        device=device,
        batch_size=Config.BATCH_SIZE
    )
    
    # Train
    print(f"\n Running {Config.NUM_FL_ROUNDS} FL rounds...")
    trainer.train(
        num_rounds=Config.NUM_FL_ROUNDS,
        local_epochs=Config.LOCAL_EPOCHS
    )
    
    # Save
    save_path = './results/checkpoints/federated_global.pth'
    torch.save(trainer.get_global_model().state_dict(), save_path)
    print(f"\n Model saved: {save_path}")
    
    # Plot
    if Config.PLOT_METRICS:
        print("\n Generating FL plots...")
        plot_federated_metrics(
            trainer.metrics,
            save_path='./results/plots/federated_metrics.png'
        )
    
    final_acc = trainer.metrics['test_acc'][-1]
    print(f"\n Standard FL complete! Final accuracy: {final_acc:.2f}%")
    
    return trainer.get_global_model(), trainer.metrics


def phase_7_ifca_clustered(model, datasets):
    """
    PHASE 7: IFCA Clustered FL + Knowledge Transfer
    YOUR INNOVATION!
    
    Args:
        model: Pre-trained multi-modal model
        datasets: Tuple of (train, val, test) datasets
        
    Returns:
        global_model, metrics
    """
    print("\n" + "="*70)
    print("PHASE 7: IFCA + KNOWLEDGE TRANSFER (YOUR INNOVATION)")
    print("="*70)
    
    train_dataset, val_dataset, test_dataset = datasets
    device = Config.DEVICE
    
    # Create IFCA trainer
    print(f"\n Setting up IFCA with {Config.NUM_CLUSTERS} clusters...")
    trainer = IFCAClusteredFederatedLearning(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        num_clients=Config.NUM_CLIENTS,
        num_clusters=Config.NUM_CLUSTERS,
        device=device,
        batch_size=Config.BATCH_SIZE
    )
    
    # Train
    print(f"\n Running IFCA for {Config.NUM_IFCA_ROUNDS} rounds...")
    trainer.train(
        num_ifca_rounds=Config.NUM_IFCA_ROUNDS,
        cluster_train_rounds=Config.CLUSTER_TRAIN_ROUNDS,
        clients_per_round=Config.CLIENTS_PER_ROUND,
        local_epochs=Config.LOCAL_EPOCHS,
        enable_knowledge_transfer=Config.ENABLE_INTER_CLUSTER_TRANSFER
    )
    
    # Save
    save_path = './results/checkpoints/ifca_global.pth'
    torch.save(trainer.get_global_model().state_dict(), save_path)
    print(f"\n Model saved: {save_path}")
    
    # Plot
    if Config.PLOT_METRICS:
        print("\n Generating IFCA plots...")
        plot_cluster_comparison(
            trainer.metrics['cluster_acc'],
            trainer.metrics['final_acc'],
            save_path='./results/plots/ifca_cluster_comparison.png'
        )
        
        if 'cluster_sizes_history' in trainer.metrics:
            plot_ifca_cluster_evolution(
                trainer.metrics['cluster_sizes_history'],
                save_path='./results/plots/ifca_evolution.png'
            )
    
    final_acc = trainer.metrics['final_acc']
    print(f"\n IFCA complete! Final accuracy: {final_acc:.2f}%")
    
    return trainer.get_global_model(), trainer.metrics


def compare_all_approaches(model, datasets):
    """
    Compare Standard FL vs IFCA (with and without transfer)
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE COMPARISON")
    print("="*70)
    
    results = {}
    
    # Standard FL
    print("\n Running Standard FL...")
    _, fed_metrics = phase_6_standard_federated(model, datasets)
    results['Standard FL'] = fed_metrics['test_acc'][-1]
    
    # IFCA without transfer
    print("\n Running IFCA (no knowledge transfer)...")
    train_dataset, val_dataset, test_dataset = datasets
    
    trainer_no_transfer = IFCAClusteredFederatedLearning(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        num_clients=Config.NUM_CLIENTS,
        num_clusters=Config.NUM_CLUSTERS,
        device=Config.DEVICE,
        batch_size=Config.BATCH_SIZE
    )
    
    trainer_no_transfer.train(
        num_ifca_rounds=Config.NUM_IFCA_ROUNDS,
        cluster_train_rounds=Config.CLUSTER_TRAIN_ROUNDS,
        enable_knowledge_transfer=False  # NO TRANSFER
    )
    
    results['IFCA (no transfer)'] = trainer_no_transfer.metrics['final_acc']
    
    # IFCA with transfer (YOUR INNOVATION)
    print("\n Running IFCA (with knowledge transfer)...")
    _, ifca_metrics = phase_7_ifca_clustered(model, datasets)
    results['IFCA + Transfer'] = ifca_metrics['final_acc']
    
    # Print comparison
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    for method, acc in results.items():
        print(f"{method:25s}: {acc:.2f}%")
    
    improvement = results['IFCA + Transfer'] - results['Standard FL']
    print(f"\n Your Innovation Improvement: +{improvement:.2f}%")
    print("="*70)
    
    # Plot
    if Config.PLOT_METRICS:
        compare_methods(results, save_path='./results/plots/method_comparison.png')
    
    # Save results
    with open('./results/comparison_results.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("METHOD COMPARISON RESULTS\n")
        f.write("="*70 + "\n\n")
        for method, acc in results.items():
            f.write(f"{method}: {acc:.2f}%\n")
        f.write(f"\nImprovement: +{improvement:.2f}%\n")
    
    return results


def main():
    """Main function - Entry point"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Federated Recommendation System')
    parser.add_argument(
        '--phase',
        type=str,
        default='all',
        choices=['all', 'multimodal', 'federated', 'ifca', 'compare'],
        help='Which phase to run'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test mode'
    )
    
    args = parser.parse_args()
    
    # Header
    print("\n" + "="*70)
    print("FEDERATED LEARNING FOR SOCIAL MEDIA RECOMMENDATIONS")
    print("="*70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Quick mode
    if args.quick:
        print("\n⚡ QUICK TEST MODE")
        Config.set_quick_test_mode()
    
    # Print config
    Config.print_config()
    
    # Setup
    setup_environment()
    
    try:
        # Run phases based on argument
        if args.phase in ['all', 'multimodal']:
            model, datasets, tokenizer = phase_1_to_4_multimodal()
        
        if args.phase == 'federated':
            if args.phase != 'all':
                model, datasets, tokenizer = phase_1_to_4_multimodal()
            phase_6_standard_federated(model, datasets)
        
        if args.phase == 'ifca':
            if args.phase != 'all':
                model, datasets, tokenizer = phase_1_to_4_multimodal()
            phase_7_ifca_clustered(model, datasets)
        
        if args.phase == 'all':
            phase_6_standard_federated(model, datasets)
            phase_7_ifca_clustered(model, datasets)
        
        if args.phase == 'compare':
            if args.phase == 'compare':
                model, datasets, tokenizer = phase_1_to_4_multimodal()
            compare_all_approaches(model, datasets)
        
        # Success
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n Results saved in: ./results/")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()