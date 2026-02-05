"""
Configuration File for Federated Recommendation System

All hyperparameters and settings in one place.
Change values here instead of modifying code everywhere.
"""

import torch


class Config:
    """
    Complete configuration for the entire project
    """
    
    # ========================================================================
    # DEVICE CONFIGURATION
    # ========================================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ========================================================================
    # MODEL CONFIGURATION
    # ========================================================================
    
    # Text Encoder (BERT)
    BERT_MODEL_NAME = 'bert-base-uncased'
    TEXT_HIDDEN_DIM = 256
    MAX_TEXT_LENGTH = 128
    FREEZE_BERT = True  # Set False to fine-tune BERT
    
    # Image Encoder (ResNet)
    RESNET_MODEL = 'resnet50'
    IMAGE_HIDDEN_DIM = 256
    FREEZE_RESNET = True  # Set False to fine-tune ResNet
    
    # Multi-Modal Fusion
    FUSION_TYPE = 'attention'  # Options: 'simple', 'attention', 'gated'
    FUSION_HIDDEN_DIM = 256
    
    # Classification
    NUM_CLASSES = 10  # Number of categories (adjust for your dataset)
    
    # ========================================================================
    # TRAINING CONFIGURATION
    # ========================================================================
    
    # General Training
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    GRADIENT_CLIP_VALUE = 1.0
    
    # Learning Rate Scheduler
    USE_SCHEDULER = True
    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.5
    
    # Dropout
    DROPOUT_RATE = 0.3
    
    # ========================================================================
    # FEDERATED LEARNING CONFIGURATION (Phase 6)
    # ========================================================================
    
    # Clients
    NUM_CLIENTS = 10
    CLIENTS_PER_ROUND = 5
    
    # Federated Rounds
    NUM_FL_ROUNDS = 100
    LOCAL_EPOCHS = 5
    
    # Client Data Distribution
    DATA_DISTRIBUTION = 'iid'  # Options: 'iid', 'non-iid'
    NON_IID_ALPHA = 0.5  # Dirichlet alpha for non-IID split
    
    # ========================================================================
    # IFCA CLUSTERED FL CONFIGURATION (Phase 7)
    # ========================================================================
    
    # Clustering
    NUM_CLUSTERS = 3
    NUM_IFCA_ROUNDS = 3  # How many times to reassign clusters
    CLUSTER_TRAIN_ROUNDS = 5  # FL rounds within each cluster
    
    # Knowledge Transfer
    ENABLE_INTER_CLUSTER_TRANSFER = True
    TRANSFER_TEMPERATURE = 4.0
    TRANSFER_ALPHA = 0.3
    TRANSFER_EPOCHS = 2
    
    # ========================================================================
    # BIGRAPHNET CONFIGURATION (Phase 5)
    # ========================================================================
    
    # Graph
    NUM_USERS = 100
    NUM_ITEMS = 500
    
    # BiGraphNet Architecture
    GRAPH_HIDDEN_DIMS = [256, 128, 64]
    GRAPH_NUM_LAYERS = 3
    GRAPH_DROPOUT = 0.3
    
    # Recommendations
    TOP_K_RECOMMENDATIONS = 10
    
    # ========================================================================
    # DATASET CONFIGURATION
    # ========================================================================
    
    # Paths
    DATA_DIR = './data'
    RAW_DATA_DIR = './data/raw'
    PROCESSED_DATA_DIR = './data/processed'
    SPLITS_DIR = './data/splits'
    
    # Dataset Type
    USE_SYNTHETIC_DATA = True  # Set False when using real data
    SYNTHETIC_NUM_SAMPLES_TRAIN = 2000
    SYNTHETIC_NUM_SAMPLES_VAL = 400
    SYNTHETIC_NUM_SAMPLES_TEST = 400
    
    # Real Data (when USE_SYNTHETIC_DATA = False)
    TRAIN_CSV = './data/splits/train.csv'
    VAL_CSV = './data/splits/val.csv'
    TEST_CSV = './data/splits/test.csv'
    IMAGE_DIR = './data/processed/images'
    
    # Data Split Ratios
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # ========================================================================
    # RESULTS & CHECKPOINTS
    # ========================================================================
    
    # Directories
    RESULTS_DIR = './results'
    CHECKPOINT_DIR = './results/checkpoints'
    LOGS_DIR = './results/logs'
    PLOTS_DIR = './results/plots'
    
    # Model Checkpoints
    SAVE_BEST_MODEL = True
    SAVE_EVERY_N_EPOCHS = 10
    
    # Checkpoint Names
    MULTIMODAL_CHECKPOINT = 'multimodal_best.pth'
    BIGRAPHNET_CHECKPOINT = 'bigraphnet_best.pth'
    FEDERATED_CHECKPOINT = 'federated_global.pth'
    CLUSTERED_CHECKPOINT = 'clustered_global.pth'
    
    # ========================================================================
    # LOGGING & VISUALIZATION
    # ========================================================================
    
    # TensorBoard
    USE_TENSORBOARD = True
    TENSORBOARD_LOG_DIR = './results/logs/tensorboard'
    
    # Console Logging
    LOG_INTERVAL = 10  # Print every N batches
    
    # Plotting
    PLOT_METRICS = True
    SAVE_PLOTS = True
    
    # ========================================================================
    # PRIVACY & SECURITY (Optional - for future work)
    # ========================================================================
    
    # Differential Privacy
    DIFFERENTIAL_PRIVACY_ENABLED = False
    DP_EPSILON = 1.0
    DP_DELTA = 1e-5
    DP_MAX_GRAD_NORM = 1.0
    
    # Secure Aggregation
    SECURE_AGGREGATION_ENABLED = False
    
    # ========================================================================
    # EVALUATION METRICS
    # ========================================================================
    
    # Which metrics to compute
    COMPUTE_ACCURACY = True
    COMPUTE_PRECISION = True
    COMPUTE_RECALL = True
    COMPUTE_F1 = True
    COMPUTE_CONFUSION_MATRIX = True
    
    # Recommendation Metrics
    COMPUTE_PRECISION_AT_K = True
    COMPUTE_RECALL_AT_K = True
    COMPUTE_NDCG = True
    K_VALUES = [5, 10, 20]  # For P@K, R@K, NDCG@K
    
    # ========================================================================
    # EXPERIMENT SETTINGS
    # ========================================================================
    
    # Random Seeds
    RANDOM_SEED = 42
    
    # Experiment Name
    EXPERIMENT_NAME = 'federated_recommendation_ifca'
    
    # Debug Mode
    DEBUG_MODE = False
    QUICK_TEST = False  # Use tiny dataset for quick testing
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    @staticmethod
    def get_model_config():
        """Get model-related config as dict"""
        return {
            'bert_model': Config.BERT_MODEL_NAME,
            'text_hidden_dim': Config.TEXT_HIDDEN_DIM,
            'image_hidden_dim': Config.IMAGE_HIDDEN_DIM,
            'fusion_type': Config.FUSION_TYPE,
            'num_classes': Config.NUM_CLASSES
        }
    
    @staticmethod
    def get_training_config():
        """Get training-related config as dict"""
        return {
            'batch_size': Config.BATCH_SIZE,
            'num_epochs': Config.NUM_EPOCHS,
            'learning_rate': Config.LEARNING_RATE,
            'weight_decay': Config.WEIGHT_DECAY
        }
    
    @staticmethod
    def get_federated_config():
        """Get federated learning config as dict"""
        return {
            'num_clients': Config.NUM_CLIENTS,
            'clients_per_round': Config.CLIENTS_PER_ROUND,
            'num_fl_rounds': Config.NUM_FL_ROUNDS,
            'local_epochs': Config.LOCAL_EPOCHS,
            'num_clusters': Config.NUM_CLUSTERS,
            'num_ifca_rounds': Config.NUM_IFCA_ROUNDS
        }
    
    @staticmethod
    def print_config():
        """Print current configuration"""
        print("="*70)
        print("CONFIGURATION")
        print("="*70)
        print(f"\n📱 Device: {Config.DEVICE}")
        print(f"\n🤖 Model:")
        print(f"  BERT: {Config.BERT_MODEL_NAME}")
        print(f"  Text Dim: {Config.TEXT_HIDDEN_DIM}")
        print(f"  Image Dim: {Config.IMAGE_HIDDEN_DIM}")
        print(f"  Fusion: {Config.FUSION_TYPE}")
        print(f"  Classes: {Config.NUM_CLASSES}")
        print(f"\n🎓 Training:")
        print(f"  Batch Size: {Config.BATCH_SIZE}")
        print(f"  Epochs: {Config.NUM_EPOCHS}")
        print(f"  Learning Rate: {Config.LEARNING_RATE}")
        print(f"\n🌐 Federated Learning:")
        print(f"  Clients: {Config.NUM_CLIENTS}")
        print(f"  Clients/Round: {Config.CLIENTS_PER_ROUND}")
        print(f"  FL Rounds: {Config.NUM_FL_ROUNDS}")
        print(f"  Clusters: {Config.NUM_CLUSTERS}")
        print(f"  IFCA Rounds: {Config.NUM_IFCA_ROUNDS}")
        print(f"\n💾 Data:")
        print(f"  Synthetic: {Config.USE_SYNTHETIC_DATA}")
        if Config.USE_SYNTHETIC_DATA:
            print(f"  Train Samples: {Config.SYNTHETIC_NUM_SAMPLES_TRAIN}")
        else:
            print(f"  Train CSV: {Config.TRAIN_CSV}")
        print("="*70)
    
    @staticmethod
    def set_quick_test_mode():
        """Set config for quick testing (small dataset, few epochs)"""
        print("⚡ QUICK TEST MODE ENABLED")
        Config.NUM_EPOCHS = 5
        Config.NUM_FL_ROUNDS = 10
        Config.NUM_IFCA_ROUNDS = 2
        Config.CLUSTER_TRAIN_ROUNDS = 3
        Config.NUM_CLIENTS = 5
        Config.CLIENTS_PER_ROUND = 3
        Config.SYNTHETIC_NUM_SAMPLES_TRAIN = 500
        Config.SYNTHETIC_NUM_SAMPLES_VAL = 100
        Config.SYNTHETIC_NUM_SAMPLES_TEST = 100
        Config.QUICK_TEST = True


# Create default config instance
config = Config()


if __name__ == "__main__":
    # Test configuration
    Config.print_config()
    
    print("\n\nTesting quick mode:")
    Config.set_quick_test_mode()
    Config.print_config()