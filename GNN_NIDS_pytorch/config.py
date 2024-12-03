import os

class Config:
    # Get the absolute base directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Paths
    TRAIN_PATH = os.path.join(BASE_DIR, 'preprocess_dataset/preprocessed_IDS2017/TRAIN')
    VAL_PATH = os.path.join(BASE_DIR, 'preprocess_dataset/preprocessed_IDS2017/EVAL')
    LOG_DIR = os.path.join(BASE_DIR, 'GNN_NIDS_pytorch/logs')
    CHECKPOINT_DIR = os.path.join(BASE_DIR, 'GNN_NIDS_pytorch/checkpoints')

    # Model parameters
    NODE_STATE_DIM = 64
    NUM_LAYERS = 4  
    INPUT_DIM = 84  
    OUTPUT_DIM = 16  
    HIDDEN_DIM = 64
    DROPOUT = 0.2  

    # Training parameters
    BATCH_SIZE = 2  
    LEARNING_RATE = 0.00001
    NUM_EPOCHS = 100
    WEIGHT_DECAY = 0.01
    WARMUP_EPOCHS = 5
    GRADIENT_CLIP = 0.5
    
    # Memory management
    GRADIENT_CHECKPOINTING = True
    EMPTY_CACHE_FREQ = 1  
    USE_AMP = True  
    EDGE_SAMPLING_RATIO = 0.3  
    
    # Numerical stability
    USE_LAYER_NORM = True
    
    # Early stopping
    PATIENCE = 10
    
    # Device configuration
    DEVICE = 'cpu'  
