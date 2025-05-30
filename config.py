"""
Configuration file for quantization experiments
"""
import torch

class Config:
    # Dataset configuration
    DATASET_NAME = "imagenet"
    DATA_PATH = "./data"
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    # Model configuration
    MODELS = {
        'resnet18': 'torchvision',
        'resnet50': 'torchvision', 
        'mobilenet_v2': 'torchvision',
        'mobilenet_v3_large': 'torchvision',
        'mobilenet_v3_small': 'torchvision'
    }
    
    # Quantization configuration
    QUANTIZATION_METHODS = [
        'dynamic',      # Dynamic quantization
        'static',       # Static quantization 
        'qat',          # Quantization Aware Training
        'fx',           # FX Graph Mode quantization
        'int8'          # INT8 quantization
    ]
    
    # Training configuration
    EPOCHS = 10
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Evaluation configuration
    NUM_CALIBRATION_SAMPLES = 1000
    NUM_EVALUATION_SAMPLES = 5000
    
    # Results configuration
    RESULTS_DIR = "./results"
    SAVE_MODELS = True
    
    # Quantization specific settings
    QCONFIG_MAPPING = {
        'dynamic': torch.quantization.default_dynamic_qconfig,
        'static': torch.quantization.get_default_qconfig('fbgemm'),
        'qat': torch.quantization.get_default_qat_qconfig('fbgemm')
    }
    
    # Benchmark settings
    WARMUP_ITERATIONS = 10
    BENCHMARK_ITERATIONS = 100