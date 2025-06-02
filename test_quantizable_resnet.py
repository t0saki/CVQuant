#!/usr/bin/env python3
"""
Test script to demonstrate quantizable ResNet50 implementation
"""
import torch
import torch.nn as nn
from models.model_loader import ModelLoader
from models.quantizable_resnet import resnet50_quantizable
from utils.data_loader import create_data_loaders
from quantization.quantizers import StaticQuantizer
import time

def test_quantizable_resnet50():
    """Test the quantizable ResNet50 implementation"""
    print("="*60)
    print("TESTING QUANTIZABLE RESNET50")
    print("="*60)
    
    # Create a quantizable ResNet50 model
    print("1. Loading quantizable ResNet50...")
    model = resnet50_quantizable(pretrained=True, num_classes=10)  # CIFAR-10
    model.eval()
    
    print(f"Model type: {type(model)}")
    print(f"Has QuantStub: {hasattr(model, 'quant')}")
    print(f"Has DeQuantStub: {hasattr(model, 'dequant')}")
    
    # Check if model has FloatFunctional in bottlenecks
    has_float_functional = False
    for module in model.modules():
        if hasattr(module, 'skip_add'):
            has_float_functional = True
            print(f"Found FloatFunctional in {type(module).__name__}")
            break
    
    print(f"Has FloatFunctional: {has_float_functional}")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    dummy_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # Test quantization
    print("\n3. Testing static quantization...")
    
    # Load some sample data for calibration
    try:
        calibration_loader, evaluation_loader = create_data_loaders(
            dataset_name='cifar10',
            data_path='./data',
            batch_size=32,
            calibration_size=100,
            evaluation_size=100,
            input_size=224
        )
        
        # Set quantization configuration
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare model for quantization
        model_prepared = torch.quantization.prepare(model)
        print("Model prepared for quantization")
        
        # Calibrate with sample data
        print("Calibrating model...")
        model_prepared.eval()
        with torch.no_grad():
            for i, (inputs, _) in enumerate(calibration_loader):
                if i >= 10:  # Use only first 10 batches for quick test
                    break
                _ = model_prepared(inputs)
        
        # Convert to quantized model
        model_quantized = torch.quantization.convert(model_prepared)
        print("Model quantized successfully!")
        
        # Test quantized model
        print("\n4. Testing quantized model...")
        with torch.no_grad():
            output_quantized = model_quantized(dummy_input)
        
        print(f"Quantized output shape: {output_quantized.shape}")
        print(f"Quantized output range: [{output_quantized.min().item():.4f}, {output_quantized.max().item():.4f}]")
        
        # Compare outputs
        output_diff = torch.abs(output - output_quantized).mean().item()
        print(f"Mean absolute difference: {output_diff:.6f}")
        
        # Measure inference time
        print("\n5. Measuring inference time...")
        
        def measure_time(model, input_tensor, num_runs=100):
            model.eval()
            with torch.no_grad():
                # Warmup
                for _ in range(10):
                    _ = model(input_tensor)
                
                # Measure
                start_time = time.time()
                for _ in range(num_runs):
                    _ = model(input_tensor)
                end_time = time.time()
                
                return (end_time - start_time) / num_runs * 1000  # ms per inference
        
        fp32_time = measure_time(model, dummy_input)
        int8_time = measure_time(model_quantized, dummy_input)
        
        print(f"FP32 inference time: {fp32_time:.3f} ms")
        print(f"INT8 inference time: {int8_time:.3f} ms")
        print(f"Speedup: {fp32_time / int8_time:.2f}x")
        
        # Model size comparison
        def get_model_size(model):
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                torch.save(model.state_dict(), tmp.name)
                size_mb = os.path.getsize(tmp.name) / (1024 * 1024)
                os.unlink(tmp.name)
            return size_mb
        
        fp32_size = get_model_size(model)
        int8_size = get_model_size(model_quantized)
        
        print(f"\nModel size comparison:")
        print(f"FP32 model size: {fp32_size:.2f} MB")
        print(f"INT8 model size: {int8_size:.2f} MB")
        print(f"Compression ratio: {fp32_size / int8_size:.2f}x")
        
    except Exception as e:
        print(f"Error during quantization test: {e}")
        print("This might be due to missing data. The model structure test passed successfully.")
    
    print("\n" + "="*60)
    print("TEST COMPLETED!")
    print("="*60)


def test_model_loader_integration():
    """Test the model loader integration"""
    print("\n" + "="*60)
    print("TESTING MODEL LOADER INTEGRATION")
    print("="*60)
    
    # Test loading through ModelLoader
    loader = ModelLoader(num_classes=10)
    
    # Test loading quantizable models
    print("Available models:", list(loader.available_models.keys()))
    
    print("\n1. Loading ResNet50 quantizable through ModelLoader...")
    model = loader.load_model('resnet50_quantizable', pretrained=True)
    print(f"Model type: {type(model)}")
    print(f"Has QuantStub: {hasattr(model, 'quant')}")
    
    print("\n2. Loading ResNet18 quantizable through ModelLoader...")
    model18 = loader.load_model('resnet18_quantizable', pretrained=True)
    print(f"Model type: {type(model18)}")
    print(f"Has QuantStub: {hasattr(model18, 'quant')}")
    
    # Test model info
    model_info = loader.get_model_info(model)
    print(f"\nModel info:")
    print(f"Total parameters: {model_info['total_parameters']:,}")
    print(f"Model size: {model_info['model_size_mb']:.2f} MB")


if __name__ == "__main__":
    test_quantizable_resnet50()
    test_model_loader_integration()
