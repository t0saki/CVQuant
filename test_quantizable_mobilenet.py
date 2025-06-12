#!/usr/bin/env python3
"""
Test script for quantizable MobileNet V3 implementation
"""

import torch
import torch.nn as nn
from models.model_loader import ModelLoader
from models.quantizable_mobilenet import mobilenet_v3_small_quantizable, mobilenet_v3_large_quantizable

def test_quantizable_mobilenet():
    """Test the quantizable MobileNet V3 implementation"""
    print("Testing quantizable MobileNet V3 implementation...")
    
    # Test direct function calls
    print("\n1. Testing direct function calls:")
    
    # Test small variant
    print("   Loading MobileNet V3 Small quantizable...")
    model_small = mobilenet_v3_small_quantizable(pretrained=False, num_classes=10)
    print(f"   Model loaded successfully: {type(model_small).__name__}")
    
    # Test large variant
    print("   Loading MobileNet V3 Large quantizable...")
    model_large = mobilenet_v3_large_quantizable(pretrained=False, num_classes=10)
    print(f"   Model loaded successfully: {type(model_large).__name__}")
    
    # Test model loader integration
    print("\n2. Testing ModelLoader integration:")
    loader = ModelLoader(num_classes=10)
    
    print("   Available models:", loader.available_models.keys())
    
    # Test loading through ModelLoader
    print("   Loading mobilenet_v3_small_quantizable through ModelLoader...")
    model_from_loader = loader.load_model('mobilenet_v3_small_quantizable', pretrained=False)
    print(f"   Model loaded successfully: {type(model_from_loader).__name__}")
    
    # Test forward pass
    print("\n3. Testing forward pass:")
    test_input = torch.randn(2, 3, 224, 224)  # Batch of 2 images
    
    model_small.eval()
    with torch.no_grad():
        output = model_small(test_input)
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Expected shape: torch.Size([2, 10])")
        assert output.shape == torch.Size([2, 10]), f"Expected output shape [2, 10], got {output.shape}"
    
    # Test quantization preparation
    print("\n4. Testing quantization preparation:")
    model_small.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model_small, inplace=True)
    print("   Model prepared for quantization successfully")
    
    # Test that QuantStub and DeQuantStub exist
    print("\n5. Testing QuantStub and DeQuantStub:")
    has_quant_stub = hasattr(model_small, 'quant')
    has_dequant_stub = hasattr(model_small, 'dequant')
    print(f"   Has QuantStub: {has_quant_stub}")
    print(f"   Has DeQuantStub: {has_dequant_stub}")
    assert has_quant_stub and has_dequant_stub, "Model should have QuantStub and DeQuantStub"
    
    # Test FloatFunctional usage
    print("\n6. Testing FloatFunctional usage:")
    # Check if the model has FloatFunctional modules
    float_functional_count = 0
    for name, module in model_small.named_modules():
        if isinstance(module, torch.nn.quantized.FloatFunctional):
            float_functional_count += 1
    print(f"   Found {float_functional_count} FloatFunctional modules")
    
    print("\n✅ All tests passed! Quantizable MobileNet V3 implementation is working correctly.")
    
    return True

if __name__ == "__main__":
    try:
        test_quantizable_mobilenet()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
