"""
Test script for knowledge distillation functionality
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def test_distillation_imports():
    """Test that all distillation modules can be imported"""
    print("Testing imports...")
    try:
        from utils.distillation import KnowledgeDistiller
        print("✓ KnowledgeDistiller imported successfully")
        
        from utils.fine_tuning import FineTuner
        print("✓ FineTuner imported successfully")
        
        from models.model_loader import ModelLoader
        print("✓ ModelLoader imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_teacher_model_selection():
    """Test teacher model auto-selection functionality"""
    print("\nTesting teacher model selection...")
    
    try:
        from utils.distillation import auto_select_teacher_model, get_compatible_teacher_models
        
        # Test ResNet models
        student = 'resnet18_quantizable'
        compatible_teachers = get_compatible_teacher_models(student)
        print(f"Compatible teachers for {student}: {compatible_teachers}")
        
        available_models = ['resnet18', 'resnet50', 'resnet18_quantizable', 'resnet50_quantizable']
        selected_teacher = auto_select_teacher_model(student, available_models)
        print(f"Auto-selected teacher for {student}: {selected_teacher}")
        
        # Test MobileNet models
        student = 'mobilenet_v3_small_quantizable'
        compatible_teachers = get_compatible_teacher_models(student)
        print(f"Compatible teachers for {student}: {compatible_teachers}")
        
        available_models = ['mobilenet_v3_small', 'mobilenet_v3_large', 'mobilenet_v3_large_quantizable']
        selected_teacher = auto_select_teacher_model(student, available_models)
        print(f"Auto-selected teacher for {student}: {selected_teacher}")
        
        return True
    except Exception as e:
        print(f"✗ Teacher model selection test failed: {e}")
        return False


def test_distiller_creation():
    """Test KnowledgeDistiller creation"""
    print("\nTesting KnowledgeDistiller creation...")
    
    try:
        import torch
        device = torch.device('cpu')
        
        from utils.distillation import KnowledgeDistiller
        distiller = KnowledgeDistiller(device=device)
        print("✓ KnowledgeDistiller created successfully")
        
        # Test configuration methods
        config = distiller.get_distillation_config('resnet50', 'resnet18', 'cifar10')
        print(f"✓ Distillation config generated: {config}")
        
        return True
    except Exception as e:
        print(f"✗ KnowledgeDistiller test failed: {e}")
        return False


def main():
    """Main test function"""
    print("="*60)
    print("KNOWLEDGE DISTILLATION FUNCTIONALITY TESTS")
    print("="*60)
    
    tests = [
        test_distillation_imports,
        test_teacher_model_selection,
        test_distiller_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Knowledge distillation functionality is ready.")
    else:
        print("✗ Some tests failed. Please check the implementation.")
    
    print("="*60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
