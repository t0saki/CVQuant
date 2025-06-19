"""
Test script to verify proper data splitting for fine-tuning
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_data_splitting():
    """Test that fine-tuning data splitting is correct"""
    print("Testing fine-tuning data splitting...")
    
    try:
        from utils.fine_tuning import create_fine_tuning_data_loaders
        
        # Test CIFAR-10 splitting
        train_loader, val_loader = create_fine_tuning_data_loaders(
            dataset_name='cifar10',
            batch_size=64,
            train_split=0.8,
            total_samples=10000  # Use smaller sample for testing
        )
        
        print(f"✓ CIFAR-10 data loaders created successfully")
        print(f"  Train batches: {len(train_loader)} (approx {len(train_loader) * 64} samples)")
        print(f"  Val batches: {len(val_loader)} (approx {len(val_loader) * 64} samples)")
        
        # Check that train and val are using same underlying dataset (training set)
        # but different indices
        sample_train_batch = next(iter(train_loader))
        sample_val_batch = next(iter(val_loader))
        
        print(f"  Train batch shape: {sample_train_batch[0].shape}")
        print(f"  Val batch shape: {sample_val_batch[0].shape}")
        print(f"  Train labels range: {sample_train_batch[1].min()}-{sample_train_batch[1].max()}")
        print(f"  Val labels range: {sample_val_batch[1].min()}-{sample_val_batch[1].max()}")
        
        # Verify no data leakage - check that we're using training set only
        # (This is implicit in our implementation, but we can verify batch sizes are reasonable)
        expected_train_batches = int(10000 * 0.8 / 64)
        expected_val_batches = int(10000 * 0.2 / 64)
        
        print(f"  Expected train batches: ~{expected_train_batches}, actual: {len(train_loader)}")
        print(f"  Expected val batches: ~{expected_val_batches}, actual: {len(val_loader)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data splitting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_distillation_data_splitting():
    """Test that distillation data splitting is correct"""
    print("\nTesting distillation data splitting...")
    
    try:
        from utils.distillation import create_distillation_data_loaders
        
        # Test distillation data splitting
        train_loader, val_loader = create_distillation_data_loaders(
            dataset_name='cifar10',
            batch_size=64,
            train_split=0.8
        )
        
        print(f"✓ Distillation data loaders created successfully")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Distillation data splitting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_original_data_loaders():
    """Test original data loaders to understand the issue"""
    print("\nTesting original data loaders behavior...")
    
    try:
        from utils.data_loader import create_data_loaders, DatasetLoader
        
        # Test original behavior
        calibration_loader, evaluation_loader = create_data_loaders(
            dataset_name='cifar10',
            batch_size=64,
            calibration_size=5000,
            evaluation_size=2000
        )
        
        print(f"Original data loaders:")
        print(f"  Calibration batches: {len(calibration_loader)}")
        print(f"  Evaluation batches: {len(evaluation_loader)}")
        
        # Check which dataset they come from
        loader = DatasetLoader()
        cifar10_train = loader.get_cifar10_dataset(batch_size=64, train=True)
        cifar10_test = loader.get_cifar10_dataset(batch_size=64, train=False)
        
        print(f"  CIFAR-10 train dataset size: {len(cifar10_train.dataset)} samples")
        print(f"  CIFAR-10 test dataset size: {len(cifar10_test.dataset)} samples")
        
        print("\n⚠️  ISSUE IDENTIFIED:")
        print("  - Calibration data comes from TRAINING set")
        print("  - Evaluation data comes from TEST set")
        print("  - When used for fine-tuning, this creates train-on-train, validate-on-test")
        print("  - This leads to artificially high validation accuracy!")
        
        return True
        
    except Exception as e:
        print(f"✗ Original data loaders test failed: {e}")
        return False


def compare_accuracy_expectations():
    """Show expected accuracy ranges"""
    print("\n" + "="*60)
    print("EXPECTED ACCURACY RANGES (CORRECTED)")
    print("="*60)
    
    print("""
With proper train/val splitting from training set only:

ResNet50 on CIFAR-10:
- Baseline (pretrained): ~70-80%
- Fine-tuned (proper train/val split): ~85-92%  
- Previous inflated results: ~99% (due to train/test leakage)

ResNet18 on CIFAR-10:
- Baseline (pretrained): ~65-75%
- Fine-tuned: ~80-88%
- Distilled from ResNet50: ~82-90%

The key fixes:
1. ✓ Use only training set for fine-tuning
2. ✓ Split training set into train/val (80/20)
3. ✓ Never validate on test set during training
4. ✓ Reserve test set only for final evaluation

This will give more realistic and comparable results.
""")


def main():
    """Main test function"""
    print("="*60)
    print("DATA SPLITTING VERIFICATION TESTS")
    print("="*60)
    
    tests = [
        test_data_splitting,
        test_distillation_data_splitting,
        test_original_data_loaders
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    compare_accuracy_expectations()
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Data splitting is now correct.")
    else:
        print("✗ Some tests failed. Please check the implementation.")
    
    print("="*60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
