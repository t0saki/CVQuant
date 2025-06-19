"""
Example script demonstrating knowledge distillation usage
"""
import subprocess
import sys
import os

def run_example_with_distillation():
    """Run an example with knowledge distillation enabled"""
    
    print("=" * 80)
    print("KNOWLEDGE DISTILLATION EXAMPLE")
    print("=" * 80)
    print()
    
    # Example 1: ResNet18 quantizable with ResNet50 as teacher
    print("Example 1: Distilling ResNet50 → ResNet18 quantizable on CIFAR-10")
    print("-" * 60)
    
    cmd1 = [
        "python3", "main.py",
        "--model", "resnet18_quantizable",
        "--dataset", "cifar10",
        "--methods", "dynamic", "static", "qat",
        "--enable-distillation",
        "--teacher-model", "resnet50",
        "--batch-size", "64",
        "--calibration-size", "5000",
        "--evaluation-size", "1000",
        "--device", "cpu",  # Use CPU for demo to avoid GPU memory issues
        "--output-dir", "./results_distillation_demo"
    ]
    
    print("Command:")
    print(" ".join(cmd1))
    print()
    
    # Example 2: MobileNet V3 Small with auto teacher selection
    print("Example 2: Auto-selecting teacher for MobileNet V3 Small quantizable on CIFAR-10")
    print("-" * 60)
    
    cmd2 = [
        "python3", "main.py",
        "--model", "mobilenet_v3_small_quantizable",
        "--dataset", "cifar10",
        "--methods", "dynamic", "static",
        "--enable-distillation",
        # No --teacher-model specified, will auto-select
        "--batch-size", "64",
        "--calibration-size", "5000",
        "--evaluation-size", "1000",
        "--device", "cpu",
        "--output-dir", "./results_distillation_demo"
    ]
    
    print("Command:")
    print(" ".join(cmd2))
    print()
    
    # Example 3: Compare fine-tuning vs distillation
    print("Example 3: Compare fine-tuning vs knowledge distillation")
    print("-" * 60)
    
    print("Fine-tuning command:")
    cmd3a = [
        "python3", "main.py",
        "--model", "resnet18_quantizable",
        "--dataset", "cifar10",
        "--methods", "static",
        "--enable-finetuning",
        "--batch-size", "64",
        "--calibration-size", "2000",
        "--evaluation-size", "500",
        "--device", "cpu",
        "--output-dir", "./results_finetuning_demo"
    ]
    print(" ".join(cmd3a))
    print()
    
    print("Knowledge distillation command:")
    cmd3b = [
        "python3", "main.py",
        "--model", "resnet18_quantizable",
        "--dataset", "cifar10",
        "--methods", "static",
        "--enable-distillation",
        "--teacher-model", "resnet50",
        "--batch-size", "64",
        "--calibration-size", "2000",
        "--evaluation-size", "500",
        "--device", "cpu",
        "--output-dir", "./results_distillation_demo"
    ]
    print(" ".join(cmd3b))
    print()
    
    print("=" * 80)
    print("KNOWLEDGE DISTILLATION USAGE GUIDE")
    print("=" * 80)
    print()
    
    usage_guide = """
Key Parameters for Knowledge Distillation:

1. --enable-distillation
   Enable knowledge distillation instead of traditional fine-tuning
   
2. --teacher-model <model_name>
   Specify the teacher model name (optional)
   If not specified, the system will auto-select a suitable teacher model
   
3. Compatible Teacher-Student Pairs:
   
   ResNet Students:
   - resnet18 → resnet50, resnet50_quantizable
   - resnet18_quantizable → resnet50, resnet50_quantizable, resnet18
   - resnet18_low_rank → resnet50, resnet50_quantizable, resnet18, resnet18_quantizable
   
   MobileNet Students:
   - mobilenet_v3_small → mobilenet_v3_large, mobilenet_v4_conv_medium, mobilenet_v4_conv_large
   - mobilenet_v3_small_quantizable → mobilenet_v3_large, mobilenet_v3_large_quantizable, mobilenet_v4_conv_medium, mobilenet_v4_conv_large
   - mobilenet_v3_large → mobilenet_v4_conv_medium, mobilenet_v4_conv_large
   - mobilenet_v4_conv_small → mobilenet_v4_conv_medium, mobilenet_v4_conv_large
   
4. Distillation Process:
   - The teacher model will be loaded with pretrained weights
   - If fine-tuned weights exist for the teacher on the dataset, they will be used
   - The student model learns from both the teacher's soft predictions and hard labels
   - Temperature and alpha parameters are automatically configured based on model pair and dataset
   
5. Benefits of Knowledge Distillation:
   - Often achieves better accuracy than fine-tuning alone
   - Transfers knowledge from larger, more capable models to smaller ones
   - Particularly effective for quantization tasks
   - Can improve the performance of quantized models significantly
   
6. Output:
   - Distilled weights are saved in ./distilled_weights/ directory
   - Results include both distillation metrics and quantization performance
   - Model comparisons show improvements over baseline approaches

Example Commands:

# Auto-select teacher and run distillation
python3 main.py --model resnet18_quantizable --dataset cifar10 --enable-distillation --methods dynamic static

# Specify teacher model
python3 main.py --model resnet18_quantizable --dataset cifar10 --enable-distillation --teacher-model resnet50 --methods qat

# Compare with fine-tuning (run both and compare results)
python3 main.py --model resnet18_quantizable --dataset cifar10 --enable-finetuning --methods static
python3 main.py --model resnet18_quantizable --dataset cifar10 --enable-distillation --methods static

"""
    
    print(usage_guide)
    
    print("To run any of the above examples, copy and paste the commands.")
    print("Make sure you have sufficient computational resources and time.")
    print("For faster testing, use smaller --calibration-size and --evaluation-size values.")
    print()

if __name__ == "__main__":
    run_example_with_distillation()
