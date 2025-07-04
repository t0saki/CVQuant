"""
Main script for quantization experiments on ResNet and MobileNet models
"""
import argparse
import torch
import torch.nn as nn
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from config import Config
from models.model_loader import ModelLoader, load_model, get_available_models
from quantization.quantizers import (
    DynamicQuantizer, StaticQuantizer, QATQuantizer,
    FXQuantizer, INT8Quantizer, QuantizationBenchmark
)
from utils.data_loader import create_data_loaders, adjust_dataset_for_model


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Quantization experiments for ResNet and MobileNet models')

    # Model configuration
    parser.add_argument('--model', type=str, default='resnet18_quantizable',
                        choices=['resnet18', 'resnet50', 'resnet18_quantizable', 'resnet50_quantizable', 'resnet18_low_rank', 'resnet50_low_rank', 'mobilenet_v2', 'mobilenet_v3_large',
                                 'mobilenet_v3_small', 'mobilenet_v3_large_quantizable', 'mobilenet_v3_small_quantizable', 'mobilenet_v4_conv_small', 'mobilenet_v4_conv_medium', 'mobilenet_v4_conv_large'],
                        help='Model to use for quantization experiments')

    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenet'],
                        help='Dataset to use for experiments')
    parser.add_argument('--data-path', type=str, default='./data',
                        help='Path to dataset')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for data loading')
    parser.add_argument('--calibration-size', type=int, default=50000,
                        help='Size of calibration dataset')
    parser.add_argument('--evaluation-size', type=int, default=10000,
                        help='Size of evaluation dataset')

    # Quantization configuration
    parser.add_argument('--methods', nargs='+',
                        default=['dynamic', 'static', 'qat'],
                        choices=['dynamic', 'static', 'qat',
                                 'fx', 'int8', 'official'],
                        help='Quantization methods to benchmark')
    parser.add_argument('--backend', type=str, default='x86',
                        choices=['fbgemm', 'qnnpack', 'x86'],
                        help='Quantization backend')

    # Experiment configuration
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--save-models', action='store_true',
                        help='Save quantized models')
    parser.add_argument('--plot-results', action='store_true',
                        help='Generate result plots')
    parser.add_argument('--enable-finetuning', action='store_true', default=True,
                        help='Enable automatic fine-tuning for dataset-specific weights')
    parser.add_argument('--disable-finetuning', action='store_true',
                        help='Disable fine-tuning and use only pretrained weights')
    parser.add_argument('--enable-distillation', action='store_true',
                        help='Enable knowledge distillation (for pre-training and/or QAT)')
    parser.add_argument('--teacher-model', type=str, default=None,
                        help='Teacher model name for knowledge distillation (auto-selected if not specified)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Device to use for experiments')

    # Benchmark configuration
    parser.add_argument('--warmup-iterations', type=int, default=50,
                        help='Number of warmup iterations for timing')
    parser.add_argument('--benchmark-iterations', type=int, default=500,
                        help='Number of benchmark iterations for timing')

    # Other model configuration
    parser.add_argument('--low-rank-epsilon', type=float, default=0.3,
                        help='Epsilon for Low Rank Factorization')
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Setup computation device"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available(
        ) else 'mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)

    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print("NOTE: Quantization will run on CPU as PyTorch quantization operations are CPU-only")
    else:
        print(f"Using device: {device}")

    return device


def load_and_prepare_model(model_name: str, dataset_name: str, device: torch.device,
                           enable_finetuning: bool = True, low_rank_epsilon: float = 0.3,
                           enable_distillation: bool = False, teacher_model_name: str = None) -> nn.Module:
    """Load and prepare model for quantization with optional fine-tuning or knowledge distillation"""
    print(f"\nLoading model: {model_name}")

    # Adjust dataset and get number of classes
    dataset_name, num_classes = adjust_dataset_for_model(
        dataset_name, model_name)

    # Load model with fine-tuning and/or distillation support
    model_loader = ModelLoader(num_classes=num_classes, device=device, enable_finetuning=enable_finetuning,
                               low_rank_epsilon=low_rank_epsilon, enable_distillation=enable_distillation)

    use_distill_pretrain = enable_finetuning and enable_distillation

    if use_distill_pretrain:
        print(
            f"Knowledge distillation pre-training enabled for dataset: {dataset_name}")
        if teacher_model_name:
            print(f"Using teacher model: {teacher_model_name}")
        else:
            print("Auto-selecting teacher model for pre-training")
        model = model_loader.load_model(model_name, pretrained=True, dataset_name=dataset_name,
                                        auto_finetune=True, use_distillation=True,
                                        teacher_model_name=teacher_model_name)
    elif enable_finetuning:
        print(f"Standard fine-tuning enabled for dataset: {dataset_name}")
        model = model_loader.load_model(
            model_name, pretrained=True, dataset_name=dataset_name, auto_finetune=True)
    else:
        model = model_loader.load_model(model_name, pretrained=True)

    model = model.to(device)
    model.eval()

    # Print model info
    model_info = model_loader.get_model_info(model)
    print(f"Model parameters: {model_info['total_parameters']:,}")
    print(f"Model size: {model_info['model_size_mb']:.2f} MB")

    return model


def run_quantization_experiments(model: nn.Module, model_name: str,
                                 calibration_loader, evaluation_loader,
                                 methods: list, device: torch.device,
                                 args, teacher_model: nn.Module = None) -> dict:
    """Run quantization experiments"""
    print(f"\nRunning quantization experiments...")
    print(f"Methods: {methods}")

    # Create benchmark instance
    benchmark = QuantizationBenchmark(
        device=device, backend=args.backend, model_name=model_name)

    # Run benchmark
    results = benchmark.benchmark_quantization_methods(
        model=model,
        calibration_data=calibration_loader,
        evaluation_data=evaluation_loader,
        methods=methods,
        teacher_model=teacher_model,
        args=args
    )

    # Print results
    benchmark.print_results()

    return results


def save_results(results: dict, model_name: str, dataset_name: str,
                 methods: list, output_dir: str):
    """Save experiment results"""
    os.makedirs(output_dir, exist_ok=True)

    # Create results filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(
        output_dir, f"quantization_results_{model_name}_{dataset_name}_{timestamp}.json")

    # Prepare results for saving
    save_data = {
        'experiment_info': {
            'model': model_name,
            'dataset': dataset_name,
            'methods': methods,
            'timestamp': timestamp
        },
        'results': results
    }

    # Save to JSON
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    return results_file


def plot_results(results: dict, model_name: str, dataset_name: str, output_dir: str):
    """Generate and save result plots"""
    print("\nGenerating result plots...")

    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    # Extract data for plotting
    methods = []
    accuracies = []
    inference_times = []
    model_sizes = []
    speedups = []
    compression_ratios = []

    for method, result in results.items():
        if 'error' in result or method == 'original':
            continue

        methods.append(method)
        accuracies.append(result['accuracy'])
        inference_times.append(result['inference_time_ms'])
        model_sizes.append(result['model_size_mb'])
        speedups.append(result.get('speedup', 1.0))
        compression_ratios.append(result.get('compression_ratio', 1.0))

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        f'Quantization Results: {model_name} on {dataset_name}', fontsize=16)

    # Plot 1: Accuracy comparison
    axes[0, 0].bar(methods, accuracies, alpha=0.7)
    axes[0, 0].set_title('Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    if results.get('original'):
        axes[0, 0].axhline(y=results['original']['accuracy'], color='red', linestyle='--',
                           label=f'Original: {results["original"]["accuracy"]:.2f}%')
        axes[0, 0].legend()

    # Plot 2: Inference time comparison
    axes[0, 1].bar(methods, inference_times, alpha=0.7, color='orange')
    axes[0, 1].set_title('Inference Time Comparison')
    axes[0, 1].set_ylabel('Inference Time (ms)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    if results.get('original'):
        axes[0, 1].axhline(y=results['original']['inference_time_ms'], color='red', linestyle='--',
                           label=f'Original: {results["original"]["inference_time_ms"]:.2f}ms')
        axes[0, 1].legend()

    # Plot 3: Model size comparison
    axes[1, 0].bar(methods, model_sizes, alpha=0.7, color='green')
    axes[1, 0].set_title('Model Size Comparison')
    axes[1, 0].set_ylabel('Model Size (MB)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    if results.get('original'):
        axes[1, 0].axhline(y=results['original']['model_size_mb'], color='red', linestyle='--',
                           label=f'Original: {results["original"]["model_size_mb"]:.2f}MB')
        axes[1, 0].legend()

    # Plot 4: Speedup vs Compression
    axes[1, 1].scatter(compression_ratios, speedups, s=100, alpha=0.7)
    for i, method in enumerate(methods):
        axes[1, 1].annotate(method, (compression_ratios[i], speedups[i]),
                            xytext=(5, 5), textcoords='offset points')
    axes[1, 1].set_title('Speedup vs Compression Ratio')
    axes[1, 1].set_xlabel('Compression Ratio')
    axes[1, 1].set_ylabel('Speedup')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = os.path.join(
        output_dir, f"quantization_plots_{model_name}_{dataset_name}_{timestamp}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Plots saved to: {plot_file}")


def save_quantized_models(benchmark, methods: list, model_name: str,
                          dataset_name: str, output_dir: str):
    """Save quantized models"""
    print("\nSaving quantized models...")

    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for method in methods:
        if hasattr(benchmark, 'quantizers') and method in benchmark.quantizers:
            quantizer = benchmark.quantizers[method]
            if hasattr(quantizer, 'quantized_model') and quantizer.quantized_model is not None:
                model_file = os.path.join(models_dir,
                                          f"{model_name}_{dataset_name}_{method}_{timestamp}.pth")
                torch.save(quantizer.quantized_model.state_dict(), model_file)
                print(f"Saved {method} quantized model to: {model_file}")


def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()

    # Setup device
    device = setup_device(args.device)

    # Print experiment configuration
    print("="*60)
    print("QUANTIZATION EXPERIMENT CONFIGURATION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Quantization methods: {args.methods}")
    print(f"Backend: {args.backend}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Calibration size: {args.calibration_size}")
    print(f"Evaluation size: {args.evaluation_size}")
    print(
        f"Fine-tuning enabled: {args.enable_finetuning and not args.disable_finetuning}")
    print(f"Knowledge distillation enabled: {args.enable_distillation}")
    if args.enable_distillation:
        print(f"Teacher model: {args.teacher_model or 'Auto-selected'}")
    print("="*60)

    try:
        # Load and prepare student model
        enable_finetuning = args.enable_finetuning and not args.disable_finetuning
        # Decide if the initial weights should be from distillation
        use_distill_pretrain = enable_finetuning and args.enable_distillation

        model = load_and_prepare_model(
            args.model, args.dataset, device, enable_finetuning,
            args.low_rank_epsilon, use_distill_pretrain, args.teacher_model
        )

        # Load teacher model if distillation is enabled for any method (e.g., QAT)
        teacher_model = None
        if args.enable_distillation:
            print(
                "\nKnowledge distillation is enabled. Loading teacher model for potential use in QAT.")
            teacher_model_name = args.teacher_model
            if not teacher_model_name:
                from utils.distillation import auto_select_teacher_model
                teacher_model_name = auto_select_teacher_model(
                    args.model, get_available_models())
                print(f"Auto-selected teacher model: {teacher_model_name}")

            if teacher_model_name:
                # Load teacher, ensuring it is also fine-tuned on the dataset.
                # The teacher itself is not distilled.
                teacher_model = load_and_prepare_model(
                    teacher_model_name, args.dataset, device,
                    enable_finetuning=True, enable_distillation=False
                )
            else:
                print(
                    "Warning: Could not find a suitable teacher model. QAT will proceed without distillation if selected.")

        # Load data
        print(f"\nLoading dataset: {args.dataset}")
        calibration_loader, evaluation_loader = create_data_loaders(
            dataset_name=args.dataset,
            data_path=args.data_path,
            batch_size=args.batch_size,
            calibration_size=args.calibration_size,
            evaluation_size=args.evaluation_size,
            input_size=224 if 'mobilenet_v4' in args.model else 224
        )

        print(f"Calibration samples: {len(calibration_loader.dataset)}")
        print(f"Evaluation samples: {len(evaluation_loader.dataset)}")

        # Run quantization experiments
        results = run_quantization_experiments(
            model=model,
            model_name=args.model,
            calibration_loader=calibration_loader,
            evaluation_loader=evaluation_loader,
            methods=args.methods,
            device=device,
            args=args,
            teacher_model=teacher_model
        )

        # Save results
        results_file = save_results(
            results=results,
            model_name=args.model,
            dataset_name=args.dataset,
            methods=args.methods,
            output_dir=args.output_dir
        )

        # Generate plots
        if args.plot_results:
            plot_results(
                results=results,
                model_name=args.model,
                dataset_name=args.dataset,
                output_dir=args.output_dir
            )

        # Save quantized models
        if args.save_models:
            print("Note: Model saving functionality requires additional implementation")

        print("\n" + "="*60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*60)

    except Exception as e:
        print(f"\nError during experiment: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
