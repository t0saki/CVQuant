main.py
```python
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

```

utils/data_loader.py
```python
"""
Data loading utilities for quantization experiments
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import os
from typing import Tuple, Optional
import random


class DatasetLoader:
    """Dataset loader for various computer vision datasets"""
    
    def __init__(self, data_path: str = "./data", download: bool = True):
        self.data_path = data_path
        self.download = download
        
        # Create data directory if it doesn't exist
        os.makedirs(data_path, exist_ok=True)
        
        # Standard ImageNet normalization
        self.imagenet_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def get_imagenet_transforms(self, input_size: int = 224, is_training: bool = False) -> transforms.Compose:
        """
        Get ImageNet preprocessing transforms
        
        Args:
            input_size: Input image size
            is_training: Whether for training (includes augmentation)
            
        Returns:
            Composed transforms
        """
        if is_training:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.imagenet_normalize
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(int(input_size * 1.143)),  # Resize to 256 for 224
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                self.imagenet_normalize
            ])
        
        return transform
    
    def get_cifar10_dataset(self, batch_size: int = 32, num_workers: int = 4,
                            train: bool = True, input_size: int = 32) -> DataLoader:  # 添加 input_size 参数
        """
        Get CIFAR-10 dataset loader
        
        Args:
            batch_size: Batch size
            num_workers: Number of worker processes
            train: Whether to load training set
            
        Returns:
            DataLoader for CIFAR-10
        """
        if train:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.Resize(input_size),  # 训练时也统一尺寸
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            # --- 这是关键修改 ---
            transform = transforms.Compose([
                transforms.Resize(input_size),  # 添加 Resize
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

        dataset = torchvision.datasets.CIFAR10(
            root=self.data_path, train=train, download=self.download, transform=transform
        )

        return DataLoader(
            dataset, batch_size=batch_size, shuffle=train,
            num_workers=num_workers, pin_memory=True
        )

    def get_cifar100_dataset(self, batch_size: int = 32, num_workers: int = 4,
                             train: bool = True, input_size: int = 32) -> DataLoader:  # 添加 input_size 参数
        """
        Get CIFAR-100 dataset loader
        
        Args:
            batch_size: Batch size
            num_workers: Number of worker processes
            train: Whether to load training set
            
        Returns:
            DataLoader for CIFAR-100
        """
        if train:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.Resize(input_size),  # 训练时也统一尺寸
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        else:
            # --- 这是关键修改 ---
            transform = transforms.Compose([
                transforms.Resize(input_size),  # 添加 Resize
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])

        dataset = torchvision.datasets.CIFAR100(
            root=self.data_path, train=train, download=self.download, transform=transform
        )

        return DataLoader(
            dataset, batch_size=batch_size, shuffle=train,
            num_workers=num_workers, pin_memory=True
        )
    
    def get_imagenet_subset(self, batch_size: int = 32, num_workers: int = 4,
                           subset_size: int = 1000, input_size: int = 224,
                           train: bool = False) -> DataLoader:
        """
        Get ImageNet subset (uses CIFAR-10 as proxy for demo purposes)
        
        Args:
            batch_size: Batch size
            num_workers: Number of worker processes
            subset_size: Size of subset
            input_size: Input image size
            train: Whether to load training set
            
        Returns:
            DataLoader for ImageNet subset
        """
        print("Note: Using CIFAR-10 resized to ImageNet size as ImageNet proxy for demo")
        
        if train:
            base_transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.imagenet_normalize
            ])
        else:
            base_transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                self.imagenet_normalize
            ])
        
        # Use CIFAR-10 as proxy
        full_dataset = torchvision.datasets.CIFAR10(
            root=self.data_path, train=train, download=self.download, transform=base_transform
        )
        
        # Create subset
        if subset_size < len(full_dataset):
            indices = random.sample(range(len(full_dataset)), subset_size)
            dataset = Subset(full_dataset, indices)
        else:
            dataset = full_dataset
        
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=train,
            num_workers=num_workers, pin_memory=True
        )
    
    def get_calibration_data(self, dataset_name: str = "cifar10",
                                 calibration_size: int = 1000,
                                 batch_size: int = 32, input_size: int = 224) -> DataLoader:
        """
        Get calibration data for quantization
        
        Args:
            dataset_name: Name of dataset ('cifar10', 'cifar100', 'imagenet')
            calibration_size: Size of calibration set
            batch_size: Batch size
            input_size: Input image size
            
        Returns:
            DataLoader for calibration
        """
        if dataset_name.lower() == 'cifar10':
            # --- 传递 input_size ---
            full_loader = self.get_cifar10_dataset(
                batch_size=batch_size, train=True, input_size=input_size)
        elif dataset_name.lower() == 'cifar100':
            # --- 传递 input_size ---
            full_loader = self.get_cifar100_dataset(
                batch_size=batch_size, train=True, input_size=input_size)
        elif dataset_name.lower() == 'imagenet':
            full_loader = self.get_imagenet_subset(
                batch_size=batch_size, subset_size=calibration_size,
                input_size=input_size, train=True
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Create calibration subset
        if dataset_name.lower() in ['cifar10', 'cifar100']:
            dataset = full_loader.dataset
            if calibration_size < len(dataset):
                indices = random.sample(range(len(dataset)), calibration_size)
                calibration_dataset = Subset(dataset, indices)
            else:
                calibration_dataset = dataset
            
            return DataLoader(
                calibration_dataset, batch_size=batch_size, shuffle=False,
                num_workers=4, pin_memory=True
            )
        else:
            return full_loader
    
    def get_evaluation_data(self, dataset_name: str = "cifar10",
                            evaluation_size: int = 5000,
                            batch_size: int = 32, input_size: int = 224) -> DataLoader:
        """
        Get evaluation data for quantization experiments
        
        Args:
            dataset_name: Name of dataset ('cifar10', 'cifar100', 'imagenet')
            evaluation_size: Size of evaluation set
            batch_size: Batch size
            input_size: Input image size
            
        Returns:
            DataLoader for evaluation
        """
        if dataset_name.lower() == 'cifar10':
            # --- 传递 input_size ---
            full_loader = self.get_cifar10_dataset(
                batch_size=batch_size, train=False, input_size=input_size)
        elif dataset_name.lower() == 'cifar100':
            # --- 传递 input_size ---
            full_loader = self.get_cifar100_dataset(
                batch_size=batch_size, train=False, input_size=input_size)
        elif dataset_name.lower() == 'imagenet':
            full_loader = self.get_imagenet_subset(
                batch_size=batch_size, subset_size=evaluation_size,
                input_size=input_size, train=False
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Create evaluation subset
        if dataset_name.lower() in ['cifar10', 'cifar100']:
            dataset = full_loader.dataset
            if evaluation_size < len(dataset):
                indices = random.sample(range(len(dataset)), evaluation_size)
                evaluation_dataset = Subset(dataset, indices)
            else:
                evaluation_dataset = dataset
            
            return DataLoader(
                evaluation_dataset, batch_size=batch_size, shuffle=False,
                num_workers=4, pin_memory=True
            )
        else:
            return full_loader


def create_data_loaders(dataset_name: str = "cifar10", 
                       data_path: str = "./data",
                       batch_size: int = 32,
                       calibration_size: int = 1000,
                       evaluation_size: int = 5000,
                       input_size: int = 224) -> Tuple[DataLoader, DataLoader]:
    """
    Create calibration and evaluation data loaders
    
    Args:
        dataset_name: Name of dataset ('cifar10', 'cifar100', 'imagenet')
        data_path: Path to data directory
        batch_size: Batch size
        calibration_size: Size of calibration set
        evaluation_size: Size of evaluation set
        input_size: Input image size
        
    Returns:
        Tuple of (calibration_loader, evaluation_loader)
    """
    loader = DatasetLoader(data_path=data_path)
    
    calibration_loader = loader.get_calibration_data(
        dataset_name=dataset_name,
        calibration_size=calibration_size,
        batch_size=batch_size,
        input_size=input_size
    )
    
    evaluation_loader = loader.get_evaluation_data(
        dataset_name=dataset_name,
        evaluation_size=evaluation_size,
        batch_size=batch_size,
        input_size=input_size
    )
    
    return calibration_loader, evaluation_loader


def get_sample_input(data_loader: DataLoader, device: torch.device) -> torch.Tensor:
    """
    Get a sample input tensor from data loader
    
    Args:
        data_loader: DataLoader
        device: Target device
        
    Returns:
        Sample input tensor
    """
    for inputs, _ in data_loader:
        return inputs[:1].to(device)  # Return first sample
    
    raise RuntimeError("Empty data loader")


def adjust_dataset_for_model(dataset_name: str, model_name: str, 
                           num_classes: Optional[int] = None) -> Tuple[str, int]:
    """
    Adjust dataset and get number of classes based on model
    
    Args:
        dataset_name: Original dataset name
        model_name: Model name
        num_classes: Override number of classes
        
    Returns:
        Tuple of (adjusted_dataset_name, num_classes)
    """
    if num_classes is not None:
        return dataset_name, num_classes
    
    # Map datasets to number of classes
    dataset_classes = {
        'cifar10': 10,
        'cifar100': 100,
        'imagenet': 1000
    }
    
    # For ImageNet pretrained models, use appropriate dataset
    if 'resnet' in model_name or 'mobilenet' in model_name:
        if dataset_name.lower() == 'imagenet':
            return dataset_name, 1000
        elif dataset_name.lower() == 'cifar10':
            return dataset_name, 10
        elif dataset_name.lower() == 'cifar100':
            return dataset_name, 100
    
    return dataset_name, dataset_classes.get(dataset_name.lower(), 1000)
```

models/model_loader.py
```python
"""
Model loader for ResNet and MobileNet models with fine-tuning support
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.quantization import resnet
import timm
from typing import Dict, Any, Optional
from .quantizable_resnet import resnet50_quantizable, resnet18_quantizable, resnet101_quantizable
from .quantizable_resnet_lrf import resnet18_low_rank, resnet50_low_rank
from .quantizable_mobilenet import mobilenet_v3_small_quantizable, mobilenet_v3_large_quantizable

class ModelLoader:
    """Load and prepare models for quantization experiments with fine-tuning support"""
    
    def __init__(self, num_classes: int = 1000, device: torch.device = None, enable_finetuning: bool = True, 
                 low_rank_epsilon: float = 0.3, enable_distillation: bool = False):
        # Set seed for consistent model initialization
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        self.num_classes = num_classes
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.enable_finetuning = enable_finetuning
        self.enable_distillation = enable_distillation
        self.low_rank_epsilon = low_rank_epsilon
        
        # Initialize fine-tuner if enabled
        self.fine_tuner = None
        if enable_finetuning:
            try:
                import sys
                import os
                # Add the project root to sys.path if not already there
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)
                    
                from utils.fine_tuning import FineTuner
                self.fine_tuner = FineTuner(device=self.device, enable_distillation=enable_distillation)
            except ImportError as e:
                print(f"Warning: Could not import fine-tuning module: {e}")
                print("Fine-tuning will be disabled")
                self.enable_finetuning = False
                self.enable_distillation = False
            
        self.available_models = {
            'resnet18': self._load_resnet18,
            'resnet50': self._load_resnet50,
            'resnet18_quantizable': self._load_resnet18_quantizable,
            'resnet50_quantizable': self._load_resnet50_quantizable,
            'resnet101_quantizable': self._load_resnet101_quantizable,
            "resnet18_low_rank": self._load_resnet18_low_rank,
            "resnet50_low_rank": self._load_resnet50_low_rank,
            'mobilenet_v2': self._load_mobilenet_v2,
            'mobilenet_v3_large': self._load_mobilenet_v3_large,
            'mobilenet_v3_small': self._load_mobilenet_v3_small,
            'mobilenet_v3_large_quantizable': self._load_mobilenet_v3_large_quantizable,
            'mobilenet_v3_small_quantizable': self._load_mobilenet_v3_small_quantizable,
            'mobilenet_v4_conv_small': self._load_mobilenet_v4_conv_small,
            'mobilenet_v4_conv_medium': self._load_mobilenet_v4_conv_medium,
            'mobilenet_v4_conv_large': self._load_mobilenet_v4_conv_large,
        }
    
    def load_model(self, model_name: str, pretrained: bool = True, dataset_name: Optional[str] = None, 
                   auto_finetune: bool = True, use_distillation: bool = False, teacher_model_name: Optional[str] = None) -> nn.Module:
        """
        Load a model by name with optional fine-tuning or knowledge distillation support
        
        Args:
            model_name: Name of the model to load
            pretrained: Whether to load pretrained weights
            dataset_name: Dataset name for fine-tuning (if None, no fine-tuning)
            auto_finetune: Whether to automatically fine-tune if weights don't exist
            use_distillation: Whether to use knowledge distillation instead of fine-tuning
            teacher_model_name: Name of teacher model for distillation (auto-selected if None)
            
        Returns:
            PyTorch model (potentially fine-tuned or distilled)
        """
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not available. "
                           f"Available models: {list(self.available_models.keys())}")
        
        # Load base model
        model = self.available_models[model_name](pretrained)
        
        # Handle optimization if dataset is specified and fine-tuning is enabled
        if dataset_name and self.enable_finetuning and self.fine_tuner:
            if use_distillation and self.enable_distillation:
                # Try knowledge distillation approach
                model = self._handle_distillation(model, model_name, dataset_name, teacher_model_name, auto_finetune)
            else:
                # Traditional fine-tuning approach
                model = self._handle_finetuning(model, model_name, dataset_name, auto_finetune)
        
        return model
    
    def _handle_finetuning(self, model: nn.Module, model_name: str, dataset_name: str, auto_finetune: bool) -> nn.Module:
        """Handle traditional fine-tuning approach"""
        if self.fine_tuner.has_finetuned_weights(model_name, dataset_name):
            # Load existing fine-tuned weights
            print(f"Found fine-tuned weights for {model_name} on {dataset_name}")
            model = self.fine_tuner.load_finetuned_weights(model, model_name, dataset_name)
        elif auto_finetune:
            # Perform fine-tuning
            print(f"No fine-tuned weights found for {model_name} on {dataset_name}")
            print("Starting fine-tuning process...")
            model = self._perform_finetuning(model, model_name, dataset_name)
        else:
            print(f"No fine-tuned weights found for {model_name} on {dataset_name}, using pretrained weights")
        
        return model
    
    def _handle_distillation(self, model: nn.Module, model_name: str, dataset_name: str, 
                           teacher_model_name: Optional[str], auto_finetune: bool) -> nn.Module:
        """Handle knowledge distillation approach"""
        # Auto-select teacher model if not specified
        if teacher_model_name is None:
            from utils.distillation import auto_select_teacher_model
            teacher_model_name = auto_select_teacher_model(model_name, list(self.available_models.keys()))
            if teacher_model_name is None:
                print(f"No suitable teacher model found for {model_name}, falling back to fine-tuning")
                return self._handle_finetuning(model, model_name, dataset_name, auto_finetune)
            print(f"Auto-selected teacher model: {teacher_model_name}")
        
        # Check if distilled weights already exist
        if self.fine_tuner.has_distilled_weights(model_name, teacher_model_name, dataset_name):
            print(f"Found distilled weights for {model_name} from {teacher_model_name} on {dataset_name}")
            model = self.fine_tuner.load_distilled_weights(model, model_name, teacher_model_name, dataset_name)
        elif auto_finetune:
            # Perform knowledge distillation
            print(f"No distilled weights found for {model_name} from {teacher_model_name} on {dataset_name}")
            print("Starting knowledge distillation process...")
            model = self._perform_distillation(model, model_name, teacher_model_name, dataset_name)
        else:
            print(f"No distilled weights found for {model_name}, using pretrained weights")
        
        return model
    
    def _perform_distillation(self, student_model: nn.Module, student_model_name: str, 
                            teacher_model_name: str, dataset_name: str) -> nn.Module:
        """Perform knowledge distillation process"""
        if not self.enable_distillation or not self.fine_tuner:
            print("Knowledge distillation is disabled, returning student model as-is")
            return student_model
        
        try:
            # Load teacher model
            print(f"Loading teacher model: {teacher_model_name}")
            teacher_model = self.available_models[teacher_model_name](pretrained=True)
            
            # Ensure teacher model is fine-tuned on the dataset
            if self.fine_tuner.has_finetuned_weights(teacher_model_name, dataset_name):
                print(f"Loading existing fine-tuned teacher weights for {teacher_model_name} on {dataset_name}")
                teacher_model = self.fine_tuner.load_finetuned_weights(teacher_model, teacher_model_name, dataset_name)
            else:
                print(f"No fine-tuned weights found for teacher {teacher_model_name} on {dataset_name}")
                print("Fine-tuning teacher model with full dataset...")
                teacher_model = self._perform_teacher_finetuning(teacher_model, teacher_model_name, dataset_name)
            
            teacher_model = teacher_model.to(self.device)
            teacher_model.eval()
            
            # Perform distillation
            student_model = self.fine_tuner.distill_model(
                teacher_model=teacher_model,
                student_model=student_model,
                teacher_model_name=teacher_model_name,
                student_model_name=student_model_name,
                dataset_name=dataset_name,
                auto_config=True
            )
            
            return student_model
            
        except Exception as e:
            print(f"Knowledge distillation failed: {e}")
            print("Falling back to fine-tuning approach")
            return self._handle_finetuning(student_model, student_model_name, dataset_name, True)
    
    def _load_resnet18(self, pretrained: bool = True) -> nn.Module:
        """Load ResNet-18 model"""
        model = resnet.resnet18(pretrained=pretrained)
        if self.num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        return model
    
    def _load_resnet50(self, pretrained: bool = True) -> nn.Module:
        """Load ResNet-50 model"""
        model = resnet.resnet50(pretrained=pretrained)
        if self.num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        return model
    
    def _load_resnet18_quantizable(self, pretrained: bool = True) -> nn.Module:
        """Load quantizable ResNet-18 model with QuantStub and DeQuantStub"""
        model = resnet18_quantizable(pretrained=pretrained, num_classes=self.num_classes)
        return model
    
    def _load_resnet50_quantizable(self, pretrained: bool = True) -> nn.Module:
        """Load quantizable ResNet-50 model with QuantStub and DeQuantStub"""
        model = resnet50_quantizable(pretrained=pretrained, num_classes=self.num_classes)
        return model
    
    def _load_resnet101_quantizable(self, pretrained: bool = True) -> nn.Module:
        """Load quantizable ResNet-101 model with QuantStub and DeQuantStub"""
        model = resnet101_quantizable(pretrained=pretrained, num_classes=self.num_classes)
        return model

    def _load_resnet18_low_rank(self, pretrained: bool = True) -> nn.Module:
        """Load ResNet-18 model with low rank factorization"""
        model = resnet18_low_rank(
            pretrained=pretrained,
            num_classes=self.num_classes,
            epsilon=self.low_rank_epsilon,
            device=self.device
        )
        return model

    def _load_resnet50_low_rank(self, pretrained: bool = True) -> nn.Module:
        """Load ResNet-50 model with low rank factorization"""
        model = resnet50_low_rank(
            pretrained=pretrained,
            num_classes=self.num_classes,
            epsilon=self.low_rank_epsilon,
            device=self.device
        )
        return model
    
    def _load_mobilenet_v2(self, pretrained: bool = True) -> nn.Module:
        """Load MobileNet V2 model"""
        model = models.mobilenet_v2(pretrained=pretrained)
        if self.num_classes != 1000:
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
        return model
    
    def _load_mobilenet_v3_large(self, pretrained: bool = True) -> nn.Module:
        """Load MobileNet V3 Large model"""
        model = models.mobilenet_v3_large(pretrained=pretrained)
        if self.num_classes != 1000:
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, self.num_classes)
        return model
    
    def _load_mobilenet_v3_small(self, pretrained: bool = True) -> nn.Module:
        """Load MobileNet V3 Small model"""
        model = models.mobilenet_v3_small(pretrained=pretrained)
        if self.num_classes != 1000:
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, self.num_classes)
        return model
    
    def _load_mobilenet_v3_large_quantizable(self, pretrained: bool = True) -> nn.Module:
        """Load quantizable MobileNet V3 Large model with QuantStub and DeQuantStub"""
        model = mobilenet_v3_large_quantizable(pretrained=pretrained, num_classes=self.num_classes)
        return model
    
    def _load_mobilenet_v3_small_quantizable(self, pretrained: bool = True) -> nn.Module:
        """Load quantizable MobileNet V3 Small model with QuantStub and DeQuantStub"""
        model = mobilenet_v3_small_quantizable(pretrained=pretrained, num_classes=self.num_classes)
        return model
    
    def _load_mobilenet_v4_conv_small(self, pretrained: bool = True) -> nn.Module:
        """Load MobileNet V4 ConvSmall model using timm"""
        try:
            model = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k', 
                                    pretrained=pretrained, num_classes=self.num_classes)
            return model
        except Exception as e:
            print(f"Warning: Could not load MobileNet V4 ConvSmall: {e}")
            print("Falling back to MobileNet V3 Large")
            return self._load_mobilenet_v3_large(pretrained)
    
    def _load_mobilenet_v4_conv_medium(self, pretrained: bool = True) -> nn.Module:
        """Load MobileNet V4 ConvMedium model using timm"""
        try:
            model = timm.create_model('mobilenetv4_conv_medium.e500_r256_in1k', 
                                    pretrained=pretrained, num_classes=self.num_classes)
            return model
        except Exception as e:
            print(f"Warning: Could not load MobileNet V4 ConvMedium: {e}")
            print("Falling back to MobileNet V3 Large")
            return self._load_mobilenet_v3_large(pretrained)
    
    def _load_mobilenet_v4_conv_large(self, pretrained: bool = True) -> nn.Module:
        """Load MobileNet V4 ConvLarge model using timm"""
        try:
            model = timm.create_model('mobilenetv4_conv_large.e600_r384_in1k', 
                                    pretrained=pretrained, num_classes=self.num_classes)
            return model
        except Exception as e:
            print(f"Warning: Could not load MobileNet V4 ConvLarge: {e}")
            print("Falling back to MobileNet V3 Large")
            return self._load_mobilenet_v3_large(pretrained)

    def get_model_info(self, model: nn.Module, bytes: int = 4) -> Dict[str, Any]:
        """
        Get information about a model
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * bytes / (1024 * 1024),
        }

        # 添加压缩统计信息（如果有的话）
        if hasattr(model, 'compression_stats'):
            stats = model.compression_stats
            info.update({
                'compression_stats': stats,
                'original_parameters': stats.get('total_old_size', total_params),
                'compression_ratio': stats.get('compression_ratio', 1.0),
                'parameter_reduction': stats.get('total_reduction', 0),
                'epsilon': stats.get('epsilon', 'N/A')
            })

        return info
    
    def prepare_model_for_quantization(self, model: nn.Module, method: str) -> nn.Module:
        """
        Prepare model for specific quantization method
        
        Args:
            model: PyTorch model
            method: Quantization method ('dynamic', 'static', 'qat', 'fx')
            
        Returns:
            Prepared model
        """
        model.eval()
        
        if method in ['static', 'qat']:
            # Add QuantStub and DeQuantStub for static/QAT quantization
            model = self._add_quant_dequant_stubs(model)
        
        return model
    
    def _add_quant_dequant_stubs(self, model: nn.Module) -> nn.Module:
        """Add quantization and dequantization stubs to model"""
        class QuantizedModel(nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.quant = torch.quantization.QuantStub()
                self.model = original_model
                self.dequant = torch.quantization.DeQuantStub()
            
            def forward(self, x):
                x = self.quant(x)
                x = self.model(x)
                x = self.dequant(x)
                return x
        
        return QuantizedModel(model)
    
    def _perform_finetuning(self, model: nn.Module, model_name: str, dataset_name: str) -> nn.Module:
        """
        Perform fine-tuning on the model for the specified dataset
        
        Args:
            model: Base model to fine-tune
            model_name: Name of the model
            dataset_name: Name of the dataset
            
        Returns:
            Fine-tuned model
        """
        if not self.fine_tuner:
            print("Fine-tuning is disabled, returning base model")
            return model
        
        try:
            # Import here to avoid circular imports
            import sys
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            from utils.fine_tuning import create_fine_tuning_data_loaders
            
            # Get training configuration
            config = self.fine_tuner.get_training_config(dataset_name, model_name)
            if "mobilenet" in model_name:
                # For MobileNet models, use a smaller batch size
                config['epochs'] = config.get('epochs', 64) * 10

            # Create data loaders for fine-tuning
            train_loader, val_loader = create_fine_tuning_data_loaders(
                dataset_name=dataset_name,
                batch_size=config.get('batch_size', 128),
                total_samples=50000  # Standard dataset size for student training
            )
            
            # Perform fine-tuning
            history = self.fine_tuner.fine_tune_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                model_name=model_name,
                dataset_name=dataset_name,
                epochs=config.get('epochs', 10),
                learning_rate=config.get('learning_rate', 0.001),
                weight_decay=config.get('weight_decay', 1e-4)
            )
            
            print(f"Fine-tuning completed. Best validation accuracy: {history['best_val_acc']:.2f}%")
            return model
            
        except Exception as e:
            print(f"Fine-tuning failed: {e}")
            print("Returning base model with pretrained weights")
            return model
    
    def _perform_teacher_finetuning(self, teacher_model: nn.Module, teacher_model_name: str, dataset_name: str) -> nn.Module:
        """
        Perform fine-tuning on the teacher model for the specified dataset using full dataset
        
        Args:
            teacher_model: Teacher model to fine-tune
            teacher_model_name: Name of the teacher model
            dataset_name: Name of the dataset
            
        Returns:
            Fine-tuned teacher model
        """
        if not self.fine_tuner:
            print("Fine-tuning is disabled, returning base teacher model")
            return teacher_model
        
        try:
            # Import here to avoid circular imports
            import sys
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            from utils.fine_tuning import create_fine_tuning_data_loaders
            
            # Get training configuration for teacher model
            config = self.fine_tuner.get_training_config(dataset_name, teacher_model_name)
            
            # Use larger dataset and more epochs for teacher model training
            if "mobilenet" in teacher_model_name:
                # For MobileNet models, use appropriate configuration
                config['epochs'] = max(config.get('epochs', 10), 15)
                config['batch_size'] = min(config.get('batch_size', 128), 64)  # Smaller batch for larger models
            else:
                # For ResNet models
                config['epochs'] = max(config.get('epochs', 10), 20)  # More epochs for teacher
                config['batch_size'] = min(config.get('batch_size', 128), 96)  # Adjust batch size
            
            # Use larger dataset for teacher training (train_split=0.9 to use more data)
            train_loader, val_loader = create_fine_tuning_data_loaders(
                dataset_name=dataset_name,
                batch_size=config.get('batch_size', 96),
                train_split=0.9,  # Use 90% for training, 10% for validation
                total_samples=100000  # Use larger dataset for teacher training
            )
            
            print(f"Fine-tuning teacher model {teacher_model_name} with {len(train_loader.dataset)} training samples")
            
            # Perform fine-tuning on teacher model
            history = self.fine_tuner.fine_tune_model(
                model=teacher_model,
                train_loader=train_loader,
                val_loader=val_loader,
                model_name=teacher_model_name,
                dataset_name=dataset_name,
                epochs=config.get('epochs', 20),
                learning_rate=config.get('learning_rate', 0.0005),  # Slightly lower LR for teacher
                weight_decay=config.get('weight_decay', 1e-4)
            )
            
            print(f"Teacher fine-tuning completed. Best validation accuracy: {history['best_val_acc']:.2f}%")
            return teacher_model
            
        except Exception as e:
            print(f"Teacher fine-tuning failed: {e}")
            print("Using pretrained teacher weights")
            return teacher_model


def get_available_models() -> list:
    """Get list of available model names"""
    loader = ModelLoader()
    return list(loader.available_models.keys())


def load_model(model_name: str, pretrained: bool = True, num_classes: int = 1000, 
               dataset_name: Optional[str] = None, auto_finetune: bool = True, 
               device: torch.device = None, use_distillation: bool = False, 
               teacher_model_name: Optional[str] = None) -> nn.Module:
    """
    Convenience function to load a model with optional fine-tuning or knowledge distillation
    
    Args:
        model_name: Name of the model to load
        pretrained: Whether to load pretrained weights
        num_classes: Number of output classes
        dataset_name: Dataset name for fine-tuning (if None, no fine-tuning)
        auto_finetune: Whether to automatically fine-tune if weights don't exist
        device: Device to use for model
        use_distillation: Whether to use knowledge distillation instead of fine-tuning
        teacher_model_name: Name of teacher model for distillation (auto-selected if None)
        
    Returns:
        PyTorch model (potentially fine-tuned or distilled)
    """
    loader = ModelLoader(num_classes=num_classes, device=device, enable_finetuning=True, 
                        enable_distillation=use_distillation)
    return loader.load_model(model_name, pretrained, dataset_name, auto_finetune, 
                           use_distillation, teacher_model_name)

```

utils/fine_tuning.py
```python
"""
Fine-tuning utilities for models on specific datasets
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple
import time
from tqdm import tqdm


class FineTuner:
    """Fine-tune pre-trained models on specific datasets"""
    
    def __init__(self, device: torch.device = None, weights_dir: str = "./fine_tuned_weights", 
                 enable_distillation: bool = False):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weights_dir = weights_dir
        self.enable_distillation = enable_distillation
        os.makedirs(weights_dir, exist_ok=True)
        
        # Initialize distiller if enabled
        self.distiller = None
        if enable_distillation:
            try:
                from .distillation import KnowledgeDistiller
                self.distiller = KnowledgeDistiller(device=self.device)
            except ImportError as e:
                print(f"Warning: Could not import distillation module: {e}")
                print("Knowledge distillation will be disabled")
                self.enable_distillation = False
    
    def get_finetuned_weights_path(self, model_name: str, dataset_name: str) -> str:
        """Get the path for fine-tuned weights"""
        return os.path.join(self.weights_dir, f"{model_name}_{dataset_name}_finetuned.pth")
    
    def has_finetuned_weights(self, model_name: str, dataset_name: str) -> bool:
        """Check if fine-tuned weights exist for the model-dataset combination"""
        weights_path = self.get_finetuned_weights_path(model_name, dataset_name)
        return os.path.exists(weights_path)
    
    def load_finetuned_weights(self, model: nn.Module, model_name: str, dataset_name: str) -> nn.Module:
        """Load fine-tuned weights into the model"""
        weights_path = self.get_finetuned_weights_path(model_name, dataset_name)
        
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Fine-tuned weights not found at {weights_path}")
        
        print(f"Loading fine-tuned weights from {weights_path}")
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            checkpoint_state_dict = checkpoint['model_state_dict']
        else:
            checkpoint_state_dict = checkpoint
        
        # Get the model's state dict
        model_dict = model.state_dict()
        
        # 1. Filter out unnecessary keys and keys with mismatched shapes
        pretrained_dict = {
            k: v for k, v in checkpoint_state_dict.items() 
            if k in model_dict and v.shape == model_dict[k].shape
        }
        
        # 2. Overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        
        # 3. Load the new state dict
        model.load_state_dict(model_dict)
        
        # Verify loading worked correctly
        if 'best_val_acc' in checkpoint:
            print(f"Loaded weights with validation accuracy: {checkpoint['best_val_acc']:.2f}%")
        
        return model
    
    def fine_tune_model(self, 
                       model: nn.Module, 
                       train_loader: DataLoader,
                       val_loader: DataLoader,
                       model_name: str,
                       dataset_name: str,
                       epochs: int = 10,
                       learning_rate: float = 0.001,
                       weight_decay: float = 1e-4,
                       save_best: bool = True) -> Dict[str, Any]:
        """
        Fine-tune a model on the given dataset
        
        Args:
            model: Pre-trained model to fine-tune
            train_loader: Training data loader
            val_loader: Validation data loader
            model_name: Name of the model
            dataset_name: Name of the dataset
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            save_best: Whether to save the best model
            
        Returns:
            Dictionary with training history and metrics
        """
        print(f"Starting fine-tuning of {model_name} on {dataset_name}")
        print(f"Training for {epochs} epochs with lr={learning_rate}")
        
        model = model.to(self.device)
        
        # Setup optimizer and loss criterion
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epochs': epochs,
            'best_val_acc': 0.0,
            'best_epoch': 0
        }
        
        best_val_acc = 0.0
        best_model_state = None
        
        print(f"Training on device: {self.device}")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training phase
            train_loss, train_acc = self._train_epoch(model, train_loader, optimizer, criterion)
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(model, val_loader, criterion)
            
            # Update learning rate
            scheduler.step()
            
            # Record history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                history['best_val_acc'] = best_val_acc
                history['best_epoch'] = epoch
                if save_best:
                    best_model_state = model.state_dict().copy()
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.2f}s): "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save the best model
        if save_best and best_model_state is not None:
            weights_path = self.get_finetuned_weights_path(model_name, dataset_name)
            
            # Add detailed validation for fc layer saving
            print(f"\n=== MODEL SAVING VALIDATION ===")
            print(f"Model name: {model_name}")
            print(f"Dataset: {dataset_name}")
            print(f"Total parameters in state_dict: {len(best_model_state)}")
            
            # Specifically check fc layer parameters
            fc_keys = [key for key in best_model_state.keys() if key.startswith('fc.')]
            print(f"FC layer parameters found: {fc_keys}")
            
            if fc_keys:
                for fc_key in fc_keys:
                    fc_param = best_model_state[fc_key]
                    print(f"  {fc_key}: shape={fc_param.shape}, dtype={fc_param.dtype}")
                    if 'weight' in fc_key:
                        print(f"    Weight stats: min={fc_param.min().item():.6f}, max={fc_param.max().item():.6f}, mean={fc_param.mean().item():.6f}")
                    elif 'bias' in fc_key and fc_param is not None:
                        print(f"    Bias stats: min={fc_param.min().item():.6f}, max={fc_param.max().item():.6f}, mean={fc_param.mean().item():.6f}")
            else:
                print("  WARNING: No fc layer parameters found in state_dict!")
            
            # Verify current model's fc layer before saving
            current_model_state = model.state_dict()
            current_fc_keys = [key for key in current_model_state.keys() if key.startswith('fc.')]
            print(f"Current model FC parameters: {current_fc_keys}")
            
            # Ensure the best_model_state actually contains the fc layer
            if fc_keys and current_fc_keys:
                for fc_key in fc_keys:
                    if fc_key in current_model_state:
                        current_param = current_model_state[fc_key]
                        saved_param = best_model_state[fc_key]
                        if torch.equal(current_param, saved_param):
                            print(f"  ✓ {fc_key}: Best model state matches current model")
                        else:
                            print(f"  ⚠ {fc_key}: Best model state differs from current model (this is expected)")
            
            checkpoint = {
                'model_state_dict': best_model_state,
                'history': history,
                'model_name': model_name,
                'dataset_name': dataset_name,
                'best_val_acc': best_val_acc,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'parameter_count': len(best_model_state),
                'model_size_mb': sum(p.numel() * 4 for p in best_model_state.values()) / (1024 * 1024),
                'pytorch_version': torch.__version__,
                'save_timestamp': time.time(),
                # Add fc layer specific metadata for verification
                'fc_layer_info': {
                    'fc_keys': fc_keys,
                    'fc_shapes': {key: best_model_state[key].shape for key in fc_keys} if fc_keys else {},
                    'has_fc_layer': len(fc_keys) > 0
                }
            }
            
            # Save the checkpoint
            torch.save(checkpoint, weights_path)
            print(f"Fine-tuned model saved to {weights_path}")
            print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {history['best_epoch']+1}")
            
            # Additional verification: immediately load and verify the saved model
            try:
                print(f"\n=== SAVE VERIFICATION ===")
                saved_checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
                saved_state_dict = saved_checkpoint['model_state_dict']
                saved_fc_keys = [key for key in saved_state_dict.keys() if key.startswith('fc.')]
                print(f"Verification: FC keys in saved file: {saved_fc_keys}")
                
                if 'fc_layer_info' in saved_checkpoint:
                    fc_info = saved_checkpoint['fc_layer_info']
                    print(f"Verification: FC layer metadata - has_fc_layer: {fc_info['has_fc_layer']}")
                    print(f"Verification: FC shapes in metadata: {fc_info['fc_shapes']}")
                
                # Verify shapes match
                if fc_keys and saved_fc_keys:
                    for fc_key in fc_keys:
                        if fc_key in saved_state_dict:
                            original_shape = best_model_state[fc_key].shape
                            saved_shape = saved_state_dict[fc_key].shape
                            if original_shape == saved_shape:
                                print(f"  ✓ {fc_key}: Shape verification passed ({saved_shape})")
                            else:
                                print(f"  ✗ {fc_key}: Shape mismatch! Original: {original_shape}, Saved: {saved_shape}")
                        else:
                            print(f"  ✗ {fc_key}: Missing in saved file!")
                else:
                    print("  ✗ FC layer verification failed - no FC keys found")
                    
                print(f"=== SAVE VERIFICATION COMPLETE ===\n")
                
            except Exception as e:
                print(f"Error during save verification: {e}")
            
            # Load best weights back into model
            model.load_state_dict(best_model_state)
        
        return history
    
    def _train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                    optimizer: optim.Optimizer, criterion: nn.Module) -> Tuple[float, float]:
        """Train for one epoch"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            current_acc = 100. * correct / total
            progress_bar.set_postfix({
                'Loss': f'{total_loss/(batch_idx+1):.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, model: nn.Module, val_loader: DataLoader, 
                       criterion: nn.Module) -> Tuple[float, float]:
        """Validate for one epoch"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def get_training_config(self, dataset_name: str, model_name: str) -> Dict[str, Any]:
        """Get recommended training configuration for dataset-model combination"""
        configs = {
            'cifar10': {
                'epochs': 10,
                'learning_rate': 0.0001,
                'weight_decay': 1e-4,
                'batch_size': 128
            },
            'cifar100': {
                'epochs': 20,
                'learning_rate': 0.0001,
                'weight_decay': 5e-4,
                'batch_size': 128
            },
            'imagenet': {
                'epochs': 10,
                'learning_rate': 0.0001,
                'weight_decay': 1e-4,
                'batch_size': 64
            }
        }
        
        # Adjust for model size
        if 'resnet50' in model_name or 'mobilenet_v4' in model_name:
            # Larger models may need smaller learning rates
            base_config = configs.get(dataset_name.lower(), configs['cifar10'])
            base_config = base_config.copy()
            base_config['learning_rate'] *= 0.5
            return base_config
        
        return configs.get(dataset_name.lower(), configs['cifar10'])
    
    def get_distilled_weights_path(self, student_model_name: str, teacher_model_name: str, dataset_name: str) -> str:
        """Get the path for distilled weights"""
        if self.distiller:
            return self.distiller.get_distilled_weights_path(student_model_name, teacher_model_name, dataset_name)
        return os.path.join(self.weights_dir, f"{student_model_name}_distilled_from_{teacher_model_name}_{dataset_name}.pth")
    
    def has_distilled_weights(self, student_model_name: str, teacher_model_name: str, dataset_name: str) -> bool:
        """Check if distilled weights exist for the model combination"""
        if self.distiller:
            return self.distiller.has_distilled_weights(student_model_name, teacher_model_name, dataset_name)
        weights_path = self.get_distilled_weights_path(student_model_name, teacher_model_name, dataset_name)
        return os.path.exists(weights_path)
    
    def load_distilled_weights(self, student_model: nn.Module, student_model_name: str, 
                             teacher_model_name: str, dataset_name: str) -> nn.Module:
        """Load distilled weights into the student model"""
        if self.distiller:
            return self.distiller.load_distilled_weights(student_model, student_model_name, teacher_model_name, dataset_name)
        
        weights_path = self.get_distilled_weights_path(student_model_name, teacher_model_name, dataset_name)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Distilled weights not found at {weights_path}")
        
        print(f"Loading distilled weights from {weights_path}")
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        
        if 'student_state_dict' in checkpoint:
            student_model.load_state_dict(checkpoint['student_state_dict'])
        else:
            student_model.load_state_dict(checkpoint)
        
        return student_model
    
    def distill_model(self, teacher_model: nn.Module, student_model: nn.Module,
                     teacher_model_name: str, student_model_name: str, dataset_name: str,
                     auto_config: bool = True, **kwargs) -> nn.Module:
        """
        Perform knowledge distillation from teacher to student model
        
        Args:
            teacher_model: Pre-trained teacher model
            student_model: Student model to be trained
            teacher_model_name: Name of the teacher model
            student_model_name: Name of the student model
            dataset_name: Name of the dataset
            auto_config: Whether to use automatic configuration
            **kwargs: Additional distillation parameters
            
        Returns:
            Distilled student model
        """
        if not self.enable_distillation or not self.distiller:
            print("Knowledge distillation is disabled, returning student model as-is")
            return student_model
        
        try:
            # Import here to avoid circular imports
            from .distillation import create_distillation_data_loaders
            
            # Get distillation configuration
            if auto_config:
                config = self.distiller.get_distillation_config(teacher_model_name, student_model_name, dataset_name)
                # Override with any provided kwargs
                config.update(kwargs)
            else:
                config = kwargs
            
            # Create data loaders for distillation
            train_loader, val_loader = create_distillation_data_loaders(
                dataset_name=dataset_name,
                batch_size=config.get('batch_size', 128)
            )
            
            # Perform knowledge distillation
            history = self.distiller.distill_knowledge(
                teacher_model=teacher_model,
                student_model=student_model,
                train_loader=train_loader,
                val_loader=val_loader,
                teacher_model_name=teacher_model_name,
                student_model_name=student_model_name,
                dataset_name=dataset_name,
                epochs=config.get('epochs', 15),
                learning_rate=config.get('learning_rate', 0.001),
                weight_decay=config.get('weight_decay', 1e-4),
                temperature=config.get('temperature', 4.0),
                alpha=config.get('alpha', 0.7)
            )
            
            print(f"Knowledge distillation completed. Best validation accuracy: {history['best_val_acc']:.2f}%")
            return student_model
            
        except Exception as e:
            print(f"Knowledge distillation failed: {e}")
            print("Returning student model as-is")
            return student_model


def create_fine_tuning_data_loaders(dataset_name: str, data_path: str = "./data", 
                                   batch_size: int = 128, 
                                   train_split: float = 0.8,
                                   total_samples: int = 50000) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders specifically for fine-tuning with proper train/val split
    
    Args:
        dataset_name: Name of the dataset
        data_path: Path to dataset
        batch_size: Batch size for data loaders
        train_split: Fraction of data to use for training (rest for validation)
        total_samples: Total number of samples to use (allows for larger datasets for teacher training)
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Subset
    import random
    
    # Set random seed for reproducible splits
    random.seed(42)
    
    # Define transforms
    if dataset_name.lower() == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.Resize((224, 224)),  # Resize to 224 for pretrained models
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Load full training dataset
        full_train_dataset = torchvision.datasets.CIFAR10(
            root=data_path, train=True, download=True, transform=train_transform
        )
        
        # Create validation dataset with different transform
        val_dataset_base = torchvision.datasets.CIFAR10(
            root=data_path, train=True, download=False, transform=val_transform
        )
        
    elif dataset_name.lower() == 'cifar100':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        full_train_dataset = torchvision.datasets.CIFAR100(
            root=data_path, train=True, download=True, transform=train_transform
        )
        
        val_dataset_base = torchvision.datasets.CIFAR100(
            root=data_path, train=True, download=False, transform=val_transform
        )
    
    else:
        raise ValueError(f"Dataset {dataset_name} not supported for fine-tuning")
    
    # Create train/val split from training set only
    total_train_samples = len(full_train_dataset)
    
    # Limit total samples if specified
    if total_samples < total_train_samples:
        sample_indices = random.sample(range(total_train_samples), total_samples)
    else:
        sample_indices = list(range(total_train_samples))
    
    # Split into train and validation
    split_point = int(len(sample_indices) * train_split)
    train_indices = sample_indices[:split_point]
    val_indices = sample_indices[split_point:]
    
    print(f"Fine-tuning data split: {len(train_indices)} train, {len(val_indices)} val")
    
    # Create subsets
    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(val_dataset_base, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    return train_loader, val_loader
    
```

utils/distillation.py
```python
"""
Knowledge distillation utilities for model optimization
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple, Union
import time
from tqdm import tqdm
import copy


class KnowledgeDistiller:
    """Knowledge distillation for model optimization"""
    
    def __init__(self, device: torch.device = None, weights_dir: str = "./distilled_weights"):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weights_dir = weights_dir
        os.makedirs(weights_dir, exist_ok=True)
    
    def get_distilled_weights_path(self, student_model_name: str, teacher_model_name: str, dataset_name: str) -> str:
        """Get the path for distilled weights"""
        return os.path.join(self.weights_dir, f"{student_model_name}_distilled_from_{teacher_model_name}_{dataset_name}.pth")
    
    def has_distilled_weights(self, student_model_name: str, teacher_model_name: str, dataset_name: str) -> bool:
        """Check if distilled weights exist for the model combination"""
        weights_path = self.get_distilled_weights_path(student_model_name, teacher_model_name, dataset_name)
        return os.path.exists(weights_path)
    
    def load_distilled_weights(self, student_model: nn.Module, student_model_name: str, 
                             teacher_model_name: str, dataset_name: str) -> nn.Module:
        """Load distilled weights into the student model"""
        weights_path = self.get_distilled_weights_path(student_model_name, teacher_model_name, dataset_name)
        
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Distilled weights not found at {weights_path}")
        
        print(f"Loading distilled weights from {weights_path}")
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        if 'student_state_dict' in checkpoint:
            state_dict = checkpoint['student_state_dict']
        else:
            state_dict = checkpoint
        
        # Validate parameter count
        model_params = len(student_model.state_dict())
        checkpoint_params = len(state_dict)
        if model_params != checkpoint_params:
            print(f"Warning: Parameter count mismatch - Student model: {model_params}, Checkpoint: {checkpoint_params}")
        
        # Load weights
        student_model.load_state_dict(state_dict)
        
        # Verify loading worked correctly
        if 'best_val_acc' in checkpoint:
            print(f"Loaded distilled weights with validation accuracy: {checkpoint['best_val_acc']:.2f}%")
        if 'temperature' in checkpoint and 'alpha' in checkpoint:
            print(f"Distillation parameters - Temperature: {checkpoint['temperature']}, Alpha: {checkpoint['alpha']}")
        
        return student_model
    
    def distill_knowledge(self, 
                         teacher_model: nn.Module,
                         student_model: nn.Module,
                         train_loader: DataLoader,
                         val_loader: DataLoader,
                         teacher_model_name: str,
                         student_model_name: str,
                         dataset_name: str,
                         epochs: int = 15,
                         learning_rate: float = 0.001,
                         weight_decay: float = 1e-4,
                         temperature: float = 4.0,
                         alpha: float = 0.7,
                         save_best: bool = True) -> Dict[str, Any]:
        """
        Perform knowledge distillation from teacher to student model
        
        Args:
            teacher_model: Pre-trained teacher model
            student_model: Student model to be trained
            train_loader: Training data loader
            val_loader: Validation data loader
            teacher_model_name: Name of the teacher model
            student_model_name: Name of the student model
            dataset_name: Name of the dataset
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            temperature: Temperature for softmax distillation
            alpha: Weight for distillation loss (1-alpha for hard target loss)
            save_best: Whether to save the best model
            
        Returns:
            Dictionary with training history and metrics
        """
        print(f"Starting knowledge distillation from {teacher_model_name} to {student_model_name} on {dataset_name}")
        print(f"Training for {epochs} epochs with lr={learning_rate}, T={temperature}, alpha={alpha}")
        
        # Setup models
        teacher_model = teacher_model.to(self.device)
        student_model = student_model.to(self.device)
        
        # Freeze teacher model
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        
        # Setup optimizer and loss criterion
        optimizer = optim.Adam(student_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
        criterion_ce = nn.CrossEntropyLoss()
        criterion_kd = nn.KLDivLoss(reduction='batchmean')
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'train_distill_loss': [],
            'train_ce_loss': [],
            'val_loss': [],
            'val_acc': [],
            'epochs': epochs,
            'best_val_acc': 0.0,
            'best_epoch': 0,
            'temperature': temperature,
            'alpha': alpha
        }
        
        best_val_acc = 0.0
        best_student_state = None
        
        print(f"Training on device: {self.device}")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training phase
            train_loss, train_acc, train_distill_loss, train_ce_loss = self._distillation_train_epoch(
                teacher_model, student_model, train_loader, optimizer, 
                criterion_ce, criterion_kd, temperature, alpha
            )
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(student_model, val_loader, criterion_ce)
            
            # Update learning rate
            scheduler.step()
            
            # Record history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_distill_loss'].append(train_distill_loss)
            history['train_ce_loss'].append(train_ce_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                history['best_val_acc'] = best_val_acc
                history['best_epoch'] = epoch
                if save_best:
                    best_student_state = student_model.state_dict().copy()
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.2f}s): "
                  f"Train Loss: {train_loss:.4f} (Distill: {train_distill_loss:.4f}, CE: {train_ce_loss:.4f}), "
                  f"Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save the best model
        if save_best and best_student_state is not None:
            weights_path = self.get_distilled_weights_path(student_model_name, teacher_model_name, dataset_name)
            checkpoint = {
                'student_state_dict': best_student_state,
                'history': history,
                'teacher_model_name': teacher_model_name,
                'student_model_name': student_model_name,
                'dataset_name': dataset_name,
                'best_val_acc': best_val_acc,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'temperature': temperature,
                'alpha': alpha
            }
            torch.save(checkpoint, weights_path)
            print(f"Distilled model saved to {weights_path}")
            print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {history['best_epoch']+1}")
            
            # Load best weights back into student model
            student_model.load_state_dict(best_student_state)
        
        return history
    
    def _distillation_train_epoch(self, teacher_model: nn.Module, student_model: nn.Module, 
                                train_loader: DataLoader, optimizer: optim.Optimizer,
                                criterion_ce: nn.Module, criterion_kd: nn.Module,
                                temperature: float, alpha: float) -> Tuple[float, float, float, float]:
        """Train for one epoch with knowledge distillation"""
        student_model.train()
        teacher_model.eval()
        
        total_loss = 0.0
        total_distill_loss = 0.0
        total_ce_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Distillation Training", leave=False)
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            
            # Get teacher and student outputs
            with torch.no_grad():
                teacher_outputs = teacher_model(data)
            
            student_outputs = student_model(data)
            
            # Calculate losses
            # Cross-entropy loss with hard targets
            ce_loss = criterion_ce(student_outputs, target)
            
            # Knowledge distillation loss with soft targets
            teacher_soft = F.softmax(teacher_outputs / temperature, dim=1)
            student_soft = F.log_softmax(student_outputs / temperature, dim=1)
            distill_loss = criterion_kd(student_soft, teacher_soft) * (temperature ** 2)
            
            # Combined loss
            loss = alpha * distill_loss + (1 - alpha) * ce_loss
            
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            total_distill_loss += distill_loss.item()
            total_ce_loss += ce_loss.item()
            
            pred = student_outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            current_acc = 100. * correct / total
            progress_bar.set_postfix({
                'Loss': f'{total_loss/(batch_idx+1):.4f}',
                'Acc': f'{current_acc:.2f}%',
                'Distill': f'{total_distill_loss/(batch_idx+1):.4f}',
                'CE': f'{total_ce_loss/(batch_idx+1):.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_distill_loss = total_distill_loss / len(train_loader)
        avg_ce_loss = total_ce_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy, avg_distill_loss, avg_ce_loss
    
    def _validate_epoch(self, model: nn.Module, val_loader: DataLoader, 
                       criterion: nn.Module) -> Tuple[float, float]:
        """Validate for one epoch"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def get_distillation_config(self, teacher_model_name: str, student_model_name: str, 
                               dataset_name: str) -> Dict[str, Any]:
        """Get recommended distillation configuration for model-dataset combination"""
        configs = {
            'cifar10': {
                'epochs': 15,
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'batch_size': 128,
                'temperature': 4.0,
                'alpha': 0.7
            },
            'cifar100': {
                'epochs': 20,
                'learning_rate': 0.0005,
                'weight_decay': 5e-4,
                'batch_size': 128,
                'temperature': 6.0,
                'alpha': 0.8
            },
            'imagenet': {
                'epochs': 15,
                'learning_rate': 0.0005,
                'weight_decay': 1e-4,
                'batch_size': 64,
                'temperature': 5.0,
                'alpha': 0.7
            }
        }
        
        base_config = configs.get(dataset_name.lower(), configs['cifar10']).copy()
        
        # Adjust based on model complexity difference
        if 'resnet50' in teacher_model_name and 'resnet18' in student_model_name:
            # Larger teacher-student gap, use higher temperature and alpha
            base_config['temperature'] *= 1.2
            base_config['alpha'] = min(0.9, base_config['alpha'] + 0.1)
        elif 'mobilenet_v4' in teacher_model_name and 'mobilenet_v3' in student_model_name:
            # Moderate teacher-student gap
            base_config['temperature'] *= 1.1
            base_config['alpha'] = min(0.8, base_config['alpha'] + 0.05)
        
        # Adjust for quantizable models (they might need more gentle training)
        if 'quantizable' in student_model_name:
            base_config['learning_rate'] *= 0.8
            base_config['epochs'] = max(base_config['epochs'], 20)
        
        return base_config


def create_distillation_data_loaders(dataset_name: str, data_path: str = "./data", 
                                   batch_size: int = 128, 
                                   train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders specifically for knowledge distillation with proper train/val split
    
    Args:
        dataset_name: Name of the dataset
        data_path: Path to dataset
        batch_size: Batch size for data loaders
        train_split: Fraction of data to use for training (rest for validation)
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Import here to avoid circular imports
    try:
        from .fine_tuning import create_fine_tuning_data_loaders
    except ImportError:
        # Fallback import
        import sys
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from utils.fine_tuning import create_fine_tuning_data_loaders

    # Use the same proper data splitting as fine-tuning
    return create_fine_tuning_data_loaders(
        dataset_name=dataset_name,
        data_path=data_path,
        batch_size=batch_size,
        train_split=train_split,
        total_samples=50000  # Standard size for distillation
    )


def get_compatible_teacher_models(student_model_name: str) -> list:
    """
    Get list of compatible teacher models for a given student model
    
    Args:
        student_model_name: Name of the student model
        
    Returns:
        List of compatible teacher model names
    """
    # Define teacher-student compatibility mappings
    compatibility_map = {
        # ResNet students
        'resnet18': ['resnet50', 'resnet50_quantizable'],
        'resnet18_quantizable': ['resnet50', 'resnet50_quantizable', 'resnet18'],
        'resnet18_low_rank': ['resnet50', 'resnet50_quantizable', 'resnet18', 'resnet18_quantizable'],
        
        # MobileNet students
        'mobilenet_v3_small': ['mobilenet_v3_large', 'mobilenet_v4_conv_medium', 'mobilenet_v4_conv_large'],
        'mobilenet_v3_small_quantizable': ['mobilenet_v3_large', 'mobilenet_v3_large_quantizable', 
                                          'mobilenet_v4_conv_medium', 'mobilenet_v4_conv_large'],
        'mobilenet_v3_large': ['mobilenet_v4_conv_medium', 'mobilenet_v4_conv_large'],
        'mobilenet_v3_large_quantizable': ['mobilenet_v3_large', 'mobilenet_v4_conv_medium', 'mobilenet_v4_conv_large'],
        'mobilenet_v4_conv_small': ['mobilenet_v4_conv_medium', 'mobilenet_v4_conv_large'],
        'mobilenet_v4_conv_medium': ['mobilenet_v4_conv_large'],
        
        # Cross-architecture possibilities (ResNet -> MobileNet is generally not recommended due to different architectures)
        # But we can allow some flexibility
        'mobilenet_v2': ['mobilenet_v3_large', 'mobilenet_v4_conv_medium', 'mobilenet_v4_conv_large'],
    }
    
    return compatibility_map.get(student_model_name, [])


def auto_select_teacher_model(student_model_name: str, available_models: list = None) -> Optional[str]:
    """
    Automatically select the best teacher model for a given student model
    
    Args:
        student_model_name: Name of the student model
        available_models: List of available models (if None, uses all compatible models)
        
    Returns:
        Best teacher model name or None if no suitable teacher found
    """
    compatible_teachers = get_compatible_teacher_models(student_model_name)
    
    if available_models:
        # Filter compatible teachers by available models
        compatible_teachers = [model for model in compatible_teachers if model in available_models]
    
    if not compatible_teachers:
        return None
    
    # Priority ranking for teacher selection (larger/more complex models first)
    priority_order = [
        'resnet50_quantizable', 'resnet50',
        'mobilenet_v4_conv_large', 'mobilenet_v4_conv_medium',
        'mobilenet_v3_large_quantizable', 'mobilenet_v3_large',
        'resnet18_quantizable', 'resnet18',
        'mobilenet_v4_conv_small', 'mobilenet_v2'
    ]
    
    # Select the highest priority teacher from compatible ones
    for teacher in priority_order:
        if teacher in compatible_teachers:
            return teacher
    
    # Fallback to first available compatible teacher
    return compatible_teachers[0] if compatible_teachers else None

```

quantization/quantizers.py
```python
"""
Quantization methods implementation for PyTorch models
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.quantization import quantize_dynamic, prepare, convert
from torch.quantization import QConfig, default_observer, default_weight_observer, get_default_qat_qconfig
import torch.ao.quantization as ao_quantization
from torch.ao.quantization import get_default_qconfig_mapping, QConfigMapping
from torch.ao.quantization import get_default_qat_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, prepare_qat_fx
import torch.quantization.quantize_fx as quantize_fx
import copy
import time
import traceback
from typing import Dict, List, Any, Tuple
import tempfile
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm


class BaseQuantizer:
    """Base class for all quantization methods"""

    def __init__(self, device: torch.device = None, backend: str = 'qnnpack', model_name: str = None):
        self.device = device or torch.device('cpu')
        self.quantized_model = None
        self.original_model = None
        self.backend = backend
        self.model_name = model_name

        # 设置torch.ao后端支持
        self._setup_ao_backend()

    def _setup_ao_backend(self):
        """设置torch.ao量化后端"""
        if self.device.type == 'cuda':
            # CUDA设备使用fbgemm后端
            torch.backends.quantized.engine = 'fbgemm'
        elif self.device.type == 'mps':
            # MPS设备使用qnnpack后端
            torch.backends.quantized.engine = 'qnnpack'
        else:
            # CPU使用qnnpack后端
            torch.backends.quantized.engine = 'qnnpack'

    def _get_device_compatible_model(self, model: nn.Module) -> nn.Module:
        """获取与设备兼容的模型"""
        if hasattr(model, '_modules') and any('quantized' in str(type(m)) for m in model.modules()):
            # 量化模型需要特殊处理
            if self.device.type in ['cuda', 'mps']:
                # 对于GPU设备，将量化模型转换为可在GPU上运行的格式
                return self._convert_quantized_for_device(model)
        return model.to(self.device)

    def _convert_quantized_for_device(self, quantized_model: nn.Module) -> nn.Module:
        """将量化模型转换为设备兼容格式"""
        if self.device.type == 'cuda':
            # CUDA设备支持
            try:
                # 使用torch.jit.script将量化模型转换为可在CUDA上运行的格式
                scripted_model = torch.jit.script(quantized_model)
                return scripted_model.to(self.device)
            except:
                # 如果script失败，尝试使用torch.ao的转换
                return self._ao_device_conversion(quantized_model)
        elif self.device.type == 'mps':
            # MPS设备支持
            return self._ao_device_conversion(quantized_model)
        return quantized_model

    def _ao_device_conversion(self, model: nn.Module) -> nn.Module:
        """使用torch.ao进行设备转换"""
        try:
            # 创建设备兼容的模型包装器
            class QuantizedModelWrapper(nn.Module):
                def __init__(self, quantized_model, target_device):
                    super().__init__()
                    self.quantized_model = quantized_model.cpu()  # 量化模型保持在CPU
                    self.target_device = target_device

                def forward(self, x):
                    # 输入数据移到CPU进行量化推理
                    x_cpu = x.cpu()
                    output = self.quantized_model(x_cpu)
                    # 输出移回目标设备
                    return output.to(self.target_device)

            return QuantizedModelWrapper(model, self.device)
        except Exception as e:
            print(
                f"Warning: Device conversion failed: {e}, using CPU inference")
            return model

    def quantize(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader = None) -> nn.Module:
        """
        Quantize the model

        Args:
            model: Original model to quantize
            calibration_data: Calibration data loader for static quantization or training data for QAT

        Returns:
            Quantized model
        """
        raise NotImplementedError("Subclasses must implement quantize method")

    def evaluate_model(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Evaluate model performance

        Args:
            model: Model to evaluate
            data_loader: Data loader for evaluation

        Returns:
            Dictionary with evaluation metrics
        """
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        model_device = self.device
        model = model.to(model_device)  # Ensure model is on the correct device

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(
                    model_device), targets.to(model_device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)  # Accumulate total loss correctly

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = 100. * correct / total if total > 0 else 0.0
        # Calculate average loss correctly
        avg_loss = total_loss / total if total > 0 else 0.0

        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'correct': correct,
            'total': total
        }

    def measure_inference_time(self, model: nn.Module, input_tensor: torch.Tensor,
                               warmup_iterations: int = 50, benchmark_iterations: int = 500) -> Dict[str, float]:
        """
        Measure model inference time

        Args:
            model: Model to benchmark
            input_tensor: Input tensor for inference
            warmup_iterations: Number of warmup iterations
            benchmark_iterations: Number of benchmark iterations

        Returns:
            Dictionary with timing statistics
        """
        model.eval()
        model_device = self.device
        model = model.to(model_device)
        input_tensor = input_tensor.to(model_device)

        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(input_tensor)

        times = []
        with torch.no_grad():
            for _ in range(benchmark_iterations):
                if model_device.type == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.time()

                _ = model(input_tensor)

                if model_device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)

        if not times:
            return {'mean_time_ms': 0.0, 'min_time_ms': 0.0, 'max_time_ms': 0.0, 'std_time_ms': 0.0}

        mean_time = sum(times) / len(times)
        std_time = (sum([(t - mean_time)**2 for t in times]) /
                    len(times))**0.5 if len(times) > 1 else 0.0
        return {
            'mean_time_ms': mean_time,
            'min_time_ms': min(times),
            'max_time_ms': max(times),
            'std_time_ms': std_time
        }


class DynamicQuantizer(BaseQuantizer):
    """Dynamic quantization implementation"""

    def __init__(self, device: torch.device = None, qconfig_spec: Dict = None, **kwargs):
        super().__init__(device, **kwargs)
        self.qconfig_spec = qconfig_spec or {nn.Linear, nn.Conv2d}

    def quantize(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader = None) -> nn.Module:
        self.original_model = copy.deepcopy(model)
        model_copy = copy.deepcopy(model)
        model_copy = model_copy.to('cpu')
        quantized_model = torch.quantization.quantize_dynamic(
            model_copy, self.qconfig_spec, dtype=torch.qint8
        )
        self.quantized_model = quantized_model
        return quantized_model

    def evaluate_model(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        model = model.to('cpu')

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to('cpu'), targets.to('cpu')
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = 100. * correct / total
        avg_loss = total_loss / len(data_loader)
        return {'accuracy': accuracy, 'loss': avg_loss, 'correct': correct, 'total': total}

    def measure_inference_time(self, model: nn.Module, input_tensor: torch.Tensor,
                               warmup_iterations: int = 50, benchmark_iterations: int = 500) -> Dict[str, float]:
        model.eval()
        model = model.to('cpu')
        input_tensor = input_tensor.to('cpu')

        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(input_tensor)

        times = []
        with torch.no_grad():
            for _ in range(benchmark_iterations):
                start_time = time.time()
                _ = model(input_tensor)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)

        return {
            'mean_time_ms': sum(times) / len(times),
            'min_time_ms': min(times),
            'max_time_ms': max(times),
            'std_time_ms': (sum([(t - sum(times)/len(times))**2 for t in times]) / len(times))**0.5
        }


class StaticQuantizer(BaseQuantizer):
    """Static quantization implementation with torch.ao support"""

    def __init__(self, device: torch.device = None, backend: str = None, **kwargs):
        super().__init__(device, **kwargs)
        if backend:
            self.backend = backend
        else:
            self.backend = 'fbgemm' if device and device.type == 'cuda' else 'qnnpack'
        self._setup_ao_config()

    def _setup_ao_config(self):
        self.qconfig_mapping = get_default_qconfig_mapping(self.backend)

    def quantize(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader = None) -> nn.Module:
        self.original_model = copy.deepcopy(model)
        model_copy = copy.deepcopy(model)
        try:
            return self._quantize_with_ao_fx(model_copy, calibration_data)
        except Exception as e:
            print(
                f"Warning: torch.ao FX quantization failed: {e}\nFalling back to legacy quantization")
            return self._quantize_legacy(model_copy, calibration_data)

    def _quantize_with_ao_fx(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader) -> nn.Module:
        model = model.to('cpu')
        model.eval()
        example_inputs = self._get_example_inputs(calibration_data)
        prepared_model = prepare_fx(
            model, self.qconfig_mapping, example_inputs)
        self._calibrate_model(prepared_model, calibration_data)
        quantized_model = convert_fx(prepared_model)
        self.quantized_model = self._get_device_compatible_model(
            quantized_model)
        return self.quantized_model

    def _get_example_inputs(self, calibration_data: torch.utils.data.DataLoader) -> tuple:
        for inputs, _ in calibration_data:
            return (inputs[:1].cpu(),)
        raise ValueError("Calibration data is empty")

    def _calibrate_model(self, prepared_model: nn.Module, calibration_data: torch.utils.data.DataLoader):
        prepared_model.eval()
        with torch.no_grad():
            for i, (inputs, _) in enumerate(calibration_data):
                prepared_model(inputs.cpu())
                if i >= 100:
                    break

    def _quantize_legacy(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader) -> nn.Module:
        if calibration_data is None:
            raise ValueError(
                "Calibration data is required for static quantization")
        model.eval()
        model = model.to('cpu')
        model.qconfig = torch.quantization.get_default_qconfig(self.backend)
        model_prepared = torch.quantization.prepare(model)
        self._calibrate(model_prepared, calibration_data)
        quantized_model = torch.quantization.convert(model_prepared)
        self.quantized_model = quantized_model
        return quantized_model

    def _calibrate(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader):
        model.eval()
        with torch.no_grad():
            for inputs, _ in calibration_data:
                model(inputs.to('cpu'))

    def evaluate_model(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        model.eval()
        correct, total, total_loss = 0, 0, 0.0
        criterion = nn.CrossEntropyLoss().to(self.device)
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)
                outputs = model(inputs)
                if outputs.device != targets.device:
                    outputs = outputs.to(targets.device)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        accuracy = 100. * correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0
        return {'accuracy': accuracy, 'loss': avg_loss, 'correct': correct, 'total': total}


class QATQuantizer(BaseQuantizer):
    """Quantization Aware Training implementation with torch.ao support and optional knowledge distillation."""

    def __init__(self, device: torch.device = None, backend: str = None, train_epochs: int = 3, learning_rate: float = 1e-4, **kwargs):
        super().__init__(device, **kwargs)
        self.backend = backend or (
            'fbgemm' if device and device.type == 'cuda' else 'qnnpack')
        self.train_epochs = train_epochs
        self.learning_rate = learning_rate
        self._setup_ao_qat_config()

    def _setup_ao_qat_config(self):
        """Sets up torch.ao QAT configuration."""
        self.qconfig_mapping = get_default_qat_qconfig_mapping(self.backend)

    def quantize(self, model: nn.Module, calibration_data: DataLoader = None,
                 teacher_model: nn.Module = None, distillation_params: Dict[str, Any] = None) -> nn.Module:
        """
        Apply QAT, with optional knowledge distillation.
        """
        self.original_model = copy.deepcopy(model)
        model_copy = copy.deepcopy(model)
        try:
            return self._quantize_with_ao_qat_fx(model_copy, calibration_data, teacher_model, distillation_params)
        except Exception as e:
            print(f"Warning: torch.ao QAT FX failed: {e}",
                  "\nFalling back to legacy QAT (distillation not supported in fallback)")
            traceback.print_exc()
            return self._quantize_legacy_qat(model_copy, calibration_data)

    def _quantize_with_ao_qat_fx(self, model: nn.Module, training_data: DataLoader,
                                 teacher_model: nn.Module = None, distillation_params: Dict[str, Any] = None) -> nn.Module:
        """Uses torch.ao FX graph mode for QAT, with optional distillation."""
        model = model.to(self.device)
        model.train()

        example_inputs = self._get_example_inputs(training_data)
        prepared_model = prepare_qat_fx(
            model, self.qconfig_mapping, example_inputs)

        trained_model = self._perform_qat_training(
            model=prepared_model,
            training_data=training_data,
            epochs=self.train_epochs,
            teacher_model=teacher_model,
            distillation_params=distillation_params
        )

        trained_model.eval().cpu()
        quantized_model = convert_fx(trained_model)

        self.quantized_model = self._get_device_compatible_model(
            quantized_model)
        return self.quantized_model

    def _perform_qat_training(self, model: nn.Module, training_data: DataLoader,
                              epochs: int, teacher_model: nn.Module = None,
                              distillation_params: Dict[str, Any] = None) -> nn.Module:
        """Performs QAT fine-tuning, using distillation if a teacher model is provided."""
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        is_distillation = teacher_model is not None and distillation_params is not None

        if is_distillation:
            print(
                f"Starting QAT with Knowledge Distillation for {epochs} epochs on {self.device}")
            teacher_model.to(self.device).eval()
        else:
            print(
                f"Starting standard QAT training for {epochs} epochs on {self.device}")

        for epoch in range(epochs):
            if is_distillation:
                avg_loss, _, _ = self._distillation_qat_train_epoch(
                    teacher_model, model, training_data, optimizer, distillation_params
                )
                print(
                    f"Distill-QAT Epoch {epoch+1}/{epochs}, Avg Combined Loss: {avg_loss:.4f}")
            else:
                avg_loss = self._standard_qat_train_epoch(
                    model, training_data, optimizer)
                print(
                    f"Standard QAT Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

        return model

    def _standard_qat_train_epoch(self, model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer) -> float:
        """Train for one standard QAT epoch."""
        model.train()
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss().to(self.device)
        progress_bar = tqdm(train_loader, desc="Standard QAT", leave=False)

        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        return total_loss / len(train_loader)

    def _distillation_qat_train_epoch(self, teacher_model: nn.Module, student_model: nn.Module,
                                      train_loader: DataLoader, optimizer: optim.Optimizer,
                                      distillation_params: Dict[str, Any]) -> Tuple[float, float, float]:
        """Train for one epoch with knowledge distillation during QAT."""
        student_model.train()
        criterion_ce = nn.CrossEntropyLoss()
        criterion_kd = nn.KLDivLoss(reduction='batchmean')
        alpha = distillation_params.get('alpha', 0.7)
        temperature = distillation_params.get('temperature', 4.0)
        total_loss, total_distill_loss, total_ce_loss = 0.0, 0.0, 0.0
        progress_bar = tqdm(train_loader, desc="Distill-QAT", leave=False)

        for data, target in progress_bar:
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_outputs = teacher_model(data)

            student_outputs = student_model(data)
            ce_loss = criterion_ce(student_outputs, target)
            distill_loss = criterion_kd(
                F.log_softmax(student_outputs / temperature, dim=1),
                F.softmax(teacher_outputs / temperature, dim=1)
            ) * (temperature ** 2)

            loss = alpha * distill_loss + (1 - alpha) * ce_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_distill_loss += distill_loss.item()
            total_ce_loss += ce_loss.item()
            progress_bar.set_postfix(
                {'Loss': f'{loss.item():.4f}', 'KD': f'{distill_loss.item():.4f}', 'CE': f'{ce_loss.item():.4f}'})

        return (total_loss / len(train_loader),
                total_distill_loss / len(train_loader),
                total_ce_loss / len(train_loader))

    def _get_example_inputs(self, data_loader: DataLoader) -> tuple:
        for inputs, _ in data_loader:
            return (inputs[:1].to(self.device),)
        raise ValueError("Training data is empty")

    def _quantize_legacy_qat(self, model: nn.Module, training_data: DataLoader) -> nn.Module:
        if training_data is None:
            raise ValueError("Training data is required for QAT")
        model.train()
        model.qconfig = torch.quantization.get_default_qat_qconfig(
            self.backend)
        torch.quantization.prepare_qat(model, inplace=True)
        self._perform_qat_training(model, training_data, self.train_epochs)
        model.eval()
        quantized_model = torch.quantization.convert(model)
        self.quantized_model = quantized_model
        return quantized_model


class FXQuantizer(BaseQuantizer):
    """FX Graph Mode quantization implementation"""

    def __init__(self, device: torch.device = None, backend: str = 'fbgemm', **kwargs):
        super().__init__(device, **kwargs)
        self.backend = backend
        torch.backends.quantized.engine = backend

    def quantize(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader) -> nn.Module:
        if calibration_data is None:
            raise ValueError(
                "Calibration data is required for FX quantization")
        self.original_model = copy.deepcopy(model)
        model_copy = copy.deepcopy(model).eval()
        qconfig_mapping = get_default_qconfig_mapping(self.backend)
        example_inputs = next(iter(calibration_data))[0][:1].to(self.device)
        model_prepared = prepare_fx(
            model_copy, qconfig_mapping, example_inputs)
        self._calibrate(model_prepared, calibration_data)
        quantized_model = convert_fx(model_prepared)
        self.quantized_model = quantized_model
        return quantized_model

    def _calibrate(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader):
        model.eval()
        with torch.no_grad():
            for inputs, _ in calibration_data:
                model(inputs.to(self.device))


class INT8Quantizer(BaseQuantizer):
    """INT8 quantization using custom observers"""

    def quantize(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader) -> nn.Module:
        if calibration_data is None:
            raise ValueError(
                "Calibration data is required for INT8 quantization")
        self.original_model = copy.deepcopy(model)
        model_copy = copy.deepcopy(model).eval()
        custom_qconfig = QConfig(
            activation=default_observer.with_args(dtype=torch.qint8),
            weight=default_weight_observer.with_args(dtype=torch.qint8)
        )
        model_copy.qconfig = custom_qconfig
        model_prepared = torch.quantization.prepare(model_copy)
        self._calibrate(model_prepared, calibration_data)
        quantized_model = torch.quantization.convert(model_prepared)
        self.quantized_model = quantized_model
        return quantized_model

    def _calibrate(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader):
        model.eval()
        with torch.no_grad():
            for inputs, _ in calibration_data:
                model(inputs.to(self.device))


class QuantizationBenchmark(BaseQuantizer):
    """Benchmark multiple quantization methods"""

    def __init__(self, device: torch.device = None, backend: str = 'fbgemm', **kwargs):
        super().__init__(device, **kwargs)
        self.device = device or torch.device('cpu')
        self.backend = backend
        self.results = {}
        self.quantizers = {}
        self.model_name = kwargs.get('model_name', 'unknown')

    def _load_original_resnet_for_lrf(self, lrf_model: nn.Module) -> nn.Module:
        from models.model_loader import ModelLoader
        original_model_name = 'resnet18_quantizable' if 'resnet18' in self.model_name.lower(
        ) else 'resnet50_quantizable'
        num_classes = lrf_model.fc.out_features if hasattr(
            lrf_model, 'fc') and hasattr(lrf_model.fc, 'out_features') else 1000
        try:
            model_loader = ModelLoader(
                num_classes=num_classes, device=self.device, enable_finetuning=False)
            original_model = model_loader.load_model(
                original_model_name, pretrained=True).to(self.device).eval()
            print(
                f"Successfully loaded {original_model_name} as baseline for LRF comparison")
            return original_model
        except Exception as e:
            print(
                f"Error loading original ResNet {original_model_name}: {e}\nUsing LRF model as baseline instead")
            return lrf_model

    def benchmark_quantization_methods(self, model: nn.Module,
                                       calibration_data: DataLoader,
                                       evaluation_data: DataLoader,
                                       methods: list = None,
                                       teacher_model: nn.Module = None,
                                       args: Any = None) -> Dict[str, Dict[str, Any]]:
        if methods is None:
            methods = ['dynamic', 'static', 'qat']
        self.results = {}
        example_input = next(iter(calibration_data))[0][:1]

        print("Evaluating original (unquantized) model...")
        original_model_to_eval, lrf_model = (self._load_original_resnet_for_lrf(
            model), model) if 'low_rank' in self.model_name else (model, None)

        original_eval = self.evaluate_model(
            original_model_to_eval, evaluation_data)
        original_time = self.measure_inference_time(
            original_model_to_eval, example_input)
        original_size = self._get_model_size(original_model_to_eval)
        self.results['original'] = {'accuracy': original_eval['accuracy'],
                                    'loss': original_eval['loss'], 'model_size_mb': original_size, **original_time}

        if lrf_model:
            print("\nEvaluating LRF (unquantized) model...")
            lrf_eval = self.evaluate_model(lrf_model, evaluation_data)
            lrf_time = self.measure_inference_time(lrf_model, example_input)
            lrf_size = self._get_model_size(lrf_model)
            self.results['lrf'] = {'accuracy': lrf_eval['accuracy'],
                                   'loss': lrf_eval['loss'], 'model_size_mb': lrf_size, **lrf_time}

        self.quantizers = get_available_quantizers(
            backend=self.backend, device=self.device, model_name=self.model_name)
        model_to_quantize = lrf_model if lrf_model else model

        for method_name in methods:
            quantizer = self.quantizers[method_name]
            result_key = f"lrf_{method_name}" if lrf_model else method_name
            print(f"\nBenchmarking {result_key} quantization...")

            try:
                if method_name == 'qat' and args and args.enable_distillation and teacher_model:
                    print("-> Activating QAT with Knowledge Distillation.")
                    from utils.distillation import KnowledgeDistiller, auto_select_teacher_model
                    from models.model_loader import get_available_models
                    teacher_name = args.teacher_model or auto_select_teacher_model(
                        self.model_name, get_available_models())
                    dist_config = KnowledgeDistiller().get_distillation_config(
                        teacher_name, self.model_name, args.dataset)
                    print(f"-> Using distillation config: {dist_config}")
                    quantized_model = quantizer.quantize(copy.deepcopy(
                        model_to_quantize), calibration_data, teacher_model, dist_config)
                else:
                    quantized_model = quantizer.quantize(
                        copy.deepcopy(model_to_quantize), calibration_data)

                eval_metrics = quantizer.evaluate_model(
                    quantized_model, evaluation_data)
                time_stats = quantizer.measure_inference_time(
                    quantized_model, example_input)
                model_size_mb = self._get_model_size(quantized_model)
                speedup = self.results['original']['mean_time_ms'] / \
                    time_stats['mean_time_ms'] if time_stats['mean_time_ms'] > 0 else 0
                compression = self.results['original']['model_size_mb'] / \
                    model_size_mb if model_size_mb > 0 else 0

                self.results[result_key] = {'accuracy': eval_metrics['accuracy'], 'loss': eval_metrics['loss'],
                                            'model_size_mb': model_size_mb, 'speedup': speedup, 'compression_ratio': compression, **time_stats}

            except Exception as e:
                print(f"Error benchmarking {method_name}: {e}")
                traceback.print_exc()
                self.results[result_key] = {'error': str(e)}

        return self.results

    def _get_model_size(self, model: nn.Module) -> float:
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            torch.save(model.state_dict(), temp_file.name)
            return os.path.getsize(temp_file.name) / (1024 * 1024)

    def print_results(self):
        if not self.results:
            print("No results to display.")
            return
        print("\n" + "="*80)
        print("QUANTIZATION BENCHMARK RESULTS")
        print("="*80)
        print(f"{'Method':<15} {'Accuracy (%)':<15} {'Time (ms)':<15} {'Size (MB)':<15} {'Speedup':<10} {'Compression':<12}")
        print("-"*80)

        for method, res in self.results.items():
            if 'error' in res:
                print(
                    f"{method:<15} {'ERROR':<15} {'-':<15} {'-':<15} {'-':<10} {'-':<12}")
                continue
            acc = f"{res.get('accuracy', 0.0):.2f}"
            time_ms = f"{res.get('mean_time_ms', 0.0):.2f}"
            size_mb = f"{res.get('model_size_mb', 0.0):.2f}"
            speedup = f"{res.get('speedup', 1.0):.2f}x"
            comp = f"{res.get('compression_ratio', 1.0):.2f}x"
            print(
                f"{method:<15} {acc:<15} {time_ms:<15} {size_mb:<15} {speedup:<10} {comp:<12}")
        print("="*80)


class OfficialQuantizedQuantizer(BaseQuantizer):
    """Official pre-trained quantized models from torchvision.models.quantization"""

    def __init__(self, device: torch.device = None, **kwargs):
        super().__init__(device, **kwargs)
        self.device = torch.device('cpu')

    def quantize(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader = None) -> nn.Module:
        model_name = self._get_model_name(model)
        quantized_model = self._load_official_quantized_model(model_name)
        if quantized_model is None:
            raise ValueError(
                f"No official quantized model available for {model_name}")
        self.quantized_model = quantized_model.to('cpu').eval()
        return self.quantized_model

    def _get_model_name(self, model: nn.Module) -> str:
        return self.model_name

    def _load_official_quantized_model(self, model_name: str) -> nn.Module:
        from torchvision.models import quantization
        model_map = {
            'resnet18': quantization.resnet18,
            'resnet50': quantization.resnet50,
            'mobilenet_v2': quantization.mobilenet_v2,
            'mobilenet_v3_large': quantization.mobilenet_v3_large,
        }
        if model_name.startswith('mobilenet_v3_small'):
            print(
                "Warning: official mobilenet_v3_small not available, using large instead.")
            model_name = 'mobilenet_v3_large'

        quant_model_fn = next(
            (v for k, v in model_map.items() if model_name.startswith(k)), None)
        if quant_model_fn:
            return quant_model_fn(pretrained=True, quantize=True)
        return None

    def evaluate_model(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        return DynamicQuantizer().evaluate_model(model, data_loader)

    def measure_inference_time(self, model: nn.Module, input_tensor: torch.Tensor,
                               warmup_iterations: int = 50, benchmark_iterations: int = 500) -> Dict[str, float]:
        return DynamicQuantizer().measure_inference_time(model, input_tensor, warmup_iterations, benchmark_iterations)


def get_available_quantizers(backend: str = 'qnnpack', device: torch.device = None, model_name: str = None) -> Dict[str, BaseQuantizer]:
    """Get dictionary of available quantizers"""
    device = device or torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    return {
        'dynamic': DynamicQuantizer(device=device, backend=backend, model_name=model_name),
        'static': StaticQuantizer(device=device, backend=backend, model_name=model_name),
        'qat': QATQuantizer(device=device, backend=backend, train_epochs=3, learning_rate=1e-5, model_name=model_name),
        'fx': FXQuantizer(device=device, backend=backend, model_name=model_name),
        'int8': INT8Quantizer(device=device, backend=backend, model_name=model_name),
        'official': OfficialQuantizedQuantizer(device=device, backend=backend, model_name=model_name),
    }

```


当前有些测试结果会异常地高，可能是为什么？