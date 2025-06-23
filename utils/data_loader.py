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
                batch_size=batch_size, train=False, input_size=input_size)
        elif dataset_name.lower() == 'cifar100':
            # --- 传递 input_size ---
            full_loader = self.get_cifar100_dataset(
                batch_size=batch_size, train=False, input_size=input_size)
        elif dataset_name.lower() == 'imagenet':
            full_loader = self.get_imagenet_subset(
                batch_size=batch_size, subset_size=calibration_size,
                input_size=input_size, train=False
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