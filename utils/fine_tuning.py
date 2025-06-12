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
    
    def __init__(self, device: torch.device = None, weights_dir: str = "./fine_tuned_weights"):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weights_dir = weights_dir
        os.makedirs(weights_dir, exist_ok=True)
    
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
        checkpoint = torch.load(weights_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
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
            checkpoint = {
                'model_state_dict': best_model_state,
                'history': history,
                'model_name': model_name,
                'dataset_name': dataset_name,
                'best_val_acc': best_val_acc,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay
            }
            torch.save(checkpoint, weights_path)
            print(f"Fine-tuned model saved to {weights_path}")
            print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {history['best_epoch']+1}")
            
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


def create_fine_tuning_data_loaders(dataset_name: str, data_path: str = "./data", 
                                   batch_size: int = 128, 
                                   train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders specifically for fine-tuning
    
    Args:
        dataset_name: Name of the dataset
        data_path: Path to dataset
        batch_size: Batch size for data loaders
        train_split: Fraction of data to use for training (rest for validation)
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    from .data_loader import create_data_loaders

    sample_size = 50000
    
    # For fine-tuning, we need separate train and validation sets
    # We'll use the calibration loader as training and evaluation loader as validation
    train_loader, val_loader = create_data_loaders(
        dataset_name=dataset_name,
        data_path=data_path,
        batch_size=batch_size,
        calibration_size=int(sample_size * train_split),  # Use more data for training
        evaluation_size=int(sample_size * (1 - train_split)),  # Rest for validation
        input_size=224
    )
    
    return train_loader, val_loader
