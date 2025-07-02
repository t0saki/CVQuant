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
        'resnet18': ['resnet50', 'resnet50_quantizable', 'efficientnetv2_s', 'efficientnetv2_m'],
        'resnet18_quantizable': ['resnet50', 'resnet50_quantizable', 'resnet18', 'efficientnetv2_s', 'efficientnetv2_m'],
        'resnet18_low_rank': ['resnet50', 'resnet50_quantizable', 'resnet18', 'resnet18_quantizable', 'efficientnetv2_s', 'efficientnetv2_m'],
        
        # MobileNet students
        'mobilenet_v3_small': ['mobilenet_v3_large', 'mobilenet_v4_conv_medium', 'mobilenet_v4_conv_large', 'efficientnetv2_s', 'efficientnetv2_m'],
        'mobilenet_v3_small_quantizable': ['mobilenet_v3_large', 'mobilenet_v3_large_quantizable', 
                                          'mobilenet_v4_conv_medium', 'mobilenet_v4_conv_large', 'efficientnetv2_s', 'efficientnetv2_m'],
        'mobilenet_v3_large': ['mobilenet_v4_conv_medium', 'mobilenet_v4_conv_large', 'efficientnetv2_m', 'efficientnetv2_l'],
        'mobilenet_v3_large_quantizable': ['mobilenet_v3_large', 'mobilenet_v4_conv_medium', 'mobilenet_v4_conv_large', 'efficientnetv2_m', 'efficientnetv2_l'],
        'mobilenet_v4_conv_small': ['mobilenet_v4_conv_medium', 'mobilenet_v4_conv_large', 'efficientnetv2_s', 'efficientnetv2_m'],
        'mobilenet_v4_conv_medium': ['mobilenet_v4_conv_large', 'efficientnetv2_m', 'efficientnetv2_l'],
        
        # Cross-architecture possibilities (ResNet -> MobileNet is generally not recommended due to different architectures)
        # But we can allow some flexibility
        'mobilenet_v2': ['mobilenet_v3_large', 'mobilenet_v4_conv_medium', 'mobilenet_v4_conv_large', 'efficientnetv2_s', 'efficientnetv2_m'],
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
        'efficientnetv2_xl', 'efficientnetv2_l', 'efficientnetv2_m', 'efficientnetv2_s',
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