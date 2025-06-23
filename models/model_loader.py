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