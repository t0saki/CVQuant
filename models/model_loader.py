"""
Model loader for ResNet and MobileNet models
"""
import torch
import torch.nn as nn
import torchvision.models as models
import timm
from typing import Dict, Any, Optional

class ModelLoader:
    """Load and prepare models for quantization experiments"""
    
    def __init__(self, num_classes: int = 1000):
        self.num_classes = num_classes
        self.available_models = {
            'resnet18': self._load_resnet18,
            'resnet50': self._load_resnet50,
            'mobilenet_v2': self._load_mobilenet_v2,
            'mobilenet_v3_large': self._load_mobilenet_v3_large,
            'mobilenet_v3_small': self._load_mobilenet_v3_small,
            'mobilenet_v4_conv_small': self._load_mobilenet_v4_conv_small,
            'mobilenet_v4_conv_medium': self._load_mobilenet_v4_conv_medium,
            'mobilenet_v4_conv_large': self._load_mobilenet_v4_conv_large,
        }
    
    def load_model(self, model_name: str, pretrained: bool = True) -> nn.Module:
        """
        Load a model by name
        
        Args:
            model_name: Name of the model to load
            pretrained: Whether to load pretrained weights
            
        Returns:
            PyTorch model
        """
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not available. "
                           f"Available models: {list(self.available_models.keys())}")
        
        return self.available_models[model_name](pretrained)
    
    def _load_resnet18(self, pretrained: bool = True) -> nn.Module:
        """Load ResNet-18 model"""
        model = models.resnet18(pretrained=pretrained)
        if self.num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        return model
    
    def _load_resnet50(self, pretrained: bool = True) -> nn.Module:
        """Load ResNet-50 model"""
        model = models.resnet50(pretrained=pretrained)
        if self.num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
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
    
    def get_model_info(self, model: nn.Module) -> Dict[str, Any]:
        """
        Get information about a model
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        }
    
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


def get_available_models() -> list:
    """Get list of available model names"""
    loader = ModelLoader()
    return list(loader.available_models.keys())


def load_model(model_name: str, pretrained: bool = True, num_classes: int = 1000) -> nn.Module:
    """
    Convenience function to load a model
    
    Args:
        model_name: Name of the model to load
        pretrained: Whether to load pretrained weights
        num_classes: Number of output classes
        
    Returns:
        PyTorch model
    """
    loader = ModelLoader(num_classes=num_classes)
    return loader.load_model(model_name, pretrained)