"""
Quantization methods implementation for PyTorch models
"""
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
from typing import Dict, List, Any
import tempfile

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
            print(f"Warning: Device conversion failed: {e}, using CPU inference")
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
        
        # model_device = next(model.parameters()).device
        model_device = self.device
        model = model.to(model_device)  # Ensure model is on the correct device
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(model_device), targets.to(model_device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0) # Accumulate total loss correctly
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0 # Calculate average loss correctly
        
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
        # Default implementation, can be overridden by subclasses if specific handling is needed
        model.eval()
        # model_device = next(model.parameters()).device
        model_device = self.device
        model = model.to(model_device)  # Ensure model is on the correct device
        input_tensor = input_tensor.to(model_device)

        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(input_tensor)
        
        times = []
        with torch.no_grad():
            for _ in range(benchmark_iterations):
                # Ensure synchronization for accurate timing on CUDA devices
                if model_device.type == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.time()
                
                _ = model(input_tensor)
                
                if model_device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        if not times: # Handle case with zero benchmark_iterations
            return {
                'mean_time_ms': 0.0,
                'min_time_ms': 0.0,
                'max_time_ms': 0.0,
                'std_time_ms': 0.0
            }

        mean_time = sum(times) / len(times)
        std_time = (sum([(t - mean_time)**2 for t in times]) / len(times))**0.5 if len(times) > 1 else 0.0
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
        """
        Apply dynamic quantization to the model
        
        Args:
            model: Original model to quantize
            calibration_data: Not used for dynamic quantization
            
        Returns:
            Dynamically quantized model
        """
        self.original_model = copy.deepcopy(model)
        model_copy = copy.deepcopy(model)
        
        # Move model to CPU for dynamic quantization (PyTorch dynamic quantization is CPU-only)
        model_copy = model_copy.to('cpu')
        
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model_copy,
            self.qconfig_spec,
            dtype=torch.qint8
        )
        
        self.quantized_model = quantized_model
        return quantized_model
    
    def evaluate_model(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Evaluate model performance for dynamic quantized models
        
        Args:
            model: Model to evaluate (should be on CPU for dynamic quantization)
            data_loader: Data loader for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        # Ensure model is on CPU for dynamic quantization evaluation
        model = model.to('cpu')
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                # Move data to CPU for dynamic quantized model evaluation
                inputs, targets = inputs.to('cpu'), targets.to('cpu')
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(data_loader)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'correct': correct,
            'total': total
        }
    
    def measure_inference_time(self, model: nn.Module, input_tensor: torch.Tensor, 
                             warmup_iterations: int = 50, benchmark_iterations: int = 500) -> Dict[str, float]:
        """
        Measure model inference time for dynamic quantized models
        
        Args:
            model: Model to benchmark (should be on CPU for dynamic quantization)
            input_tensor: Input tensor for inference
            warmup_iterations: Number of warmup iterations
            benchmark_iterations: Number of benchmark iterations
            
        Returns:
            Dictionary with timing statistics
        """
        model.eval()
        
        # Ensure model and input are on CPU for dynamic quantization
        model = model.to('cpu')
        input_tensor = input_tensor.to('cpu')
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(input_tensor)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(benchmark_iterations):
                start_time = time.time()
                _ = model(input_tensor)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
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
            # 根据设备自动选择后端
            if device and device.type == 'cuda':
                self.backend = 'fbgemm'
            else:
                self.backend = 'qnnpack'
        
        # 设置torch.ao配置
        self._setup_ao_config()
    
    def _setup_ao_config(self):
        """设置torch.ao量化配置"""
        self.qconfig_mapping = get_default_qconfig_mapping(self.backend)
        
    def quantize(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader = None) -> nn.Module:
        """
        Apply static quantization using torch.ao
        """
        self.original_model = copy.deepcopy(model)
        model_copy = copy.deepcopy(model)
        
        try:
            # 使用torch.ao的FX图模式量化
            return self._quantize_with_ao_fx(model_copy, calibration_data)
        except Exception as e:
            print(f"Warning: torch.ao FX quantization failed: {e}")
            print("Falling back to legacy quantization")
            return self._quantize_legacy(model_copy, calibration_data)
    
    def _quantize_with_ao_fx(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader) -> nn.Module:
        """使用torch.ao FX图模式量化"""
        model = model.to('cpu')  # 准备阶段在CPU进行
        model.eval()
        
        # 准备模型进行量化
        example_inputs = self._get_example_inputs(calibration_data)
        prepared_model = prepare_fx(model, self.qconfig_mapping, example_inputs)
        
        # 校准
        self._calibrate_model(prepared_model, calibration_data)
        
        # 转换为量化模型
        quantized_model = convert_fx(prepared_model)
        
        # 处理设备兼容性
        self.quantized_model = self._get_device_compatible_model(quantized_model)
        return self.quantized_model
    
    def _get_example_inputs(self, calibration_data: torch.utils.data.DataLoader) -> tuple:
        """获取示例输入用于torch.ao"""
        for inputs, _ in calibration_data:
            return (inputs[:1].cpu(),)  # 只需要一个批次作为示例
        raise ValueError("Calibration data is empty")
    
    def _calibrate_model(self, prepared_model: nn.Module, calibration_data: torch.utils.data.DataLoader):
        """校准模型"""
        prepared_model.eval()
        with torch.no_grad():
            for i, (inputs, _) in enumerate(calibration_data):
                inputs = inputs.cpu()
                prepared_model(inputs)
                if i >= 100:  # 限制校准样本数量
                    break
    
    def _quantize_legacy(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader) -> nn.Module:
        """传统量化方法作为后备"""
        if calibration_data is None:
            raise ValueError("Calibration data is required for static quantization")
        
        self.original_model = copy.deepcopy(model)
        model_copy = copy.deepcopy(model)
        model_copy.eval()
        
        # Move model to CPU for quantization (PyTorch quantization is CPU-only)
        model_copy = model_copy.to('cpu')
        
        # Set quantization configuration - use per_tensor for compatibility
        if self.backend == 'fbgemm':
            # Use per_tensor quantization for better compatibility
            qconfig = torch.quantization.QConfig(
                activation=torch.quantization.default_observer,
                weight=torch.quantization.default_weight_observer
            )
        else:
            qconfig = torch.quantization.get_default_qconfig(self.backend)
        
        model_copy.qconfig = qconfig
        
        # Prepare model for quantization
        model_prepared = torch.quantization.prepare(model_copy)
        
        # Calibrate with calibration data
        self._calibrate(model_prepared, calibration_data)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model_prepared)
        
        self.quantized_model = quantized_model
        return quantized_model
    
    def _calibrate(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader):
        """Calibrate the model with calibration data"""
        model.eval()
        with torch.no_grad():
            for inputs, _ in calibration_data:
                inputs = inputs.to('cpu')  # Ensure data is on CPU for quantization
                model(inputs)
    
    def evaluate_model(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """评估支持多设备的量化模型"""
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        # 确保criterion在正确的设备上
        criterion = criterion.to(self.device)
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                
                # 确保输出在正确的设备上进行损失计算
                if outputs.device != targets.device:
                    outputs = outputs.to(targets.device)
                
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'correct': correct,
            'total': total
        }

class QATQuantizer(BaseQuantizer):
    """Quantization Aware Training implementation with torch.ao support"""
    
    def __init__(self, device: torch.device = None, backend: str = None, train_epochs: int = 3, learning_rate: float = 1e-4, **kwargs):
        super().__init__(device, **kwargs)
        if backend:
            self.backend = backend
        else:
            if device and device.type == 'cuda':
                self.backend = 'fbgemm'
            else:
                self.backend = 'qnnpack'

        self.train_epochs = train_epochs
        self.learning_rate = learning_rate
        self._setup_ao_qat_config()
    
    def _setup_ao_qat_config(self):
        """设置torch.ao QAT配置"""
        self.qconfig_mapping = get_default_qat_qconfig_mapping(self.backend)
    
    def quantize(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader = None) -> nn.Module:
        """
        Apply QAT using torch.ao
        """
        self.original_model = copy.deepcopy(model)
        model_copy = copy.deepcopy(model)
        
        try:
            return self._quantize_with_ao_qat_fx(model_copy, calibration_data)
        except Exception as e:
            print(f"Warning: torch.ao QAT FX failed: {e}")
            print("Falling back to legacy QAT")
            return self._quantize_legacy_qat(model_copy, calibration_data)
    
    def _quantize_with_ao_qat_fx(self, model: nn.Module, training_data: torch.utils.data.DataLoader) -> nn.Module:
        """使用torch.ao FX图模式QAT"""
        model = model.to(self.device)  # QAT训练可以在GPU上进行
        model.train()
        
        # 准备QAT模型
        example_inputs = self._get_example_inputs(training_data)
        prepared_model = prepare_qat_fx(model, self.qconfig_mapping, example_inputs)
        
        # 进行QAT训练
        trained_model = self._perform_qat_training(prepared_model, training_data)
        
        # 转换为量化模型
        trained_model.eval()
        trained_model = trained_model.cpu()  # 转换阶段在CPU进行
        quantized_model = convert_fx(trained_model)
        
        # 处理设备兼容性
        self.quantized_model = self._get_device_compatible_model(quantized_model)
        return self.quantized_model
    
    def _perform_qat_training(self, model: nn.Module, training_data: torch.utils.data.DataLoader, 
                             epochs: int = 3) -> nn.Module:
        """执行QAT训练"""
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss().to(self.device)
        
        print(f"Starting QAT training for {epochs} epochs on {self.device}")
        
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(training_data):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx >= 100:  # 限制训练批次
                    break
            
            avg_loss = total_loss / min(len(training_data), 100)
            print(f"QAT Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        return model
    
    def _get_example_inputs(self, data_loader: torch.utils.data.DataLoader) -> tuple:
        """获取示例输入"""
        for inputs, _ in data_loader:
            return (inputs[:1].to(self.device),)
        raise ValueError("Training data is empty")
    
    def _quantize_legacy_qat(self, model: nn.Module, training_data: torch.utils.data.DataLoader) -> nn.Module:
        """传统QAT方法作为后备"""
        if training_data is None:
            raise ValueError("Training data is required for QAT")
        
        self.original_model = copy.deepcopy(model)
        model_copy = copy.deepcopy(model)
        model_copy.train()
        
        # 设置QAT配置
        model_copy.qconfig = torch.quantization.get_default_qat_qconfig(self.backend)
        
        # 准备QAT
        torch.quantization.prepare_qat(model_copy, inplace=True)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        # QAT训练
        for epoch in range(self.train_epochs):
            running_loss = 0.0
            for i, data in enumerate(training_data, 0):
                inputs, labels = data
                
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model_copy(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                if i % 20 == 19:    # 每20个mini-batches打印一次
                    print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 20:.3f}")
                    running_loss = 0.0
        
        # 转换为量化模型
        model_copy.eval()
        quantized_model = torch.quantization.convert(model_copy)
        
        self.quantized_model = quantized_model
        return quantized_model


class FXQuantizer(BaseQuantizer):
    """FX Graph Mode quantization implementation"""
    
    def __init__(self, device: torch.device = None, backend: str = 'fbgemm', **kwargs):
        super().__init__(device, **kwargs)
        self.backend = backend
        torch.backends.quantized.engine = backend
    
    def quantize(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader) -> nn.Module:
        """
        Apply FX graph mode quantization to the model
        
        Args:
            model: Original model to quantize
            calibration_data: Calibration data loader
            
        Returns:
            FX quantized model
        """
        if calibration_data is None:
            raise ValueError("Calibration data is required for FX quantization")
        
        self.original_model = copy.deepcopy(model)
        model_copy = copy.deepcopy(model)
        model_copy.eval()
        
        # Create qconfig mapping using recommended approach
        qconfig_mapping = get_default_qconfig_mapping(self.backend)
        
        # Get example input
        example_inputs = next(iter(calibration_data))[0][:1].to(self.device)
        
        # Prepare model for quantization
        model_prepared = prepare_fx(model_copy, qconfig_mapping, example_inputs)
        
        # Calibrate with calibration data
        self._calibrate(model_prepared, calibration_data)
        
        # Convert to quantized model
        quantized_model = convert_fx(model_prepared)
        
        self.quantized_model = quantized_model
        return quantized_model
    
    def _calibrate(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader):
        """Calibrate the model with calibration data"""
        model.eval()
        with torch.no_grad():
            for inputs, _ in calibration_data:
                inputs = inputs.to(self.device)
                model(inputs)


class INT8Quantizer(BaseQuantizer):
    """INT8 quantization using custom observers"""
    
    def __init__(self, device: torch.device = None, **kwargs):
        super().__init__(device, **kwargs)
    
    def quantize(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader) -> nn.Module:
        """
        Apply INT8 quantization with custom observers
        
        Args:
            model: Original model to quantize
            calibration_data: Calibration data loader
            
        Returns:
            INT8 quantized model
        """
        if calibration_data is None:
            raise ValueError("Calibration data is required for INT8 quantization")
        
        self.original_model = copy.deepcopy(model)
        model_copy = copy.deepcopy(model)
        model_copy.eval()
        
        # Custom QConfig with INT8 observers
        custom_qconfig = QConfig(
            activation=default_observer.with_args(dtype=torch.qint8),
            weight=default_weight_observer.with_args(dtype=torch.qint8)
        )
        
        model_copy.qconfig = custom_qconfig
        
        # Prepare model for quantization
        model_prepared = torch.quantization.prepare(model_copy)
        
        # Calibrate with calibration data
        self._calibrate(model_prepared, calibration_data)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model_prepared)
        
        self.quantized_model = quantized_model
        return quantized_model
    
    def _calibrate(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader):
        """Calibrate the model with calibration data"""
        model.eval()
        with torch.no_grad():
            for inputs, _ in calibration_data:
                inputs = inputs.to(self.device)
                model(inputs)


class QuantizationBenchmark(BaseQuantizer):
    """Benchmark multiple quantization methods"""
    
    def __init__(self, device: torch.device = None, backend: str = 'fbgemm', **kwargs):
        super().__init__(device, **kwargs)
        self.device = device or torch.device('cpu')
        self.backend = backend
        self.results = {}
        self.model_name = kwargs.get('model_name', 'unknown')
    
    def _load_original_resnet_for_lrf(self, lrf_model: nn.Module) -> nn.Module:
        """
        Load corresponding original ResNet model for LRF baseline comparison
        
        Args:
            lrf_model: LRF ResNet model
            
        Returns:
            Original ResNet model corresponding to the LRF model
        """
        # Import here to avoid circular imports
        import sys
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        from models.model_loader import ModelLoader
        
        # Determine the corresponding original ResNet model
        if 'resnet18' in self.model_name.lower():
            original_model_name = 'resnet18_quantizable'
        elif 'resnet50' in self.model_name.lower():
            original_model_name = 'resnet50_quantizable'
        else:
            print(f"Warning: Could not determine original ResNet for {self.model_name}, using LRF model as baseline")
            return lrf_model
        
        # Get number of classes from LRF model
        if hasattr(lrf_model, 'fc') and hasattr(lrf_model.fc, 'out_features'):
            num_classes = lrf_model.fc.out_features
        else:
            num_classes = 1000  # Default ImageNet classes
        
        # Load the original ResNet model
        try:
            model_loader = ModelLoader(num_classes=num_classes, device=self.device, enable_finetuning=False)
            original_model = model_loader.load_model(original_model_name, pretrained=True)
            original_model = original_model.to(self.device)
            original_model.eval()
            
            print(f"Successfully loaded {original_model_name} as baseline for LRF comparison")
            return original_model
            
        except Exception as e:
            print(f"Error loading original ResNet {original_model_name}: {e}")
            print("Using LRF model as baseline instead")
            return lrf_model

    def benchmark_quantization_methods(self, model: nn.Module, 
                                     calibration_data: torch.utils.data.DataLoader,
                                     evaluation_data: torch.utils.data.DataLoader,
                                     methods: list = None) -> Dict[str, Dict[str, Any]]:
        """
        Benchmark multiple quantization methods
        
        Args:
            model: Original model to quantize
            calibration_data: Data for calibration (used for static quantization)
            evaluation_data: Data for evaluation
            methods: List of quantization methods to benchmark
            
        Returns:
            Dictionary with results for each method
        """
        if methods is None:
            methods = ['dynamic', 'static', 'qat']
        
        self.results = {}
        
        # Get example input for timing measurements
        example_input = None
        try:
            for inputs, _ in calibration_data:
                example_input = inputs[:1]  # Take first sample
                break
        except Exception as e:
            print(f"Warning: Could not get example input: {e}")
        
        # Evaluate original model first
        print("Evaluating original (unquantized) model...")
        
        # For LRF models, use the original ResNet as baseline instead of LRF model
        original_model = model
        lrf_model = None
        is_lrf_model = 'low_rank' in self.model_name
        
        if is_lrf_model:
            print(f"Detected LRF model: {self.model_name}")
            print("Loading corresponding original ResNet as baseline...")
            original_model = self._load_original_resnet_for_lrf(model)
            lrf_model = model  # Keep reference to LRF model
        
        # Evaluate original ResNet baseline
        original_eval_metrics = self.evaluate_model(original_model, evaluation_data)
        
        if example_input is not None:
            # Move example input to the same device as the model
            model_device = next(original_model.parameters()).device
            original_time_stats = self.measure_inference_time(original_model, example_input.to(model_device))
        else:
            original_time_stats = {k: 0 for k in ['mean_time_ms', 'min_time_ms', 'max_time_ms', 'std_time_ms']}

        original_size_mb = self._get_model_size(original_model)
        
        self.results['original'] = {
            'accuracy': original_eval_metrics.get('accuracy', 0.0),
            'loss': original_eval_metrics.get('loss', 0.0),
            'model_size_mb': original_size_mb,
            'compression_ratio': 1.0,
            'speedup': 1.0,
            **original_time_stats
        }
        print(f"Original Model - Accuracy: {self.results['original']['accuracy']:.2f}%, "
              f"Size: {original_size_mb:.2f}MB, "
              f"Inference Time: {original_time_stats.get('mean_time_ms', 0.0):.2f}ms")

        # For LRF models, also evaluate the LRF model itself (unquantized)
        if is_lrf_model and lrf_model is not None:
            print("\nEvaluating LRF (unquantized) model...")
            lrf_eval_metrics = self.evaluate_model(lrf_model, evaluation_data)
            
            if example_input is not None:
                lrf_model_device = next(lrf_model.parameters()).device
                lrf_time_stats = self.measure_inference_time(lrf_model, example_input.to(lrf_model_device))
            else:
                lrf_time_stats = {k: 0 for k in ['mean_time_ms', 'min_time_ms', 'max_time_ms', 'std_time_ms']}

            lrf_size_mb = self._get_model_size(lrf_model)
            
            # Calculate LRF metrics relative to original
            lrf_speedup = original_time_stats.get('mean_time_ms', 0.0) / lrf_time_stats.get('mean_time_ms', 1.0) if lrf_time_stats.get('mean_time_ms', 0.0) > 0 else 0.0
            lrf_compression = original_size_mb / lrf_size_mb if lrf_size_mb > 0 else 0.0
            
            self.results['lrf'] = {
                'accuracy': lrf_eval_metrics.get('accuracy', 0.0),
                'loss': lrf_eval_metrics.get('loss', 0.0),
                'model_size_mb': lrf_size_mb,
                'compression_ratio': lrf_compression,
                'speedup': lrf_speedup,
                **lrf_time_stats
            }
            print(f"LRF Model - Accuracy: {self.results['lrf']['accuracy']:.2f}%, "
                  f"Size: {lrf_size_mb:.2f}MB, "
                  f"Inference Time: {lrf_time_stats.get('mean_time_ms', 0.0):.2f}ms, "
                  f"Speedup: {lrf_speedup:.2f}x, "
                  f"Compression: {lrf_compression:.2f}x")

        # Get quantizers using the stored backend
        all_quantizers = get_available_quantizers(backend=self.backend, device=self.device, model_name=self.model_name)
        
        # Use the actual model (LRF model) for quantization, not the original ResNet
        model_to_quantize = lrf_model if is_lrf_model else model
        
        for method_name in methods:
            quantizer = all_quantizers[method_name]
            
            # For LRF models, add 'lrf_' prefix to method names for clarity
            result_key = f"lrf_{method_name}" if is_lrf_model else method_name
            
            print(f"\nBenchmarking {result_key} quantization...")
            
            try:
                # Quantize model
                quantized_model = quantizer.quantize(copy.deepcopy(model_to_quantize), calibration_data)
                
                # Check if the quantized model has parameters
                params = list(quantized_model.parameters())
                if not params or not any(p.numel() > 0 for p in params): # Check if parameters exist and are not all empty
                    print(f"Warning: Quantized model for {method_name} has no parameters or only empty parameters after conversion.")
                    # Fallback to evaluating the model as is, assuming it might be a graph or a different structure
                    # that doesn't rely on standard nn.Module parameters in the same way.
                    # That means we don't have a meaningful way to measure speedup or compression ratio.
                    eval_metrics = quantizer.evaluate_model(quantized_model, evaluation_data)
                    
                    if example_input is not None:
                        # Ensure the input tensor is on the correct device for the quantized_model
                        # This is a bit tricky if the model has no parameters to infer device from.
                        # We'll try to use the quantizer's device or fall back to original model's device.
                        try:
                            # Attempt to get device from model if it has a device attribute (e.g. for FX graphs)
                            q_device = quantized_model.device if hasattr(quantized_model, 'device') else quantizer.device
                        except AttributeError:
                            q_device = quantizer.device # Fallback to quantizer's device
                        
                        time_stats = quantizer.measure_inference_time(quantized_model, example_input.to(q_device))
                    else:
                        time_stats = {k: 0 for k in ['mean_time_ms', 'min_time_ms', 'max_time_ms', 'std_time_ms']}
                    
                    model_size_mb = self._get_model_size(quantized_model) # This might return 0 if it relies on parameters

                    self.results[result_key] = {
                        'accuracy': eval_metrics.get('accuracy', 0.0),
                        'loss': eval_metrics.get('loss', 0.0),
                        'model_size_mb': model_size_mb,
                        # 'warning': f'Quantized model for {method_name} has no standard parameters or only empty ones.',
                        'mean_time_ms': time_stats.get('mean_time_ms', 0.0),
                        'min_time_ms': time_stats.get('min_time_ms', 0.0),
                        'max_time_ms': time_stats.get('max_time_ms', 0.0),
                        'std_time_ms': time_stats.get('std_time_ms', 0.0),
                        'speedup': 0.0, 
                        'compression_ratio': 0.0
                    }
                    print(f"Type of problematic quantized_model: {type(quantized_model)}")
                    print(f"{result_key.capitalize()} Quantized Model (no params) - Accuracy: {self.results[result_key]['accuracy']:.2f}%, "
                          f"Size: {model_size_mb:.2f}MB, "
                          f"Inference Time: {time_stats.get('mean_time_ms', 0.0):.2f}ms")
                    continue

                quantized_model_device = params[0].device
                print(f"Evaluating {result_key} quantized model on device: {quantized_model_device}")
                eval_metrics = quantizer.evaluate_model(quantized_model, evaluation_data)
                
                if example_input is not None:
                    time_stats = quantizer.measure_inference_time(quantized_model, example_input.to(quantized_model_device))
                else:
                    time_stats = {k: 0 for k in ['mean_time_ms', 'min_time_ms', 'max_time_ms', 'std_time_ms']}

                model_size_mb = self._get_model_size(quantized_model)
                
                # Calculate speedup and compression
                original_mean_time = self.results.get('original', {}).get('mean_time_ms', 0.0)
                current_mean_time = time_stats.get('mean_time_ms', float('inf')) # Avoid division by zero if current_mean_time is 0
                speedup_ratio = original_mean_time / current_mean_time if current_mean_time > 0 and original_mean_time > 0 else 0.0

                original_size = self.results.get('original', {}).get('model_size_mb', 0.0)
                compression_ratio_val = original_size / model_size_mb if model_size_mb > 0 and original_size > 0 else 0.0
                
                self.results[result_key] = {
                    'accuracy': eval_metrics.get('accuracy', 0.0),
                    'loss': eval_metrics.get('loss', 0.0),
                    'model_size_mb': model_size_mb,
                    'compression_ratio': compression_ratio_val,
                    'speedup': speedup_ratio,
                    **time_stats
                }
                print(f"{result_key.capitalize()} Quantized Model - Accuracy: {self.results[result_key]['accuracy']:.2f}%, "
                      f"Size: {model_size_mb:.2f}MB, "
                      f"Inference Time: {time_stats.get('mean_time_ms', 0.0):.2f}ms, "
                      f"Speedup: {speedup_ratio:.2f}x, "
                      f"Compression: {compression_ratio_val:.2f}x")

            except Exception as e:
                print(f"Error benchmarking {method_name}: {e}")
                traceback.print_exc()
                self.results[result_key] = {
                    'accuracy': 0.0, 'loss': 0.0, 'model_size_mb': 0.0, 
                    'error': str(e),
                    'mean_time_ms': 0.0, 'min_time_ms': 0.0, 'max_time_ms': 0.0, 'std_time_ms': 0.0, # Ensure keys exist
                    'speedup': 0.0, 'compression_ratio': 0.0
                }
        
        return self.results
    
    
    def _get_model_size(self, model: nn.Module) -> float:
        """
        Get model size in MB
        
        Args:
            model: PyTorch model
            
        Returns:
            Model size in MB
        """
        import tempfile
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            torch.save(model.state_dict(), temp_file.name)
            size_mb = temp_file.seek(0, 2) / (1024 * 1024)  # Get file size in MB
        return size_mb
    
    def print_results(self):
        """Print benchmark results in a formatted table"""
        if not self.results:
            print("No results to display. Run benchmark_quantization_methods first.")
            return
        
        print("\n" + "="*100) # Increased width for new columns
        print("QUANTIZATION BENCHMARK RESULTS")
        print("="*100)
        print(f"{'Method':<12} {'Accuracy':<10} {'Loss':<8} {'Time(ms)':<10} {'Size(MB)':<10} {'Speedup':<8} {'Compression':<12}")
        print("-"*100)

        original_speed = 0.0
        original_comp = 0.0

        for method, res_data in self.results.items(): # Renamed results to res_data to avoid conflict
            if 'error' in res_data and res_data['error']: # Check if error is not None or empty
                print(f"{method:<12} ERROR: {res_data['error']}")
                continue
                
            accuracy = f"{res_data.get('accuracy', 0.0):.2f}%"
            loss = f"{res_data.get('loss', 0.0):.4f}"
            time_ms = f"{res_data.get('mean_time_ms', 0.0):.2f}" # Use mean_time_ms
            size_mb = f"{res_data.get('model_size_mb', 0.0):.2f}"

            if method == 'original':
                speedup = "1.00x"
                original_speed = res_data.get('mean_time_ms', 0.0)
                compression = "1.00x"
                original_comp = res_data.get('model_size_mb', 0.0)
            else:
                # speedup = f"{res_data.get('speedup', 0.0):.2f}x"
                # compression = f"{res_data.get('compression_ratio', 0.0):.2f}x"
                speedup = f"{original_speed / res_data.get('mean_time_ms', 0.0):.2f}x" if res_data.get('mean_time_ms', 0.0) > 0 else f"{res_data.get('speedup', 0.0):.2f}x"
                compression = f"{original_comp / res_data.get('model_size_mb', 0.0):.2f}x" if res_data.get('model_size_mb', 0.0) > 0 else f"{res_data.get('compression_ratio', 0.0):.2f}x"
            
            print(f"{method:<12} {accuracy:<10} {loss:<8} {time_ms:<10} {size_mb:<10} {speedup:<8} {compression:<12}")
        print("="*100)

class OfficialQuantizedQuantizer(BaseQuantizer):
    """Official pre-trained quantized models from torchvision.models.quantization"""
    
    def __init__(self, device: torch.device = None, **kwargs):
        super().__init__(device, **kwargs)
        # Official quantized models are designed for CPU inference
        self.device = torch.device('cpu')
    
    def quantize(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader = None) -> nn.Module:
        """
        Load official pre-trained quantized model as reference
        
        Args:
            model: Original model (used to determine which official quantized model to load)
            calibration_data: Not used for this method (official models are already quantized)
            
        Returns:
            Official pre-trained quantized model
        """
        self.original_model = copy.deepcopy(model)
        
        # Determine model type and load corresponding official quantized model
        model_name = self._get_model_name(model)
        quantized_model = self._load_official_quantized_model(model_name)
        
        if quantized_model is None:
            raise ValueError(f"No official quantized model available for {model_name}")
        
        # Move to CPU (official quantized models are designed for CPU)
        quantized_model = quantized_model.to('cpu')
        quantized_model.eval()
        
        self.quantized_model = quantized_model
        return quantized_model
    
    def _get_model_name(self, model: nn.Module) -> str:
        """Extract model name from model instance"""
        return self.model_name
    
    def _load_official_quantized_model(self, model_name: str) -> nn.Module:
        """Load official pre-trained quantized model"""
        try:
            if model_name.startswith('resnet18'):
                from torchvision.models.quantization import resnet18
                return resnet18(pretrained=True, quantize=True, backend=self.backend)
            elif model_name.startswith('resnet50'):
                from torchvision.models.quantization import resnet50
                return resnet50(pretrained=True, quantize=True, backend=self.backend)
            elif model_name.startswith('mobilenet_v2'):
                from torchvision.models.quantization import mobilenet_v2
                return mobilenet_v2(pretrained=True, quantize=True, backend=self.backend)
            elif model_name.startswith('mobilenet_v3_large'):
                from torchvision.models.quantization import mobilenet_v3_large
                return mobilenet_v3_large(pretrained=True, quantize=True, backend=self.backend)
            elif model_name.startswith('mobilenet_v3_small'):
                # Note: mobilenet_v3_small quantized version might not be available
                # Fall back to mobilenet_v3_large
                try:
                    from torchvision.models.quantization import mobilenet_v3_small
                    return mobilenet_v3_small(pretrained=True, quantize=True)
                except (ImportError, AttributeError):
                    print("Warning: mobilenet_v3_small quantized version not available, using mobilenet_v3_large")
                    from torchvision.models.quantization import mobilenet_v3_large
                    return mobilenet_v3_large(pretrained=True, quantize=True)
            else:
                print(f"Warning: No official quantized model available for {model_name}")
                return None
                
        except ImportError as e:
            print(f"Error importing official quantized model for {model_name}: {e}")
            return None
        except Exception as e:
            print(f"Error loading official quantized model for {model_name}: {e}")
            return None

    def evaluate_model(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Evaluate the official quantized model
        Note: Official quantized models run on CPU
        """
        model.eval()
        model = model.to('cpu')  # Ensure model is on CPU
        
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to('cpu'), targets.to('cpu')  # Ensure data is on CPU
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total if total > 0 else 0.0
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
        Measure inference time for official quantized model
        Note: Official quantized models run on CPU
        """
        model.eval()
        model = model.to('cpu')
        input_tensor = input_tensor.to('cpu')
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(input_tensor)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(benchmark_iterations):
                start_time = time.time()
                _ = model(input_tensor)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        return {
            'mean_time_ms': sum(times) / len(times),
            'min_time_ms': min(times),
            'max_time_ms': max(times),
            'std_time_ms': (sum([(t - sum(times)/len(times))**2 for t in times]) / len(times))**0.5
        }

def get_available_quantizers(backend: str = 'qnnpack', device: torch.device = None, model_name: str = None) -> Dict[str, BaseQuantizer]:
    """Get dictionary of available quantizers"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    qat_train_epochs = 10
    qat_learning_rate = 1e-5

    return {
        'dynamic': DynamicQuantizer(device, backend=backend, model_name=model_name),
        'static': StaticQuantizer(device, backend=backend, model_name=model_name),
        'qat': QATQuantizer(device, backend=backend, train_epochs=qat_train_epochs,
                            learning_rate=qat_learning_rate, model_name=model_name),
        'fx': FXQuantizer(device, backend=backend, model_name=model_name),
        'int8': INT8Quantizer(device, backend=backend, model_name=model_name),
        'official': OfficialQuantizedQuantizer(device, backend=backend, model_name=model_name),
    }