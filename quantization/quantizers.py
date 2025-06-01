"""
Quantization methods implementation for PyTorch models
"""
import torch
import torch.nn as nn
import torch.utils.data
import torch.quantization as quant
from torch.quantization import QConfig, default_observer, default_weight_observer
try:
    # Try new API first
    from torch.ao.quantization import QConfigMapping, get_default_qconfig
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
    NEW_API = True
except ImportError:
    try:
        # Fallback to old API
        from torch.quantization.quantize_fx import prepare_fx, convert_fx
        from torch.quantization import get_default_qconfig
        NEW_API = False
    except ImportError:
        # FX quantization not available
        prepare_fx = None
        convert_fx = None
        get_default_qconfig = None
        NEW_API = False

# Import torchvision quantizable models
try:
    from torchvision.models.quantization import resnet18 as quantizable_resnet18
    from torchvision.models.quantization import resnet50 as quantizable_resnet50
    from torchvision.models.quantization import ResNet18_QuantizedWeights, ResNet50_QuantizedWeights
    QUANTIZABLE_MODELS_AVAILABLE = True
except ImportError:
    quantizable_resnet18 = None
    quantizable_resnet50 = None
    ResNet18_QuantizedWeights = None
    ResNet50_QuantizedWeights = None
    QUANTIZABLE_MODELS_AVAILABLE = False

import copy
from typing import Dict, Any, Callable, Optional, Tuple
import time
import os


class BaseQuantizer:
    """Base class for all quantization methods"""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.quantized_model = None
        self.original_model = None
    
    def quantize(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader = None) -> nn.Module:
        """
        Quantize the model
        
        Args:
            model: Original model to quantize
            calibration_data: Calibration data loader for static quantization
            
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
        # Determine the device the model is on
        model_device = next(model.parameters()).device
        return self._evaluate_model_on_device(model, data_loader, model_device)
    
    def evaluate_model_on_optimal_device(self, model: nn.Module, data_loader: torch.utils.data.DataLoader, 
                                       quantization_method: str = None) -> Dict[str, float]:
        """
        Evaluate model performance on the optimal device based on quantization method
        
        Args:
            model: Model to evaluate
            data_loader: Data loader for evaluation
            quantization_method: The quantization method used (if known)
            
        Returns:
            Dictionary with evaluation metrics
        """
        # For dynamic quantization, prefer CPU; for others, try current device
        if quantization_method == 'dynamic':
            cpu_device = torch.device('cpu')
            model_cpu = model.to(cpu_device)
            # Create CPU dataloader
            cpu_data_loader = self._create_cpu_dataloader(data_loader)
            return self._evaluate_model_on_device(model_cpu, cpu_data_loader, cpu_device)
        else:
            # Use the device the model is currently on
            model_device = next(model.parameters()).device
            return self._evaluate_model_on_device(model, data_loader, model_device)
    
    def _create_cpu_dataloader(self, dataloader):
        """Create a CPU version of the dataloader"""
        class CPUDataLoader:
            def __init__(self, original_loader):
                self.original_loader = original_loader
                self.dataset = original_loader.dataset
                
            def __iter__(self):
                for batch in self.original_loader:
                    inputs, targets = batch
                    yield inputs.to('cpu'), targets.to('cpu')
                    
            def __len__(self):
                return len(self.original_loader)
        
        return CPUDataLoader(dataloader)
    
    def measure_inference_time(self, model: nn.Module, input_tensor: torch.Tensor, 
                             warmup_iterations: int = 10, benchmark_iterations: int = 100) -> Dict[str, float]:
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
        # Determine the device the model is on
        model_device = next(model.parameters()).device
        return self._measure_inference_time_on_device(model, input_tensor, model_device, warmup_iterations, benchmark_iterations)
    
    def _evaluate_model_on_device(self, model: nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device) -> Dict[str, float]:
        """
        Evaluate model performance on a specific device
        
        Args:
            model: Model to evaluate
            data_loader: Data loader for evaluation
            device: Device to run evaluation on
            
        Returns:
            Dictionary with evaluation metrics
        """
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
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
    
    def _measure_inference_time_on_device(self, model: nn.Module, input_tensor: torch.Tensor, device: torch.device,
                                        warmup_iterations: int = 10, benchmark_iterations: int = 100) -> Dict[str, float]:
        """
        Measure model inference time on a specific device
        
        Args:
            model: Model to benchmark
            input_tensor: Input tensor for inference
            device: Device to run inference on
            warmup_iterations: Number of warmup iterations
            benchmark_iterations: Number of benchmark iterations
            
        Returns:
            Dictionary with timing statistics
        """
        model.eval()
        input_tensor = input_tensor.to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(input_tensor)
        
        # Synchronize CUDA if using GPU
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(benchmark_iterations):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.time()
                _ = model(input_tensor)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        return {
            'mean_time_ms': sum(times) / len(times),
            'min_time_ms': min(times),
            'max_time_ms': max(times),
            'std_time_ms': (sum([(t - sum(times)/len(times))**2 for t in times]) / len(times))**0.5
        }


class DynamicQuantizer(BaseQuantizer):
    """Dynamic quantization implementation - optimized for CPU evaluation"""
    
    def __init__(self, device: torch.device = None, qconfig_spec: Dict = None):
        super().__init__(device)
        self.qconfig_spec = qconfig_spec or {nn.Linear, nn.Conv2d}
        # Dynamic quantization models are evaluated on CPU for optimal compatibility
        self.eval_device = torch.device('cpu')
    
    def quantize(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader = None) -> nn.Module:
        """
        Apply dynamic quantization to the model
        
        Args:
            model: Original model to quantize
            calibration_data: Not used for dynamic quantization
            
        Returns:
            Dynamically quantized model (configured for CPU evaluation)
        """
        self.original_model = copy.deepcopy(model)
        model_copy = copy.deepcopy(model)
        
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model_copy,
            self.qconfig_spec,
            dtype=torch.qint8
        )
        
        # Keep on CPU for optimal performance (user requirement)
        quantized_model = quantized_model.to(self.eval_device)
        
        self.quantized_model = quantized_model
        return quantized_model
    
    def evaluate_model(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Evaluate dynamic quantized model on CPU (optimized setup)
        """
        # Ensure model is on CPU
        model_cpu = model.to(self.eval_device)
        # Create CPU dataloader for evaluation
        cpu_data_loader = self._create_cpu_dataloader(data_loader)
        return self._evaluate_model_on_device(model_cpu, cpu_data_loader, self.eval_device)
    
    def measure_inference_time(self, model: nn.Module, input_tensor: torch.Tensor, 
                             warmup_iterations: int = 10, benchmark_iterations: int = 100) -> Dict[str, float]:
        """
        Measure inference time for dynamic quantized model on CPU
        """
        # Ensure model and input are on CPU
        model_cpu = model.to(self.eval_device)
        input_cpu = input_tensor.to(self.eval_device)
        return self._measure_inference_time_on_device(model_cpu, input_cpu, self.eval_device, 
                                                    warmup_iterations, benchmark_iterations)


class StaticQuantizer(BaseQuantizer):
    """Static quantization implementation - optimized for GPU when possible"""
    
    def __init__(self, device: torch.device = None, backend: str = 'fbgemm'):
        super().__init__(device)
        self.backend = backend
        torch.backends.quantized.engine = backend
        # Prefer faster device for evaluation when possible
        self.preferred_eval_device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def evaluate_model(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Evaluate static quantized model, trying GPU first for better performance
        """
        # Try GPU evaluation first if available
        if self.preferred_eval_device.type == 'cuda':
            try:
                model_gpu = model.to(self.preferred_eval_device)
                # Test with a small batch
                test_input = next(iter(data_loader))[0][:1].to(self.preferred_eval_device)
                with torch.no_grad():
                    _ = model_gpu(test_input)
                # If successful, use GPU
                return self._evaluate_model_on_device(model_gpu, data_loader, self.preferred_eval_device)
            except Exception:
                # Fall back to CPU if GPU fails
                pass
        
        # Use CPU evaluation (fallback or default)
        model_cpu = model.to(torch.device('cpu'))
        cpu_data_loader = self._create_cpu_dataloader(data_loader)
        return self._evaluate_model_on_device(model_cpu, cpu_data_loader, torch.device('cpu'))
    
    def measure_inference_time(self, model: nn.Module, input_tensor: torch.Tensor, 
                             warmup_iterations: int = 10, benchmark_iterations: int = 100) -> Dict[str, float]:
        """
        Measure inference time, trying GPU first for better performance
        """
        # Try GPU timing first if available
        if self.preferred_eval_device.type == 'cuda':
            try:
                model_gpu = model.to(self.preferred_eval_device)
                input_gpu = input_tensor.to(self.preferred_eval_device)
                # Test inference
                with torch.no_grad():
                    _ = model_gpu(input_gpu)
                return self._measure_inference_time_on_device(model_gpu, input_gpu, self.preferred_eval_device,
                                                            warmup_iterations, benchmark_iterations)
            except Exception:
                # Fall back to CPU if GPU fails
                pass
        
        # Use CPU timing (fallback or default)
        model_cpu = model.to(torch.device('cpu'))
        input_cpu = input_tensor.to(torch.device('cpu'))
        return self._measure_inference_time_on_device(model_cpu, input_cpu, torch.device('cpu'),
                                                    warmup_iterations, benchmark_iterations)
    
    def quantize(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader) -> nn.Module:
        """
        Apply static quantization to the model
        
        Args:
            model: Original model to quantize
            calibration_data: Calibration data loader
            
        Returns:
            Statically quantized model
        """
        if calibration_data is None:
            raise ValueError("Calibration data is required for static quantization")
        
        self.original_model = copy.deepcopy(model)
        model_copy = copy.deepcopy(model)
        model_copy.eval()
        
        print("Attempting static quantization...")
        
        try:
            # Simple static quantization using eager mode (more stable)
            print("Using eager mode static quantization...")
            
            # Add quant/dequant stubs for static quantization
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
            
            wrapped_model = QuantizedModel(model_copy)
            
            # Set quantization configuration
            qconfig = torch.quantization.get_default_qconfig(self.backend)
            wrapped_model.qconfig = qconfig
            
            # Prepare model for quantization
            print("Preparing model for quantization...")
            model_prepared = torch.quantization.prepare(wrapped_model)
            
            # Calibrate with calibration data (with safety measures)
            print("Calibrating model...")
            self._calibrate_safely(model_prepared, calibration_data)
            
            # Convert to quantized model
            print("Converting to quantized model...")
            quantized_model = torch.quantization.convert(model_prepared)
            
            print("Static quantization completed successfully!")
            
        except Exception as static_error:
            print(f"Static quantization failed: {static_error}")
            print("WARNING: Static quantization failed. Returning original model (no quantization applied).")
            quantized_model = model_copy
            # Mark this as a quantization failure but still return a working model
            quantized_model._quantization_failed = True
        
        self.quantized_model = quantized_model
        return quantized_model
    
    def _calibrate_safely(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader):
        """Calibrate the model with calibration data with error handling"""
        model.eval()
        print("Starting calibration...")
        
        with torch.no_grad():
            for i, (inputs, _) in enumerate(calibration_data):
                try:
                    inputs = inputs.to(self.device)
                    _ = model(inputs)
                    
                    # Progress indicator
                    if i % 10 == 0:
                        print(f"Calibration progress: {i}/{len(calibration_data)} batches")
                        
                except Exception as batch_error:
                    print(f"Error in calibration batch {i}: {batch_error}")
                    # Continue with next batch instead of failing completely
                    continue
                    
        print("Calibration completed.")
    
    def _calibrate(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader):
        """Calibrate the model with calibration data"""
        model.eval()
        with torch.no_grad():
            for inputs, _ in calibration_data:
                try:
                    inputs = inputs.to(self.device)
                    model(inputs)
                except Exception as batch_error:
                    # Skip problematic batches instead of crashing
                    print(f"Warning: Skipping calibration batch due to error: {batch_error}")
                    continue


class QATQuantizer(BaseQuantizer):
    """Quantization Aware Training implementation - optimized for GPU when possible"""
    
    def __init__(self, device: torch.device = None, backend: str = 'fbgemm'):
        super().__init__(device)
        self.backend = backend
        torch.backends.quantized.engine = backend
        # Prefer faster device for evaluation when possible
        self.preferred_eval_device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def evaluate_model(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Evaluate QAT model, trying GPU first for better performance
        """
        # Try GPU evaluation first if available
        if self.preferred_eval_device.type == 'cuda':
            try:
                model_gpu = model.to(self.preferred_eval_device)
                # Test with a small batch
                test_input = next(iter(data_loader))[0][:1].to(self.preferred_eval_device)
                with torch.no_grad():
                    _ = model_gpu(test_input)
                # If successful, use GPU
                return self._evaluate_model_on_device(model_gpu, data_loader, self.preferred_eval_device)
            except Exception:
                # Fall back to CPU if GPU fails
                pass
        
        # Use CPU evaluation (fallback or default)
        model_cpu = model.to(torch.device('cpu'))
        cpu_data_loader = self._create_cpu_dataloader(data_loader)
        return self._evaluate_model_on_device(model_cpu, cpu_data_loader, torch.device('cpu'))
    
    def measure_inference_time(self, model: nn.Module, input_tensor: torch.Tensor, 
                             warmup_iterations: int = 10, benchmark_iterations: int = 100) -> Dict[str, float]:
        """
        Measure inference time, trying GPU first for better performance
        """
        # Try GPU timing first if available
        if self.preferred_eval_device.type == 'cuda':
            try:
                model_gpu = model.to(self.preferred_eval_device)
                input_gpu = input_tensor.to(self.preferred_eval_device)
                # Test inference
                with torch.no_grad():
                    _ = model_gpu(input_gpu)
                return self._measure_inference_time_on_device(model_gpu, input_gpu, self.preferred_eval_device,
                                                            warmup_iterations, benchmark_iterations)
            except Exception:
                # Fall back to CPU if GPU fails
                pass
        
        # Use CPU timing (fallback or default)
        model_cpu = model.to(torch.device('cpu'))
        input_cpu = input_tensor.to(torch.device('cpu'))
        return self._measure_inference_time_on_device(model_cpu, input_cpu, torch.device('cpu'),
                                                    warmup_iterations, benchmark_iterations)
    
    def quantize(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader = None) -> nn.Module:
        """
        Prepare model for QAT (does not include actual training)
        
        Args:
            model: Original model to quantize
            calibration_data: Not used for QAT preparation
            
        Returns:
            Model prepared for QAT
        """
        self.original_model = copy.deepcopy(model)
        model_copy = copy.deepcopy(model)
        
        # Set QAT quantization configuration
        model_copy.qconfig = torch.quantization.get_default_qat_qconfig(self.backend)
        
        # Prepare model for QAT
        model_prepared = torch.quantization.prepare_qat(model_copy)
        
        self.quantized_model = model_prepared
        return model_prepared
    
    def convert_qat_model(self, qat_model: nn.Module) -> nn.Module:
        """
        Convert QAT model to quantized model
        
        Args:
            qat_model: QAT trained model
            
        Returns:
            Quantized model
        """
        qat_model.eval()
        quantized_model = torch.quantization.convert(qat_model)
        return quantized_model


class FXQuantizer(BaseQuantizer):
    """FX Graph Mode quantization implementation - optimized for GPU when possible"""
    
    def __init__(self, device: torch.device = None, backend: str = 'fbgemm'):
        super().__init__(device)
        self.backend = backend
        torch.backends.quantized.engine = backend
        # Prefer faster device for evaluation when possible
        self.preferred_eval_device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def evaluate_model(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Evaluate FX quantized model, trying GPU first for better performance
        """
        # Try GPU evaluation first if available
        if self.preferred_eval_device.type == 'cuda':
            try:
                model_gpu = model.to(self.preferred_eval_device)
                # Test with a small batch
                test_input = next(iter(data_loader))[0][:1].to(self.preferred_eval_device)
                with torch.no_grad():
                    _ = model_gpu(test_input)
                # If successful, use GPU
                return self._evaluate_model_on_device(model_gpu, data_loader, self.preferred_eval_device)
            except Exception:
                # Fall back to CPU if GPU fails
                pass
        
        # Use CPU evaluation (fallback or default)
        model_cpu = model.to(torch.device('cpu'))
        cpu_data_loader = self._create_cpu_dataloader(data_loader)
        return self._evaluate_model_on_device(model_cpu, cpu_data_loader, torch.device('cpu'))
    
    def measure_inference_time(self, model: nn.Module, input_tensor: torch.Tensor, 
                             warmup_iterations: int = 10, benchmark_iterations: int = 100) -> Dict[str, float]:
        """
        Measure inference time, trying GPU first for better performance
        """
        # Try GPU timing first if available
        if self.preferred_eval_device.type == 'cuda':
            try:
                model_gpu = model.to(self.preferred_eval_device)
                input_gpu = input_tensor.to(self.preferred_eval_device)
                # Test inference
                with torch.no_grad():
                    _ = model_gpu(input_gpu)
                return self._measure_inference_time_on_device(model_gpu, input_gpu, self.preferred_eval_device,
                                                            warmup_iterations, benchmark_iterations)
            except Exception:
                # Fall back to CPU if GPU fails
                pass
        
        # Use CPU timing (fallback or default)
        model_cpu = model.to(torch.device('cpu'))
        input_cpu = input_tensor.to(torch.device('cpu'))
        return self._measure_inference_time_on_device(model_cpu, input_cpu, torch.device('cpu'),
                                                    warmup_iterations, benchmark_iterations)
    
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
        
        # Check if FX quantization is available
        if prepare_fx is None or convert_fx is None:
            raise RuntimeError("FX quantization is not available in this PyTorch version")
        
        self.original_model = copy.deepcopy(model)
        model_copy = copy.deepcopy(model)
        model_copy.eval()
        
        # Create qconfig mapping
        qconfig = torch.quantization.get_default_qconfig(self.backend)
        if NEW_API:
            qconfig_mapping = QConfigMapping().set_global(qconfig)
        else:
            # For older PyTorch versions, use different approach
            qconfig_mapping = {'': qconfig}
        
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
    """INT8 quantization using custom observers - optimized for GPU when possible"""
    
    def __init__(self, device: torch.device = None):
        super().__init__(device)
        # Prefer faster device for evaluation when possible
        self.preferred_eval_device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def evaluate_model(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Evaluate INT8 quantized model, trying GPU first for better performance
        """
        # Try GPU evaluation first if available
        if self.preferred_eval_device.type == 'cuda':
            try:
                model_gpu = model.to(self.preferred_eval_device)
                # Test with a small batch
                test_input = next(iter(data_loader))[0][:1].to(self.preferred_eval_device)
                with torch.no_grad():
                    _ = model_gpu(test_input)
                # If successful, use GPU
                return self._evaluate_model_on_device(model_gpu, data_loader, self.preferred_eval_device)
            except Exception:
                # Fall back to CPU if GPU fails
                pass
        
        # Use CPU evaluation (fallback or default)
        model_cpu = model.to(torch.device('cpu'))
        cpu_data_loader = self._create_cpu_dataloader(data_loader)
        return self._evaluate_model_on_device(model_cpu, cpu_data_loader, torch.device('cpu'))
    
    def measure_inference_time(self, model: nn.Module, input_tensor: torch.Tensor, 
                             warmup_iterations: int = 10, benchmark_iterations: int = 100) -> Dict[str, float]:
        """
        Measure inference time, trying GPU first for better performance
        """
        # Try GPU timing first if available
        if self.preferred_eval_device.type == 'cuda':
            try:
                model_gpu = model.to(self.preferred_eval_device)
                input_gpu = input_tensor.to(self.preferred_eval_device)
                # Test inference
                with torch.no_grad():
                    _ = model_gpu(input_gpu)
                return self._measure_inference_time_on_device(model_gpu, input_gpu, self.preferred_eval_device,
                                                            warmup_iterations, benchmark_iterations)
            except Exception:
                # Fall back to CPU if GPU fails
                pass
        
        # Use CPU timing (fallback or default)
        model_cpu = model.to(torch.device('cpu'))
        input_cpu = input_tensor.to(torch.device('cpu'))
        return self._measure_inference_time_on_device(model_cpu, input_cpu, torch.device('cpu'),
                                                    warmup_iterations, benchmark_iterations)
    
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
        
        print("Attempting INT8 quantization...")
        
        try:
            # Add quant/dequant stubs for static quantization
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
            
            wrapped_model = QuantizedModel(model_copy)
            
            # Custom QConfig with INT8 observers using per_tensor_affine (safer approach)
            custom_qconfig = torch.quantization.get_default_qconfig('fbgemm')
            wrapped_model.qconfig = custom_qconfig
            
            # Prepare model for quantization
            print("Preparing model for INT8 quantization...")
            model_prepared = torch.quantization.prepare(wrapped_model)
            
            # Calibrate with calibration data
            print("Calibrating INT8 model...")
            self._calibrate_safely(model_prepared, calibration_data)
            
            # Convert to quantized model
            print("Converting to INT8 quantized model...")
            quantized_model = torch.quantization.convert(model_prepared)
            
            print("INT8 quantization completed successfully!")
            
        except Exception as int8_error:
            print(f"INT8 quantization failed: {int8_error}")
            print("WARNING: INT8 quantization failed. Returning original model (no quantization applied).")
            quantized_model = model_copy
            # Mark this as a quantization failure but still return a working model
            quantized_model._quantization_failed = True
        
        self.quantized_model = quantized_model
        return quantized_model
    
    def _calibrate_safely(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader):
        """Calibrate the model with calibration data with error handling"""
        model.eval()
        print("Starting INT8 calibration...")
        
        with torch.no_grad():
            for i, (inputs, _) in enumerate(calibration_data):
                try:
                    inputs = inputs.to(self.device)
                    _ = model(inputs)
                    
                    # Progress indicator
                    if i % 10 == 0:
                        print(f"INT8 calibration progress: {i}/{len(calibration_data)} batches")
                        
                except Exception as batch_error:
                    print(f"Error in INT8 calibration batch {i}: {batch_error}")
                    # Continue with next batch instead of failing completely
                    continue
                    
        print("INT8 calibration completed.")


class TorchvisionQuantizer(BaseQuantizer):
    """
    Quantizer using PyTorch's built-in QuantizableResNet models from torchvision.
    This is the recommended approach for ResNet quantization as per PyTorch documentation.
    """
    
    def __init__(self, device: torch.device = None, backend: str = 'fbgemm'):
        super().__init__(device)
        self.backend = backend
        torch.backends.quantized.engine = backend
        # Built-in quantized models generally work better on CPU
        self.eval_device = torch.device('cpu')
        
        if not QUANTIZABLE_MODELS_AVAILABLE:
            raise ImportError(
                "Torchvision quantizable models are not available. "
                "Please upgrade torchvision to a version that supports quantization."
            )
    
    def _get_quantizable_model(self, model_name: str, num_classes: int = 1000, pretrained: bool = True):
        """Get the appropriate quantizable model based on the model name"""
        model_name_lower = model_name.lower()
        
        if 'resnet18' in model_name_lower:
            if pretrained and ResNet18_QuantizedWeights is not None:
                # Load with pre-quantized weights if available
                weights = ResNet18_QuantizedWeights.DEFAULT
                model = quantizable_resnet18(weights=weights, quantize=True)
            else:
                # Load without weights or prepare for quantization
                model = quantizable_resnet18(weights=None, quantize=False)
                if num_classes != 1000:
                    # Modify classifier for different number of classes
                    model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
            
        elif 'resnet50' in model_name_lower:
            if pretrained and ResNet50_QuantizedWeights is not None:
                # Load with pre-quantized weights if available
                weights = ResNet50_QuantizedWeights.DEFAULT
                model = quantizable_resnet50(weights=weights, quantize=True)
            else:
                # Load without weights or prepare for quantization
                model = quantizable_resnet50(weights=None, quantize=False)
                if num_classes != 1000:
                    # Modify classifier for different number of classes
                    model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
        else:
            raise ValueError(
                f"Model {model_name} is not supported by TorchvisionQuantizer. "
                "Supported models: resnet18, resnet50"
            )
    
    def quantize(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader = None) -> nn.Module:
        """
        Quantize using torchvision's built-in quantizable models
        
        Args:
            model: Original model to quantize (used to determine architecture)
            calibration_data: Calibration data for post-training quantization
            
        Returns:
            Quantized model using torchvision's implementation
        """
        # Determine model type from the original model
        model_name = self._determine_model_type(model)
        num_classes = self._get_num_classes(model)
        
        print(f"Creating quantizable {model_name} with {num_classes} classes...")
        
        try:
            # Get the quantizable version
            quantizable_model = self._get_quantizable_model(model_name, num_classes, pretrained=False)
            
            # Transfer weights from original model if possible
            self._transfer_weights(model, quantizable_model)
            
            # Prepare for quantization
            quantizable_model.qconfig = torch.quantization.get_default_qconfig(self.backend)
            quantizable_model_prepared = torch.quantization.prepare(quantizable_model)
            
            # Calibrate if data is provided
            if calibration_data is not None:
                print("Calibrating quantizable model...")
                self._calibrate_safely(quantizable_model_prepared, calibration_data)
            
            # Convert to quantized model
            print("Converting to quantized model...")
            quantized_model = torch.quantization.convert(quantizable_model_prepared)
            
            # Move to evaluation device safely
            print("Moving quantized model to evaluation device...")
            quantized_model = quantized_model.to(self.eval_device)
            quantized_model.eval()  # Ensure model is in eval mode
            
            print("Torchvision quantization completed successfully!")
            
        except Exception as e:
            print(f"Torchvision quantization failed: {e}")
            print("Falling back to original model...")
            quantized_model = model.to(self.eval_device)
            quantized_model._quantization_failed = True
        
        self.quantized_model = quantized_model
        return quantized_model
    
    def _determine_model_type(self, model: nn.Module) -> str:
        """Determine the model type from the model instance"""
        model_class_name = model.__class__.__name__.lower()
        
        if 'resnet18' in model_class_name or hasattr(model, 'layer4') and len(list(model.layer1.children())) == 2:
            return 'resnet18'
        elif 'resnet50' in model_class_name or hasattr(model, 'layer4') and len(list(model.layer1.children())) == 3:
            return 'resnet50'
        else:
            # Default to resnet18 if cannot determine
            print(f"Warning: Could not determine model type from {model_class_name}, defaulting to resnet18")
            return 'resnet18'
    
    def _get_num_classes(self, model: nn.Module) -> int:
        """Get the number of output classes from the model"""
        if hasattr(model, 'fc') and hasattr(model.fc, 'out_features'):
            return model.fc.out_features
        elif hasattr(model, 'classifier') and hasattr(model.classifier, 'out_features'):
            return model.classifier.out_features
        else:
            return 1000  # Default ImageNet classes
    
    def _transfer_weights(self, source_model: nn.Module, target_model: nn.Module):
        """Transfer weights from source model to target quantizable model"""
        try:
            source_dict = source_model.state_dict()
            target_dict = target_model.state_dict()
            
            # Filter and transfer compatible weights
            transferred_keys = []
            for key in source_dict:
                if key in target_dict and source_dict[key].shape == target_dict[key].shape:
                    target_dict[key] = source_dict[key]
                    transferred_keys.append(key)
            
            target_model.load_state_dict(target_dict)
            print(f"Transferred {len(transferred_keys)} weight tensors to quantizable model")
            
        except Exception as e:
            print(f"Warning: Could not transfer weights: {e}")
            print("Proceeding with random initialization...")
    
    def _calibrate_safely(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader):
        """Calibrate the model with calibration data with error handling"""
        model.eval()
        model = model.to(self.eval_device)
        
        print("Starting torchvision model calibration...")
        
        with torch.no_grad():
            for i, (inputs, _) in enumerate(calibration_data):
                try:
                    inputs = inputs.to(self.eval_device)
                    _ = model(inputs)
                    
                    # Progress indicator
                    if i % 10 == 0:
                        print(f"Calibration progress: {i}/{len(calibration_data)} batches")
                        
                except Exception as batch_error:
                    print(f"Error in calibration batch {i}: {batch_error}")
                    continue
                    
        print("Torchvision model calibration completed.")
    
    def evaluate_model(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Evaluate torchvision quantized model on CPU (recommended for quantized models)
        """
        try:
            # Ensure model is on CPU for optimal quantized model performance
            model_cpu = model.to(self.eval_device)
            model_cpu.eval()
            cpu_data_loader = self._create_cpu_dataloader(data_loader)
            return self._evaluate_model_on_device(model_cpu, cpu_data_loader, self.eval_device)
        except Exception as e:
            print(f"Error during TorchvisionQuantizer evaluation: {e}")
            # Return default metrics if evaluation fails
            return {
                'accuracy': 0.0,
                'loss': float('inf'),
                'correct': 0,
                'total': 1
            }
    
    def measure_inference_time(self, model: nn.Module, input_tensor: torch.Tensor, 
                             warmup_iterations: int = 10, benchmark_iterations: int = 100) -> Dict[str, float]:
        """
        Measure inference time for torchvision quantized model on CPU
        """
        try:
            # Ensure model and input are on CPU for optimal quantized model performance
            model_cpu = model.to(self.eval_device)
            model_cpu.eval()
            input_cpu = input_tensor.to(self.eval_device)
            return self._measure_inference_time_on_device(model_cpu, input_cpu, self.eval_device, 
                                                        warmup_iterations, benchmark_iterations)
        except Exception as e:
            print(f"Error during TorchvisionQuantizer timing: {e}")
            # Return default timing if measurement fails
            return {
                'mean_time': float('inf'),
                'std_time': 0.0,
                'min_time': float('inf'),
                'max_time': float('inf')
            }


class QuantizationBenchmark:
    """Benchmark different quantization methods"""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Use CPU only for quantization operations (PyTorch limitation)
        self.quantization_device = torch.device('cpu')
        self.results = {}
        print(f"QuantizationBenchmark initialized with evaluation device: {self.device}")
        print(f"Quantization operations will use: {self.quantization_device}")
        print("📋 Device Selection Policy:")
        print("   • Dynamic quantization: CPU evaluation (user preference)")
        print("   • Static/FX/INT8/QAT: GPU evaluation (when possible) for speed")
    
    def benchmark_quantization_methods(self, model: nn.Module, 
                                     calibration_data: torch.utils.data.DataLoader,
                                     evaluation_data: torch.utils.data.DataLoader,
                                     methods: list = None) -> Dict[str, Dict[str, Any]]:
        """
        Benchmark multiple quantization methods
        
        Args:
            model: Original model to quantize
            calibration_data: Calibration data loader
            evaluation_data: Evaluation data loader
            methods: List of quantization methods to benchmark
            
        Returns:
            Dictionary with benchmark results for each method
        """
        if methods is None:
            methods = ['dynamic', 'static', 'fx', 'int8']
        
        quantizers = {
            'dynamic': DynamicQuantizer(self.quantization_device),
            'static': StaticQuantizer(self.quantization_device),
            'qat': QATQuantizer(self.quantization_device),
            'fx': FXQuantizer(self.quantization_device),
            'int8': INT8Quantizer(self.quantization_device)
        }
        
        # Add TorchvisionQuantizer if available
        if QUANTIZABLE_MODELS_AVAILABLE:
            quantizers['torchvision'] = TorchvisionQuantizer(self.quantization_device)
        
        results = {}
        
        # Evaluate original model on the preferred device (CUDA if available)
        print(f"Evaluating original model on {self.device}...")
        original_model = copy.deepcopy(model).to(self.device)
        original_metrics = self._evaluate_model_on_device(original_model, evaluation_data, self.device)
        example_input = next(iter(evaluation_data))[0][:1]
        original_timing = self._measure_inference_time_on_device(original_model, example_input, self.device)
        
        results['original'] = {
            'accuracy': original_metrics['accuracy'],
            'loss': original_metrics['loss'],
            'inference_time_ms': original_timing['mean_time_ms'],
            'model_size_mb': self._get_model_size(original_model)
        }
        
        # Benchmark each quantization method
        for method in methods:
            if method not in quantizers:
                print(f"Warning: Unknown quantization method '{method}', skipping...")
                continue
            
            try:
                print(f"Benchmarking {method} quantization...")
                quantizer = quantizers[method]
                
                # Move model to CPU for quantization (PyTorch quantization requirement)
                print(f"Moving model to {self.quantization_device} for quantization...")
                model_cpu = copy.deepcopy(model).to(self.quantization_device)
                
                if method == 'dynamic':
                    quantized_model = quantizer.quantize(model_cpu)
                else:
                    # Create CPU calibration data loader
                    cpu_calibration_data = self._create_cpu_dataloader(calibration_data)
                    quantized_model = quantizer.quantize(model_cpu, cpu_calibration_data)
                
                # Determine the best device for evaluating the quantized model
                eval_device, eval_data_loader = self._get_optimal_eval_setup(quantized_model, method, evaluation_data)
                
                # Evaluate quantized model using method-specific optimization
                print(f"Evaluating {method} quantized model on {eval_device}...")
                if method == 'dynamic':
                    # DynamicQuantizer has its own optimized evaluation
                    quantized_metrics = quantizer.evaluate_model(quantized_model, evaluation_data)
                    eval_example_input = example_input.to(quantizer.eval_device)
                    quantized_timing = quantizer.measure_inference_time(quantized_model, eval_example_input)
                else:
                    # Other quantizers: move model to optimal device and evaluate
                    quantized_model = quantized_model.to(eval_device)
                    quantized_metrics = quantizer.evaluate_model(quantized_model, eval_data_loader)
                    eval_example_input = example_input.to(eval_device)
                    quantized_timing = quantizer.measure_inference_time(quantized_model, eval_example_input)
                
                results[method] = {
                    'accuracy': quantized_metrics['accuracy'],
                    'loss': quantized_metrics['loss'],
                    'inference_time_ms': quantized_timing['mean_time_ms'],
                    'model_size_mb': self._get_model_size(quantized_model),
                    'accuracy_drop': original_metrics['accuracy'] - quantized_metrics['accuracy'],
                    'speedup': original_timing['mean_time_ms'] / quantized_timing['mean_time_ms'],
                    'compression_ratio': self._get_model_size(original_model) / self._get_model_size(quantized_model)
                }
            except Exception as e:
                print(f"Error benchmarking {method} quantization: {e}")
                results[method] = {
                    'error': str(e)
                }
        
        self.results = results
        return results
    
    def _create_cpu_dataloader(self, dataloader):
        """Create a CPU version of the dataloader for quantization"""
        class CPUDataLoader:
            def __init__(self, original_loader):
                self.original_loader = original_loader
                self.dataset = original_loader.dataset
                
            def __iter__(self):
                for batch in self.original_loader:
                    inputs, targets = batch
                    yield inputs.to('cpu'), targets.to('cpu')
                    
            def __len__(self):
                return len(self.original_loader)
        
        return CPUDataLoader(dataloader)
    
    def _evaluate_model_on_device(self, model: nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device) -> Dict[str, float]:
        """
        Evaluate model performance on a specific device
        
        Args:
            model: Model to evaluate
            data_loader: Data loader for evaluation
            device: Device to run evaluation on
            
        Returns:
            Dictionary with evaluation metrics
        """
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
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
    
    def _measure_inference_time_on_device(self, model: nn.Module, input_tensor: torch.Tensor, device: torch.device,
                                        warmup_iterations: int = 10, benchmark_iterations: int = 100) -> Dict[str, float]:
        """
        Measure model inference time on a specific device
        
        Args:
            model: Model to benchmark
            input_tensor: Input tensor for inference
            device: Device to run inference on
            warmup_iterations: Number of warmup iterations
            benchmark_iterations: Number of benchmark iterations
            
        Returns:
            Dictionary with timing statistics
        """
        model.eval()
        input_tensor = input_tensor.to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(input_tensor)
        
        # Synchronize CUDA if using GPU
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(benchmark_iterations):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.time()
                _ = model(input_tensor)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        return {
            'mean_time_ms': sum(times) / len(times),
            'min_time_ms': min(times),
            'max_time_ms': max(times),
            'std_time_ms': (sum([(t - sum(times)/len(times))**2 for t in times]) / len(times))**0.5
        }
    
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
        
        print("\n" + "="*80)
        print("QUANTIZATION BENCHMARK RESULTS")
        print("="*80)
        print(f"{'Method':<12} {'Accuracy':<10} {'Loss':<8} {'Time(ms)':<10} {'Size(MB)':<10} {'Speedup':<8} {'Compression':<12}")
        print("-"*80)
        
        for method, results in self.results.items():
            if 'error' in results:
                print(f"{method:<12} ERROR: {results['error']}")
                continue
                
            accuracy = f"{results['accuracy']:.2f}%"
            loss = f"{results['loss']:.4f}"
            time_ms = f"{results['inference_time_ms']:.2f}"
            size_mb = f"{results['model_size_mb']:.2f}"
            
            if method == 'original':
                speedup = "1.00x"
                compression = "1.00x"
            else:
                speedup = f"{results.get('speedup', 0):.2f}x"
                compression = f"{results.get('compression_ratio', 0):.2f}x"
            
            print(f"{method:<12} {accuracy:<10} {loss:<8} {time_ms:<10} {size_mb:<10} {speedup:<8} {compression:<12}")
    
    def _get_optimal_eval_setup(self, quantized_model: nn.Module, method: str, evaluation_data):
        """
        Determine the optimal device and data loader for evaluating quantized models
        
        Args:
            quantized_model: The quantized model
            method: Quantization method used
            evaluation_data: Original evaluation data loader
            
        Returns:
            Tuple of (optimal_device, appropriate_data_loader)
        """
        # Check device selection policy for the quantization method
        can_use_gpu = self._can_quantized_model_use_gpu(quantized_model, method)
        
        # Dynamic quantization: Force CPU evaluation (user requirement)
        if method == 'dynamic':
            print(f"🔄 {method} quantization configured to use CPU for evaluation (user preference)")
            cpu_evaluation_data = self._create_cpu_dataloader(evaluation_data)
            return self.quantization_device, cpu_evaluation_data
        
        # For other methods: Try GPU first for better performance
        if can_use_gpu and self.device.type == 'cuda':
            # Try to move model to GPU and use GPU evaluation
            try:
                quantized_model_gpu = quantized_model.to(self.device)
                # Test with a small batch to ensure it works
                test_input = next(iter(evaluation_data))[0][:1].to(self.device)
                with torch.no_grad():
                    _ = quantized_model_gpu(test_input)
                
                print(f"🚀 {method} quantized model supports GPU evaluation - using GPU for speed")
                return self.device, evaluation_data
                
            except Exception as e:
                print(f"⚠️  {method} quantized model failed GPU test: {e}")
                print(f"   Falling back to CPU evaluation")
        
        # Fall back to CPU evaluation
        print(f"💻 {method} quantized model using CPU evaluation")
        cpu_evaluation_data = self._create_cpu_dataloader(evaluation_data)
        return self.quantization_device, cpu_evaluation_data
    
    def _can_quantized_model_use_gpu(self, model: nn.Module, method: str) -> bool:
        """
        Check if a quantized model can potentially run on GPU
        
        Args:
            model: Quantized model to check
            method: Quantization method used
            
        Returns:
            Boolean indicating if GPU evaluation should be attempted
        """
        # Dynamic quantization: Force CPU evaluation as per user requirement
        if method == 'dynamic':
            return False
        
        # QAT models (before conversion) typically support GPU
        if method == 'qat':
            return True
        
        # Static, FX, and INT8 quantized models: Try GPU first for better performance
        # If they fail, we'll fall back to CPU
        if method in ['static', 'fx', 'int8']:
            return True
        
        # For unknown methods, attempt GPU by default
        return True


def get_available_quantizers() -> Dict[str, BaseQuantizer]:
    """Get dictionary of available quantizers"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    quantizers = {
        'dynamic': DynamicQuantizer(device),
        'static': StaticQuantizer(device),
        'qat': QATQuantizer(device),
        'fx': FXQuantizer(device),
        'int8': INT8Quantizer(device)
    }
    
    # Add TorchvisionQuantizer if available
    if QUANTIZABLE_MODELS_AVAILABLE:
        quantizers['torchvision'] = TorchvisionQuantizer(device)
    
    return quantizers