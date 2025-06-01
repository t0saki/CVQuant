"""
Quantization methods implementation for PyTorch models
"""
import torch
import torch.nn as nn
import torch.quantization as quant
from torch.quantization import QConfig, default_observer, default_weight_observer
try:
    # Try new API first
    from torch.ao.quantization import QConfigMapping
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
    NEW_API = True
except ImportError:
    # Fallback to old API
    from torch.quantization.quantize_fx import prepare_fx, convert_fx
    NEW_API = False
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
    """Dynamic quantization implementation"""
    
    def __init__(self, device: torch.device = None, qconfig_spec: Dict = None):
        super().__init__(device)
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
        
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model_copy,
            self.qconfig_spec,
            dtype=torch.qint8
        )
        
        self.quantized_model = quantized_model
        return quantized_model


class StaticQuantizer(BaseQuantizer):
    """Static quantization implementation"""
    
    def __init__(self, device: torch.device = None, backend: str = 'fbgemm'):
        super().__init__(device)
        self.backend = backend
        torch.backends.quantized.engine = backend
    
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
        
        try:
            # Try FX Graph Mode quantization first (more robust)
            from torch.ao.quantization import get_default_qconfig, prepare_fx, convert_fx
            from torch.ao.quantization.qconfig_mapping import QConfigMapping
            
            # Create QConfigMapping for FX mode
            qconfig_mapping = QConfigMapping().set_global(get_default_qconfig("fbgemm"))
            
            # Get example input
            example_inputs = next(iter(calibration_data))[0][:1].to(self.device)
            
            # Prepare model with FX graph mode
            model_prepared = prepare_fx(model_copy, qconfig_mapping, example_inputs)
            
            # Calibrate with calibration data
            self._calibrate(model_prepared, calibration_data)
            
            # Convert to quantized model
            quantized_model = convert_fx(model_prepared)
            
        except Exception as fx_error:
            print(f"FX quantization failed: {fx_error}")
            print("Falling back to eager mode quantization...")
            
            # Fallback to eager mode with simplified model structure
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
                
                # Set quantization configuration with per_tensor_affine
                qconfig = torch.quantization.QConfig(
                    activation=torch.quantization.default_observer.with_args(qscheme=torch.per_tensor_affine),
                    weight=torch.quantization.default_weight_observer.with_args(qscheme=torch.per_tensor_affine)
                )
                wrapped_model.qconfig = qconfig
                
                # Prepare model for quantization
                model_prepared = torch.quantization.prepare(wrapped_model)
                
                # Calibrate with calibration data
                self._calibrate(model_prepared, calibration_data)
                
                # Convert to quantized model
                quantized_model = torch.quantization.convert(model_prepared)
                
            except Exception as eager_error:
                print(f"Eager mode quantization also failed: {eager_error}")
                # Return the original model if quantization fails completely
                print("WARNING: Static quantization failed completely. This may be due to unsupported operations in the model.")
                print("Returning original model (no quantization applied).")
                quantized_model = model_copy
                # Mark this as a quantization failure but still return a working model
                quantized_model._quantization_failed = True
        
        self.quantized_model = quantized_model
        return quantized_model
    
    def _calibrate(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader):
        """Calibrate the model with calibration data"""
        model.eval()
        with torch.no_grad():
            for inputs, _ in calibration_data:
                inputs = inputs.to(self.device)
                model(inputs)


class QATQuantizer(BaseQuantizer):
    """Quantization Aware Training implementation"""
    
    def __init__(self, device: torch.device = None, backend: str = 'fbgemm'):
        super().__init__(device)
        self.backend = backend
        torch.backends.quantized.engine = backend
    
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
    """FX Graph Mode quantization implementation"""
    
    def __init__(self, device: torch.device = None, backend: str = 'fbgemm'):
        super().__init__(device)
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
    """INT8 quantization using custom observers"""
    
    def __init__(self, device: torch.device = None):
        super().__init__(device)
    
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
        
        # Custom QConfig with INT8 observers using per_tensor_affine
        custom_qconfig = QConfig(
            activation=default_observer.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
            weight=default_weight_observer.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine)
        )
        
        wrapped_model.qconfig = custom_qconfig
        
        # Prepare model for quantization
        model_prepared = torch.quantization.prepare(wrapped_model)
        
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


class QuantizationBenchmark:
    """Benchmark different quantization methods"""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Use CPU only for quantization operations (PyTorch limitation)
        self.quantization_device = torch.device('cpu')
        self.results = {}
        print(f"QuantizationBenchmark initialized with device: {self.device}")
        print(f"Quantization operations will use: {self.quantization_device}")
    
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
                
                # Create CPU evaluation data loader for quantized model
                cpu_evaluation_data = self._create_cpu_dataloader(evaluation_data)
                
                # Evaluate quantized model on CPU
                print(f"Evaluating {method} quantized model on {self.quantization_device}...")
                quantized_metrics = quantizer.evaluate_model(quantized_model, cpu_evaluation_data)
                
                # Measure timing on CPU with CPU input
                cpu_example_input = example_input.to(self.quantization_device)
                quantized_timing = quantizer.measure_inference_time(quantized_model, cpu_example_input)
                
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


def get_available_quantizers() -> Dict[str, BaseQuantizer]:
    """Get dictionary of available quantizers"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return {
        'dynamic': DynamicQuantizer(device),
        'static': StaticQuantizer(device),
        'qat': QATQuantizer(device),
        'fx': FXQuantizer(device),
        'int8': INT8Quantizer(device)
    }