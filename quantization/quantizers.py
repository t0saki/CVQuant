"""
Quantization methods implementation for PyTorch models
"""
import torch
import torch.nn as nn
import torch.quantization as quant
from torch.quantization import QConfig, default_observer, default_weight_observer
from torch.quantization.quantize_fx import prepare_fx, convert_fx
import copy
from typing import Dict, Any, Callable, Optional, Tuple
import time
import os


class BaseQuantizer:
    """Base class for all quantization methods"""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cpu')
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
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        # Determine if model is quantized (should be on CPU)
        model_device = next(model.parameters()).device
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(model_device), targets.to(model_device)
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
        model.eval()
        
        # Move input to the same device as the model
        model_device = next(model.parameters()).device
        input_tensor = input_tensor.to(model_device)
        
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
        qconfig_mapping = torch.quantization.QConfigMapping().set_global(qconfig)
        
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


class QuantizationBenchmark:
    """Benchmark different quantization methods"""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cpu')
        self.results = {}
    
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
            'dynamic': DynamicQuantizer(self.device),
            'static': StaticQuantizer(self.device),
            'qat': QATQuantizer(self.device),
            'fx': FXQuantizer(self.device),
            'int8': INT8Quantizer(self.device)
        }
        
        results = {}
        
        # Evaluate original model
        original_metrics = quantizers['dynamic'].evaluate_model(model, evaluation_data)
        example_input = next(iter(evaluation_data))[0][:1]
        original_timing = quantizers['dynamic'].measure_inference_time(model, example_input)
        
        results['original'] = {
            'accuracy': original_metrics['accuracy'],
            'loss': original_metrics['loss'],
            'inference_time_ms': original_timing['mean_time_ms'],
            'model_size_mb': self._get_model_size(model)
        }
        
        # Benchmark each quantization method
        for method in methods:
            if method not in quantizers:
                print(f"Warning: Unknown quantization method '{method}', skipping...")
                continue
            
            try:
                print(f"Benchmarking {method} quantization...")
                quantizer = quantizers[method]
                
                if method == 'dynamic':
                    quantized_model = quantizer.quantize(model)
                else:
                    quantized_model = quantizer.quantize(model, calibration_data)
                
                # Evaluate quantized model
                quantized_metrics = quantizer.evaluate_model(quantized_model, evaluation_data)
                quantized_timing = quantizer.measure_inference_time(quantized_model, example_input)
                
                results[method] = {
                    'accuracy': quantized_metrics['accuracy'],
                    'loss': quantized_metrics['loss'],
                    'inference_time_ms': quantized_timing['mean_time_ms'],
                    'model_size_mb': self._get_model_size(quantized_model),
                    'accuracy_drop': original_metrics['accuracy'] - quantized_metrics['accuracy'],
                    'speedup': original_timing['mean_time_ms'] / quantized_timing['mean_time_ms'],
                    'compression_ratio': self._get_model_size(model) / self._get_model_size(quantized_model)
                }
            except Exception as e:
                print(f"Error benchmarking {method} quantization: {e}")
                results[method] = {
                    'error': str(e)
                }
        
        self.results = results
        return results
    
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