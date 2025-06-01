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
        self.qconfig_spec = qconfig_spec or {nn.Linear, nn.Conv2d}  # nn.LSTM, nn.GRU, etc. can also be added
    
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
        model_copy.eval() # Ensure model is in eval mode for dynamic quantization
        
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
    
    def quantize(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader) -> nn.Module:
        """
        Apply static quantization to the model.
        This involves preparing the model, calibrating with data, and converting.
        """
        self.original_model = copy.deepcopy(model)
        model_to_quantize = copy.deepcopy(model)

        model_to_quantize.to('cpu').eval() # Static quantization preparation is typically on CPU and in eval mode

        torch.backends.quantized.engine = self.backend

        model_to_quantize.qconfig = torch.quantization.get_default_qconfig(self.backend)
        
        if hasattr(model_to_quantize, 'fuse_model') and callable(model_to_quantize.fuse_model):
            print(f"StaticQuant: Calling model.fuse_model() before prepare.")
            model_to_quantize.fuse_model()
        # else:
            # print(f"StaticQuant: model.fuse_model() not found. Consider manual fusion if needed.")

        print(f"StaticQuant: Preparing model with backend {self.backend}.")
        torch.quantization.prepare(model_to_quantize, inplace=True)

        print("StaticQuant: Calibrating model...")
        self._calibrate(model_to_quantize, calibration_data)

        print("StaticQuant: Converting model...")
        quantized_model = torch.quantization.convert(model_to_quantize, inplace=True)
        
        quantized_model.to(self.device) # Move final model to target device

        self.quantized_model = quantized_model
        return self.quantized_model
    
    def _calibrate(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader):
        model.eval()
        model.to('cpu') # Ensure model and data are on CPU for calibration
        with torch.no_grad():
            for inputs, _ in calibration_data:
                inputs = inputs.to('cpu')
                model(inputs)
        print("StaticQuant: Calibration finished.")


class QATQuantizer(BaseQuantizer):
    """Quantization Aware Training implementation"""
    
    def __init__(self, device: torch.device = None, backend: str = 'fbgemm',
                 train_epochs: int = 10, lr: float = 1e-4):
        super().__init__(device)
        self.backend = backend
        self.train_epochs = train_epochs
        self.lr = lr
    
    def quantize(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader) -> nn.Module:
        """
        Apply Quantization Aware Training (QAT) to the model.
        Involves preparing for QAT, fine-tuning, and converting to a quantized model.
        """
        self.original_model = copy.deepcopy(model)
        model_to_quantize = copy.deepcopy(model)

        model_to_quantize.to(self.device) # QAT training happens on the target device
        model_to_quantize.train()

        torch.backends.quantized.engine = self.backend
        qconfig = torch.quantization.get_default_qat_qconfig(self.backend)
        model_to_quantize.qconfig = qconfig

        if hasattr(model_to_quantize, 'fuse_model') and callable(model_to_quantize.fuse_model):
            print(f"QAT: Calling model.fuse_model() before prepare_qat.")
            model_to_quantize.fuse_model()
        # else:
            # print(f"QAT: model.fuse_model() not found. Proceeding with prepare_qat.")

        print(f"QAT: Preparing model for QAT with backend {self.backend}.")
        model_prepared = torch.quantization.prepare_qat(model_to_quantize)

        optimizer = torch.optim.Adam(model_prepared.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        print(f"QAT: Starting fine-tuning for {self.train_epochs} epochs with LR {self.lr}.")
        for epoch in range(self.train_epochs):
            epoch_loss = 0.0
            num_batches = 0
            for inputs, targets in calibration_data:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model_prepared(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            print(f"QAT Epoch {epoch + 1}/{self.train_epochs} - Avg Loss: {avg_epoch_loss:.4f}")

        model_prepared.eval()
        quantized_model = torch.quantization.convert(model_prepared.to('cpu')) # Convert on CPU
        quantized_model.to(self.device) # Move final model to target device

        self.quantized_model = quantized_model
        print("QAT: Model quantization and conversion complete.")
        return self.quantized_model


class FXQuantizer(BaseQuantizer):
    """FX Graph Mode quantization implementation"""
    
    def __init__(self, device: torch.device = None, backend: str = 'fbgemm'):
        super().__init__(device)
        self.backend = backend
    
    def quantize(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader) -> nn.Module:
        """
        Apply FX Graph Mode quantization.
        Requires the model to be FX traceable.
        """
        self.original_model = copy.deepcopy(model)
        model_to_quantize = copy.deepcopy(model)

        model_to_quantize.to('cpu').eval() # FX quantization is typically done on CPU

        torch.backends.quantized.engine = self.backend
        
        qconfig = torch.quantization.get_default_qconfig(self.backend)
        qconfig_mapping = torch.quantization.QConfigMapping().set_global(qconfig)
        
        print(f"FXQuant: Preparing model with backend {self.backend}.")
        try:
            # Ensure there's at least one batch in calibration_data
            if not calibration_data:
                raise ValueError("Calibration data loader is empty or None, cannot get example_inputs for FX.")
            
            example_inputs = torch.randn(1, *next(iter(calibration_data))[0].shape[1:]).to('cpu')
            prepared_model = prepare_fx(model_to_quantize, qconfig_mapping, example_inputs)
        except Exception as e:
            print(f"FXQuant: Error during prepare_fx. Model might not be FX traceable: {e}")
            print("FXQuant: Falling back to Eager Mode Static Quantization for this model.")
            eager_static_quantizer = StaticQuantizer(device=self.device, backend=self.backend)
            return eager_static_quantizer.quantize(model, calibration_data)

        print("FXQuant: Calibrating model...")
        self._calibrate(prepared_model, calibration_data)

        print("FXQuant: Converting model...")
        quantized_model = convert_fx(prepared_model)
        
        quantized_model.to(self.device) # Move final model to target device

        self.quantized_model = quantized_model
        return self.quantized_model
    
    def _calibrate(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader):
        model.eval()
        model.to('cpu')
        with torch.no_grad():
            for inputs, _ in calibration_data:
                inputs = inputs.to('cpu')
                model(inputs)
        print("FXQuant: Calibration finished.")


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