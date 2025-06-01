"""
Quantization methods implementation for PyTorch models
"""
import torch
import torch.nn as nn
import copy
import time # Added time for inference measurement
import torch.optim as optim
from torch.quantization import QConfig, default_observer, default_weight_observer, get_default_qat_qconfig
from torch.quantization.quantize_fx import prepare_fx, convert_fx, prepare_qat_fx
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from typing import Dict, Any, Callable, Optional, Tuple

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
        
        model_device = next(model.parameters()).device
        
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
        # Default implementation, can be overridden by subclasses if specific handling is needed
        model.eval()
        model_device = next(model.parameters()).device
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
                             warmup_iterations: int = 10, benchmark_iterations: int = 100) -> Dict[str, float]:
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
    """Quantization Aware Training implementation using FX Graph Mode"""
    
    def __init__(self, device: torch.device = None, backend: str = 'fbgemm', 
                 train_epochs: int = 3, learning_rate: float = 1e-4):
        super().__init__(device)
        self.backend = backend
        self.qconfig_mapping = get_default_qat_qconfig(self.backend)
        self.train_epochs = train_epochs
        self.learning_rate = learning_rate
        print(f"QATQuantizer initialized with device: {self.device}, backend: {self.backend}, epochs: {self.train_epochs}, lr: {self.learning_rate}")

    def quantize(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader) -> nn.Module:
        """
        Apply Quantization Aware Training (QAT) to the model.
        The 'calibration_data' is used as the training data for QAT.
        """
        self.original_model = copy.deepcopy(model)
        model_to_quantize = copy.deepcopy(self.original_model)
        
        # Model should be on the training device (self.device can be CUDA)
        model_to_quantize = model_to_quantize.to(self.device)
        model_to_quantize.train() # Set to train mode for QAT

        # Get example inputs for prepare_qat_fx
        try:
            # Ensure calibration_data is not empty and yields (inputs, targets)
            example_batch = next(iter(calibration_data))
            if isinstance(example_batch, (list, tuple)) and len(example_batch) > 0:
                example_inputs = example_batch[0].to(self.device)
            else: # Assuming loader yields only inputs if not a tuple/list
                example_inputs = example_batch.to(self.device)

            if not isinstance(example_inputs, torch.Tensor):
                raise ValueError(f"Example inputs must be a torch.Tensor, got {type(example_inputs)}")

        except StopIteration:
            raise ValueError("calibration_data (training_data_loader for QAT) is empty. Cannot get example_inputs.")
        except Exception as e:
            raise ValueError(f"Error getting example_inputs from calibration_data: {e}")
        
        print(f"Preparing model for QAT on device: {self.device} with example input shape: {example_inputs.shape}")
        # Prepare model for QAT using FX graph mode
        model_prepared = prepare_qat_fx(model_to_quantize, torch.ao.quantization.QConfigMapping().set_global(self.qconfig_mapping), (example_inputs,))

        # Fine-tune the model (QAT)
        print(f"Starting QAT training for {self.train_epochs} epochs on device {self.device}...")
        self._train_qat_loop(model_prepared, calibration_data)
        print("QAT training finished.")

        # Convert the QAT model to a quantized model (typically on CPU)
        print("Converting QAT model to quantized model (on CPU)...")
        quantized_model_cpu = model_prepared.eval().to('cpu')
        self.quantized_model = convert_fx(quantized_model_cpu)
        print("QAT model converted successfully.")
        
        return self.quantized_model

    def _train_qat_loop(self, model: nn.Module, train_loader: torch.utils.data.DataLoader):
        model.train() # Ensure model is in training mode
        # Model is already on self.device from the quantize method

        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.train_epochs):
            epoch_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total_samples += targets.size(0)
                correct_predictions += predicted.eq(targets).sum().item()
                
                if batch_idx % 50 == 0: # Print progress every 50 batches
                    print(f"  Epoch {epoch+1}/{self.train_epochs}, Batch {batch_idx}/{len(train_loader)} - Current Batch Loss: {loss.item():.4f}")

            avg_epoch_loss = epoch_loss / total_samples if total_samples > 0 else 0.0
            epoch_accuracy = 100. * correct_predictions / total_samples if total_samples > 0 else 0.0
            print(f"Epoch {epoch+1}/{self.train_epochs} - Avg Loss: {avg_epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    def convert_qat_model(self, qat_model: nn.Module) -> nn.Module:
        """
        Converts a QAT-trained model to a fully quantized model.
        Assumes qat_model has already been trained.
        """
        print("Converting externally trained QAT model to quantized model (on CPU)...")
        qat_model_cpu = qat_model.to('cpu')
        qat_model_cpu.eval()
        quantized_model = convert_fx(qat_model_cpu)
        self.quantized_model = quantized_model # Update self.quantized_model
        print("External QAT model converted successfully.")
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
        qconfig_mapping = torch.ao.quantization.QConfigMapping().set_global(qconfig)
        
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


class QuantizationBenchmark(BaseQuantizer):
    """Benchmark different quantization methods"""
    
    def __init__(self, device: torch.device = None, backend: str = 'fbgemm'): # Added backend
        self.device = device or torch.device('cpu')
        self.backend = backend # Store backend
        self.results = {}
        print(f"QuantizationBenchmark initialized with device: {self.device}, backend: {self.backend}")
    
    def benchmark_quantization_methods(self, model: nn.Module, 
                                     calibration_data: torch.utils.data.DataLoader,
                                     evaluation_data: torch.utils.data.DataLoader,
                                     methods: list = None) -> Dict[str, Dict[str, Any]]:
        """
        Benchmark different quantization methods on a given model.

        Args:
            model: Original model to quantize and benchmark.
            calibration_data: DataLoader for calibration (static, FX) or training (QAT).
            evaluation_data: DataLoader for evaluating model accuracy and performance.
            methods: List of quantization method names to benchmark (e.g., ['dynamic', 'static', 'qat']).
                     If None, all available methods are benchmarked.

        Returns:
            A dictionary containing benchmark results for each method.
        """
        self.results = {}
        # original_model_size = model.get_model_info().get('size_mb', 0.0) # Get original model size in MB
        original_model_size = 1 # TODO: Implement a method to get the model size in MB
        
        # Get an example input tensor for inference time measurement
        # Ensure evaluation_data is not empty and yields (inputs, targets) or just inputs
        try:
            example_batch_eval = next(iter(evaluation_data))
            if isinstance(example_batch_eval, (list, tuple)) and len(example_batch_eval) > 0:
                example_input, _ = example_batch_eval
            else: # Assuming loader yields only inputs
                example_input = example_batch_eval
            
            if not isinstance(example_input, torch.Tensor):
                raise ValueError(f"Example input for benchmark must be a torch.Tensor, got {type(example_input)}")

        except StopIteration:
            print("Warning: evaluation_data is empty. Cannot get example_input for inference time measurement.")
            example_input = None # Or create a dummy tensor based on model's expected input if known
        except Exception as e:
            print(f"Warning: Error getting example_input from evaluation_data: {e}. Inference time might not be measured.")
            example_input = None

        # Evaluate original model
        print("\nEvaluating original model...")
        if example_input is not None:
            original_time_stats = self.measure_inference_time(model, example_input.to(self.device)) # Original model on self.device
        else:
            original_time_stats = {k: 0 for k in ['mean_time_ms', 'min_time_ms', 'max_time_ms', 'std_time_ms']}
        original_eval_metrics = self.evaluate_model(model.to(self.device), evaluation_data) # Original model on self.device
        
        self.results['original'] = {
            'accuracy': original_eval_metrics.get('accuracy', 0.0),
            'loss': original_eval_metrics.get('loss', 0.0),
            'model_size_mb': original_model_size,
            **original_time_stats
        }
        print(f"Original Model - Accuracy: {self.results['original']['accuracy']:.2f}%, "
              f"Size: {original_model_size:.2f}MB, "
              f"Inference Time: {original_time_stats.get('mean_time_ms', 0.0):.2f}ms")

        # Get quantizers using the stored backend
        all_quantizers = get_available_quantizers(backend=self.backend, device=self.device)
        
        if methods is None:
            methods_to_run = list(all_quantizers.keys())
        else:
            methods_to_run = [m for m in methods if m in all_quantizers]

        for method_name in methods_to_run:
            quantizer = all_quantizers[method_name]
            print(f"\nBenchmarking {method_name} quantization...")
            
            try:
                # Quantize model
                # For QAT, calibration_data is training data. For others, it's for calibration.
                quantized_model = quantizer.quantize(copy.deepcopy(model), calibration_data)
                
                # Evaluate quantized model
                # Quantized models are typically on CPU after conversion
                quantized_model_device = next(quantized_model.parameters()).device
                print(f"Evaluating {method_name} quantized model on device: {quantized_model_device}")
                eval_metrics = quantizer.evaluate_model(quantized_model, evaluation_data) # Use quantizer's eval
                
                # Measure inference time
                if example_input is not None:
                    # Ensure example_input is on the same device as the quantized_model for timing
                    time_stats = quantizer.measure_inference_time(quantized_model, example_input.to(quantized_model_device))
                else:
                    time_stats = {k: 0 for k in ['mean_time_ms', 'min_time_ms', 'max_time_ms', 'std_time_ms']}

                model_size_mb = self._get_model_size(quantized_model)
                
                self.results[method_name] = {
                    'accuracy': eval_metrics.get('accuracy', 0.0),
                    'loss': eval_metrics.get('loss', 0.0),
                    'model_size_mb': model_size_mb,
                    'size_reduction_ratio': original_model_size / model_size_mb if model_size_mb > 0 else float('inf'),
                    **time_stats
                }
                print(f"{method_name.capitalize()} Quantized Model - Accuracy: {self.results[method_name]['accuracy']:.2f}%, "
                      f"Size: {model_size_mb:.2f}MB, "
                      f"Inference Time: {time_stats.get('mean_time_ms', 0.0):.2f}ms")

            except Exception as e:
                print(f"Error benchmarking {method_name}: {e}")
                import traceback
                traceback.print_exc()
                self.results[method_name] = {
                    'accuracy': 0.0, 'loss': 0.0, 'model_size_mb': 0.0, 
                    'error': str(e)
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

def get_available_quantizers(backend: str = 'fbgemm', device: torch.device = None) -> Dict[str, BaseQuantizer]: # Added backend and device
    """Get dictionary of available quantizers"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # QAT training parameters (epochs, lr) can be further customized if needed,
    # e.g., by passing them as arguments here or reading from a config.
    # For now, QATQuantizer uses its defaults or those passed to its constructor.
    qat_train_epochs = 3 # Example: could be configurable
    qat_learning_rate = 1e-4 # Example: could be configurable

    return {
        'dynamic': DynamicQuantizer(device),
        'static': StaticQuantizer(device, backend=backend),
        'qat': QATQuantizer(device, backend=backend, train_epochs=qat_train_epochs, learning_rate=qat_learning_rate),
        'fx': FXQuantizer(device, backend=backend),
        'int8': INT8Quantizer(device) # INT8Quantizer might also need backend or specific qconfigs
    }
