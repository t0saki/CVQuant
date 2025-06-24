"""
Quantization methods implementation for PyTorch models
"""
import os
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
from typing import Dict, List, Any, Tuple
import tempfile
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm


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
            print(
                f"Warning: Device conversion failed: {e}, using CPU inference")
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

        model_device = self.device
        model = model.to(model_device)  # Ensure model is on the correct device

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(
                    model_device), targets.to(model_device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)  # Accumulate total loss correctly

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = 100. * correct / total if total > 0 else 0.0
        # Calculate average loss correctly
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
        model_device = self.device
        model = model.to(model_device)
        input_tensor = input_tensor.to(model_device)

        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(input_tensor)

        times = []
        with torch.no_grad():
            for _ in range(benchmark_iterations):
                if model_device.type == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.time()

                _ = model(input_tensor)

                if model_device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)

        if not times:
            return {'mean_time_ms': 0.0, 'min_time_ms': 0.0, 'max_time_ms': 0.0, 'std_time_ms': 0.0}

        mean_time = sum(times) / len(times)
        std_time = (sum([(t - mean_time)**2 for t in times]) /
                    len(times))**0.5 if len(times) > 1 else 0.0
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
        self.original_model = copy.deepcopy(model)
        model_copy = copy.deepcopy(model)
        model_copy = model_copy.to('cpu')
        quantized_model = torch.quantization.quantize_dynamic(
            model_copy, self.qconfig_spec, dtype=torch.qint8
        )
        self.quantized_model = quantized_model
        return quantized_model

    def evaluate_model(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        model = model.to('cpu')

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to('cpu'), targets.to('cpu')
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = 100. * correct / total
        avg_loss = total_loss / len(data_loader)
        return {'accuracy': accuracy, 'loss': avg_loss, 'correct': correct, 'total': total}

    def measure_inference_time(self, model: nn.Module, input_tensor: torch.Tensor,
                               warmup_iterations: int = 50, benchmark_iterations: int = 500) -> Dict[str, float]:
        model.eval()
        model = model.to('cpu')
        input_tensor = input_tensor.to('cpu')

        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(input_tensor)

        times = []
        with torch.no_grad():
            for _ in range(benchmark_iterations):
                start_time = time.time()
                _ = model(input_tensor)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)

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
            self.backend = 'fbgemm' if device and device.type == 'cuda' else 'qnnpack'
        self._setup_ao_config()

    def _setup_ao_config(self):
        self.qconfig_mapping = get_default_qconfig_mapping(self.backend)

    def quantize(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader = None) -> nn.Module:
        self.original_model = copy.deepcopy(model)
        model_copy = copy.deepcopy(model)
        try:
            return self._quantize_with_ao_fx(model_copy, calibration_data)
        except Exception as e:
            print(
                f"Warning: torch.ao FX quantization failed: {e}\nFalling back to legacy quantization")
            return self._quantize_legacy(model_copy, calibration_data)

    def _quantize_with_ao_fx(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader) -> nn.Module:
        model = model.to('cpu')
        model.eval()
        example_inputs = self._get_example_inputs(calibration_data)
        prepared_model = prepare_fx(
            model, self.qconfig_mapping, example_inputs)
        self._calibrate_model(prepared_model, calibration_data)
        quantized_model = convert_fx(prepared_model)
        self.quantized_model = self._get_device_compatible_model(
            quantized_model)
        return self.quantized_model

    def _get_example_inputs(self, calibration_data: torch.utils.data.DataLoader) -> tuple:
        for inputs, _ in calibration_data:
            return (inputs[:1].cpu(),)
        raise ValueError("Calibration data is empty")

    def _calibrate_model(self, prepared_model: nn.Module, calibration_data: torch.utils.data.DataLoader):
        prepared_model.eval()
        with torch.no_grad():
            for i, (inputs, _) in enumerate(calibration_data):
                prepared_model(inputs.cpu())
                if i >= 100:
                    break

    def _quantize_legacy(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader) -> nn.Module:
        if calibration_data is None:
            raise ValueError(
                "Calibration data is required for static quantization")
        model.eval()
        model = model.to('cpu')
        model.qconfig = torch.quantization.get_default_qconfig(self.backend)
        model_prepared = torch.quantization.prepare(model)
        self._calibrate(model_prepared, calibration_data)
        quantized_model = torch.quantization.convert(model_prepared)
        self.quantized_model = quantized_model
        return quantized_model

    def _calibrate(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader):
        model.eval()
        with torch.no_grad():
            for inputs, _ in calibration_data:
                model(inputs.to('cpu'))

    def evaluate_model(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        model.eval()
        correct, total, total_loss = 0, 0, 0.0
        criterion = nn.CrossEntropyLoss().to(self.device)
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)
                outputs = model(inputs)
                if outputs.device != targets.device:
                    outputs = outputs.to(targets.device)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        accuracy = 100. * correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0
        return {'accuracy': accuracy, 'loss': avg_loss, 'correct': correct, 'total': total}


class QATQuantizer(BaseQuantizer):
    """Quantization Aware Training implementation with torch.ao support and optional knowledge distillation."""

    def __init__(self, device: torch.device = None, backend: str = None, train_epochs: int = 10, learning_rate: float = 5e-6, **kwargs):
        super().__init__(device, **kwargs)
        self.backend = backend or (
            'fbgemm' if device and device.type == 'cuda' else 'qnnpack')
        self.train_epochs = train_epochs
        self.learning_rate = learning_rate
        self._setup_ao_qat_config()

    def _setup_ao_qat_config(self):
        """Sets up torch.ao QAT configuration."""
        self.qconfig_mapping = get_default_qat_qconfig_mapping(self.backend)

    def quantize(self, model: nn.Module, calibration_data: DataLoader = None,
                 teacher_model: nn.Module = None, distillation_params: Dict[str, Any] = None) -> nn.Module:
        """
        Apply QAT, with optional knowledge distillation.
        """
        self.original_model = copy.deepcopy(model)
        model_copy = copy.deepcopy(model)
        try:
            return self._quantize_with_ao_qat_fx(model_copy, calibration_data, teacher_model, distillation_params)
        except Exception as e:
            print(f"Warning: torch.ao QAT FX failed: {e}",
                  "\nFalling back to legacy QAT (distillation not supported in fallback)")
            traceback.print_exc()
            return self._quantize_legacy_qat(model_copy, calibration_data)

    def _quantize_with_ao_qat_fx(self, model: nn.Module, training_data: DataLoader,
                                 teacher_model: nn.Module = None, distillation_params: Dict[str, Any] = None) -> nn.Module:
        """Uses torch.ao FX graph mode for QAT, with optional distillation."""
        model = model.to(self.device)
        model.train()

        example_inputs = self._get_example_inputs(training_data)
        prepared_model = prepare_qat_fx(
            model, self.qconfig_mapping, example_inputs)

        trained_model = self._perform_qat_training(
            model=prepared_model,
            training_data=training_data,
            epochs=self.train_epochs,
            teacher_model=teacher_model,
            distillation_params=distillation_params
        )

        trained_model.eval().cpu()
        quantized_model = convert_fx(trained_model)

        self.quantized_model = self._get_device_compatible_model(
            quantized_model)
        return self.quantized_model

    def _perform_qat_training(self, model: nn.Module, training_data: DataLoader,
                              epochs: int, teacher_model: nn.Module = None,
                              distillation_params: Dict[str, Any] = None) -> nn.Module:
        """Performs QAT fine-tuning, using distillation if a teacher model is provided."""
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        is_distillation = teacher_model is not None and distillation_params is not None

        if is_distillation:
            print(
                f"Starting QAT with Knowledge Distillation for {epochs} epochs on {self.device}")
            teacher_model.to(self.device).eval()
        else:
            print(
                f"Starting standard QAT training for {epochs} epochs on {self.device}")

        for epoch in range(epochs):
            if is_distillation:
                avg_loss, _, _ = self._distillation_qat_train_epoch(
                    teacher_model, model, training_data, optimizer, distillation_params
                )
                print(
                    f"Distill-QAT Epoch {epoch+1}/{epochs}, Avg Combined Loss: {avg_loss:.4f}")
            else:
                avg_loss = self._standard_qat_train_epoch(
                    model, training_data, optimizer)
                print(
                    f"Standard QAT Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

        return model

    def _standard_qat_train_epoch(self, model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer) -> float:
        """Train for one standard QAT epoch."""
        model.train()
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss().to(self.device)
        progress_bar = tqdm(train_loader, desc="Standard QAT", leave=False)

        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        return total_loss / len(train_loader)

    def _distillation_qat_train_epoch(self, teacher_model: nn.Module, student_model: nn.Module,
                                      train_loader: DataLoader, optimizer: optim.Optimizer,
                                      distillation_params: Dict[str, Any]) -> Tuple[float, float, float]:
        """Train for one epoch with knowledge distillation during QAT."""
        student_model.train()
        criterion_ce = nn.CrossEntropyLoss()
        criterion_kd = nn.KLDivLoss(reduction='batchmean')
        alpha = distillation_params.get('alpha', 0.7)
        temperature = distillation_params.get('temperature', 4.0)
        total_loss, total_distill_loss, total_ce_loss = 0.0, 0.0, 0.0
        progress_bar = tqdm(train_loader, desc="Distill-QAT", leave=False)

        for data, target in progress_bar:
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_outputs = teacher_model(data)

            student_outputs = student_model(data)
            ce_loss = criterion_ce(student_outputs, target)
            distill_loss = criterion_kd(
                F.log_softmax(student_outputs / temperature, dim=1),
                F.softmax(teacher_outputs / temperature, dim=1)
            ) * (temperature ** 2)

            loss = alpha * distill_loss + (1 - alpha) * ce_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_distill_loss += distill_loss.item()
            total_ce_loss += ce_loss.item()
            progress_bar.set_postfix(
                {'Loss': f'{loss.item():.4f}', 'KD': f'{distill_loss.item():.4f}', 'CE': f'{ce_loss.item():.4f}'})

        return (total_loss / len(train_loader),
                total_distill_loss / len(train_loader),
                total_ce_loss / len(train_loader))

    def _get_example_inputs(self, data_loader: DataLoader) -> tuple:
        for inputs, _ in data_loader:
            return (inputs[:1].to(self.device),)
        raise ValueError("Training data is empty")

    def _quantize_legacy_qat(self, model: nn.Module, training_data: DataLoader) -> nn.Module:
        if training_data is None:
            raise ValueError("Training data is required for QAT")
        model.train()
        model.qconfig = torch.quantization.get_default_qat_qconfig(
            self.backend)
        torch.quantization.prepare_qat(model, inplace=True)
        self._perform_qat_training(model, training_data, self.train_epochs)
        model.eval()
        quantized_model = torch.quantization.convert(model)
        self.quantized_model = quantized_model
        return quantized_model


class FXQuantizer(BaseQuantizer):
    """FX Graph Mode quantization implementation"""

    def __init__(self, device: torch.device = None, backend: str = 'fbgemm', **kwargs):
        super().__init__(device, **kwargs)
        self.backend = backend
        torch.backends.quantized.engine = backend

    def quantize(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader) -> nn.Module:
        if calibration_data is None:
            raise ValueError(
                "Calibration data is required for FX quantization")
        self.original_model = copy.deepcopy(model)
        model_copy = copy.deepcopy(model).eval()
        qconfig_mapping = get_default_qconfig_mapping(self.backend)
        example_inputs = next(iter(calibration_data))[0][:1].to(self.device)
        model_prepared = prepare_fx(
            model_copy, qconfig_mapping, example_inputs)
        self._calibrate(model_prepared, calibration_data)
        quantized_model = convert_fx(model_prepared)
        self.quantized_model = quantized_model
        return quantized_model

    def _calibrate(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader):
        model.eval()
        with torch.no_grad():
            for inputs, _ in calibration_data:
                model(inputs.to(self.device))


class INT8Quantizer(BaseQuantizer):
    """INT8 quantization using custom observers"""

    def quantize(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader) -> nn.Module:
        if calibration_data is None:
            raise ValueError(
                "Calibration data is required for INT8 quantization")
        self.original_model = copy.deepcopy(model)
        model_copy = copy.deepcopy(model).eval()
        custom_qconfig = QConfig(
            activation=default_observer.with_args(dtype=torch.qint8),
            weight=default_weight_observer.with_args(dtype=torch.qint8)
        )
        model_copy.qconfig = custom_qconfig
        model_prepared = torch.quantization.prepare(model_copy)
        self._calibrate(model_prepared, calibration_data)
        quantized_model = torch.quantization.convert(model_prepared)
        self.quantized_model = quantized_model
        return quantized_model

    def _calibrate(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader):
        model.eval()
        with torch.no_grad():
            for inputs, _ in calibration_data:
                model(inputs.to(self.device))


class QuantizationBenchmark(BaseQuantizer):
    """Benchmark multiple quantization methods"""

    def __init__(self, device: torch.device = None, backend: str = 'fbgemm', **kwargs):
        super().__init__(device, **kwargs)
        self.device = device or torch.device('cpu')
        self.backend = backend
        self.results = {}
        self.quantizers = {}
        self.model_name = kwargs.get('model_name', 'unknown')

    def _load_original_resnet_for_lrf(self, lrf_model: nn.Module) -> nn.Module:
        from models.model_loader import ModelLoader
        original_model_name = 'resnet18_quantizable' if 'resnet18' in self.model_name.lower(
        ) else 'resnet50_quantizable'
        num_classes = lrf_model.fc.out_features if hasattr(
            lrf_model, 'fc') and hasattr(lrf_model.fc, 'out_features') else 1000
        try:
            model_loader = ModelLoader(
                num_classes=num_classes, device=self.device, enable_finetuning=False)
            original_model = model_loader.load_model(
                original_model_name, pretrained=True).to(self.device).eval()
            print(
                f"Successfully loaded {original_model_name} as baseline for LRF comparison")
            return original_model
        except Exception as e:
            print(
                f"Error loading original ResNet {original_model_name}: {e}\nUsing LRF model as baseline instead")
            return lrf_model

    def benchmark_quantization_methods(self, model: nn.Module,
                                       calibration_data: DataLoader,
                                       evaluation_data: DataLoader,
                                       methods: list = None,
                                       teacher_model: nn.Module = None,
                                       args: Any = None) -> Dict[str, Dict[str, Any]]:
        if methods is None:
            methods = ['dynamic', 'static', 'qat']
        self.results = {}
        example_input = next(iter(calibration_data))[0][:1]

        is_lrf_model = 'low_rank' in self.model_name
        student_model = model  # The model passed in is our main subject

        # --- REVISED LOGIC START ---

        # 1. Establish the ultimate baseline: the original, non-factorized model.
        # This gives a consistent reference for all speedup/compression calculations.
        if is_lrf_model:
            print("Loading original (non-LRF) model for baseline comparison...")
            original_model_baseline = self._load_original_resnet_for_lrf(
                student_model)
        else:
            # If the student is not an LRF model, it IS the baseline.
            original_model_baseline = student_model

        print("Evaluating baseline (unquantized) model...")
        original_eval = self.evaluate_model(
            original_model_baseline, evaluation_data)
        original_time = self.measure_inference_time(
            original_model_baseline, example_input)
        original_size = self._get_model_size(original_model_baseline)

        # Save baseline stats for later calculations
        self.results['original'] = {
            'accuracy': original_eval['accuracy'],
            'loss': original_eval['loss'],
            'model_size_mb': original_size,
            'speedup': 1.0,
            'compression_ratio': 1.0,
            **original_time
        }
        original_time_ms = self.results['original']['mean_time_ms']
        original_size_mb = self.results['original']['model_size_mb']

        # 2. If the student is an LRF model, evaluate it as its own separate entry.
        # This will now correctly measure its reduced size.
        if is_lrf_model:
            print("\nEvaluating LRF (unquantized) student model...")
            lrf_eval = self.evaluate_model(student_model, evaluation_data)
            lrf_time = self.measure_inference_time(
                student_model, example_input)
            lrf_size = self._get_model_size(student_model)
            speedup = original_time_ms / \
                lrf_time['mean_time_ms'] if lrf_time['mean_time_ms'] > 0 else 0
            compression = original_size_mb / lrf_size if lrf_size > 0 else 0

            self.results['lrf'] = {
                'accuracy': lrf_eval['accuracy'],
                'loss': lrf_eval['loss'],
                'model_size_mb': lrf_size,
                'speedup': speedup,
                'compression_ratio': compression,
                **lrf_time
            }

        # 3. Proceed with quantization experiments on the student model.
        self.quantizers = get_available_quantizers(
            backend=self.backend, device=self.device, model_name=self.model_name)

        model_to_quantize = student_model

        for method_name in methods:
            quantizer = self.quantizers[method_name]
            result_key = f"lrf_{method_name}" if is_lrf_model else method_name
            print(f"\nBenchmarking {result_key} quantization...")

            try:
                if method_name == 'qat' and args and args.enable_distillation and teacher_model:
                    print("-> Activating QAT with Knowledge Distillation.")
                    from utils.distillation import KnowledgeDistiller, auto_select_teacher_model
                    from models.model_loader import get_available_models
                    teacher_name = args.teacher_model or auto_select_teacher_model(
                        self.model_name, get_available_models())
                    dist_config = KnowledgeDistiller().get_distillation_config(
                        teacher_name, self.model_name, args.dataset)
                    print(f"-> Using distillation config: {dist_config}")
                    quantized_model = quantizer.quantize(copy.deepcopy(
                        model_to_quantize), calibration_data, teacher_model, dist_config)
                else:
                    quantized_model = quantizer.quantize(
                        copy.deepcopy(model_to_quantize), calibration_data)

                eval_metrics = quantizer.evaluate_model(
                    quantized_model, evaluation_data)
                time_stats = quantizer.measure_inference_time(
                    quantized_model, example_input)
                model_size_mb = self._get_model_size(quantized_model)

                # Always calculate speedup and compression against the true original model
                speedup = original_time_ms / \
                    time_stats['mean_time_ms'] if time_stats['mean_time_ms'] > 0 else 0
                compression = original_size_mb / model_size_mb if model_size_mb > 0 else 0

                self.results[result_key] = {
                    'accuracy': eval_metrics['accuracy'],
                    'loss': eval_metrics['loss'],
                    'model_size_mb': model_size_mb,
                    'speedup': speedup,
                    'compression_ratio': compression,
                    **time_stats
                }

            except Exception as e:
                print(f"Error benchmarking {method_name}: {e}")
                traceback.print_exc()
                self.results[result_key] = {'error': str(e)}

        return self.results

    def _get_model_size(self, model: nn.Module) -> float:
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            torch.save(model.state_dict(), temp_file.name)
            return os.path.getsize(temp_file.name) / (1024 * 1024)

    def print_results(self):
        if not self.results:
            print("No results to display.")
            return
        print("\n" + "="*80)
        print("QUANTIZATION BENCHMARK RESULTS")
        print("="*80)
        print(f"{'Method':<15} {'Accuracy (%)':<15} {'Time (ms)':<15} {'Size (MB)':<15} {'Speedup':<10} {'Compression':<12}")
        print("-"*80)

        for method, res in self.results.items():
            if 'error' in res:
                print(
                    f"{method:<15} {'ERROR':<15} {'-':<15} {'-':<15} {'-':<10} {'-':<12}")
                continue
            acc = f"{res.get('accuracy', 0.0):.2f}"
            time_ms = f"{res.get('mean_time_ms', 0.0):.2f}"
            size_mb = f"{res.get('model_size_mb', 0.0):.2f}"
            speedup = f"{res.get('speedup', 1.0):.2f}x"
            comp = f"{res.get('compression_ratio', 1.0):.2f}x"
            print(
                f"{method:<15} {acc:<15} {time_ms:<15} {size_mb:<15} {speedup:<10} {comp:<12}")
        print("="*80)


class OfficialQuantizedQuantizer(BaseQuantizer):
    """Official pre-trained quantized models from torchvision.models.quantization"""

    def __init__(self, device: torch.device = None, **kwargs):
        super().__init__(device, **kwargs)
        self.device = torch.device('cpu')

    def quantize(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader = None) -> nn.Module:
        model_name = self._get_model_name(model)
        quantized_model = self._load_official_quantized_model(model_name)
        if quantized_model is None:
            raise ValueError(
                f"No official quantized model available for {model_name}")
        self.quantized_model = quantized_model.to('cpu').eval()
        return self.quantized_model

    def _get_model_name(self, model: nn.Module) -> str:
        return self.model_name

    def _load_official_quantized_model(self, model_name: str) -> nn.Module:
        from torchvision.models import quantization
        model_map = {
            'resnet18': quantization.resnet18,
            'resnet50': quantization.resnet50,
            'mobilenet_v2': quantization.mobilenet_v2,
            'mobilenet_v3_large': quantization.mobilenet_v3_large,
        }
        if model_name.startswith('mobilenet_v3_small'):
            print(
                "Warning: official mobilenet_v3_small not available, using large instead.")
            model_name = 'mobilenet_v3_large'

        quant_model_fn = next(
            (v for k, v in model_map.items() if model_name.startswith(k)), None)
        if quant_model_fn:
            return quant_model_fn(pretrained=True, quantize=True)
        return None

    def evaluate_model(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        return DynamicQuantizer().evaluate_model(model, data_loader)

    def measure_inference_time(self, model: nn.Module, input_tensor: torch.Tensor,
                               warmup_iterations: int = 50, benchmark_iterations: int = 500) -> Dict[str, float]:
        return DynamicQuantizer().measure_inference_time(model, input_tensor, warmup_iterations, benchmark_iterations)


def get_available_quantizers(backend: str = 'qnnpack', device: torch.device = None, model_name: str = None) -> Dict[str, BaseQuantizer]:
    """Get dictionary of available quantizers"""
    device = device or torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    return {
        'dynamic': DynamicQuantizer(device=device, backend=backend, model_name=model_name),
        'static': StaticQuantizer(device=device, backend=backend, model_name=model_name),
        'qat': QATQuantizer(device=device, backend=backend, train_epochs=10, learning_rate=5e-6, model_name=model_name),
        'fx': FXQuantizer(device=device, backend=backend, model_name=model_name),
        'int8': INT8Quantizer(device=device, backend=backend, model_name=model_name),
        'official': OfficialQuantizedQuantizer(device=device, backend=backend, model_name=model_name),
    }
