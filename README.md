# PyTorch Model Quantization Benchmark

A comprehensive PyTorch project for benchmarking different quantization methods on ResNet and MobileNet V4 models.

## Features

- **Multiple Model Support**: ResNet (18, 50) and MobileNet (V2, V3, V4) models
- **Multiple Quantization Methods**: 
  - Dynamic Quantization
  - Static Quantization (Post-Training Quantization)
  - Quantization Aware Training (QAT)
  - FX Graph Mode Quantization
  - INT8 Quantization with custom observers
- **Comprehensive Benchmarking**: Accuracy, inference time, model size, and compression ratio analysis
- **Visualization**: Automatic generation of comparison plots
- **Flexible Dataset Support**: CIFAR-10, CIFAR-100, and ImageNet (proxy)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd CVQuant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
CVQuant/
├── config.py                 # Configuration settings
├── main.py                   # Main experiment script
├── requirements.txt          # Python dependencies
├── models/
│   ├── __init__.py
│   └── model_loader.py       # Model loading utilities
├── quantization/
│   ├── __init__.py
│   └── quantizers.py         # Quantization method implementations
├── utils/
│   ├── __init__.py
│   └── data_loader.py        # Data loading utilities
├── data/                     # Dataset directory
└── results/                  # Experiment results directory
```

## Quick Start

### Basic Usage

Run quantization experiments on ResNet-18 with CIFAR-10:

```bash
python main.py --model resnet18 --dataset cifar10
```

### Advanced Usage

Run comprehensive experiments with custom settings:

```bash
python main.py \
    --model mobilenet_v4_conv_small \
    --dataset cifar100 \
    --methods dynamic static fx int8 \
    --batch-size 64 \
    --calibration-size 2000 \
    --evaluation-size 10000 \
    --plot-results \
    --output-dir ./results
```

## Available Models

- **ResNet**: `resnet18`, `resnet50`
- **MobileNet V2**: `mobilenet_v2`
- **MobileNet V3**: `mobilenet_v3_large`, `mobilenet_v3_small`
- **MobileNet V4**: `mobilenet_v4_conv_small`, `mobilenet_v4_conv_medium`, `mobilenet_v4_conv_large`

## Available Datasets

- **CIFAR-10**: 10 classes, 32x32 images
- **CIFAR-100**: 100 classes, 32x32 images
- **ImageNet**: 1000 classes (uses CIFAR-10 resized as proxy for demo)

## Quantization Methods

### 1. Dynamic Quantization
- **Description**: Quantizes weights to int8 while keeping activations in float
- **Pros**: Easy to apply, no calibration data needed
- **Cons**: Limited speedup, activations remain in float

### 2. Static Quantization (Post-Training Quantization)
- **Description**: Quantizes both weights and activations to int8
- **Pros**: Better compression and speedup
- **Cons**: Requires calibration data, may have accuracy drop

### 3. Quantization Aware Training (QAT)
- **Description**: Simulates quantization during training
- **Pros**: Best accuracy preservation
- **Cons**: Requires training, more complex

### 4. FX Graph Mode Quantization
- **Description**: Uses PyTorch FX for graph-level quantization
- **Pros**: More flexible, better optimization
- **Cons**: Limited model support

### 5. INT8 Quantization
- **Description**: Custom INT8 quantization with specific observers
- **Pros**: Fine-grained control
- **Cons**: More complex configuration

## Command Line Arguments

| Argument | Description | Default | Choices |
|----------|-------------|---------|---------|
| `--model` | Model architecture | `resnet18` | `resnet18`, `resnet50`, `mobilenet_v2`, `mobilenet_v3_large`, `mobilenet_v3_small`, `mobilenet_v4_conv_small`, `mobilenet_v4_conv_medium`, `mobilenet_v4_conv_large` |
| `--dataset` | Dataset to use | `cifar10` | `cifar10`, `cifar100`, `imagenet` |
| `--methods` | Quantization methods | `['dynamic', 'static', 'fx', 'int8']` | `dynamic`, `static`, `qat`, `fx`, `int8` |
| `--batch-size` | Batch size | `32` | Any positive integer |
| `--calibration-size` | Calibration dataset size | `1000` | Any positive integer |
| `--evaluation-size` | Evaluation dataset size | `5000` | Any positive integer |
| `--plot-results` | Generate result plots | `False` | Flag |
| `--save-models` | Save quantized models | `False` | Flag |
| `--device` | Computation device | `auto` | `auto`, `cpu`, `cuda` |

## Example Results

### ResNet-18 on CIFAR-10

| Method | Accuracy | Inference Time | Model Size | Speedup | Compression |
|--------|----------|----------------|------------|---------|-------------|
| Original | 95.23% | 12.5ms | 44.7MB | 1.00x | 1.00x |
| Dynamic | 95.18% | 11.8ms | 11.2MB | 1.06x | 4.00x |
| Static | 94.87% | 8.9ms | 11.2MB | 1.40x | 4.00x |
| FX | 94.92% | 8.7ms | 11.2MB | 1.44x | 4.00x |
| INT8 | 94.81% | 8.8ms | 11.2MB | 1.42x | 4.00x |

## Output Files

The experiment generates several output files in the results directory:

1. **JSON Results**: `quantization_results_{model}_{dataset}_{timestamp}.json`
   - Contains detailed numerical results
   - Includes experiment configuration
   - Machine-readable format for further analysis

2. **Plots**: `quantization_plots_{model}_{dataset}_{timestamp}.png`
   - Accuracy comparison
   - Inference time comparison
   - Model size comparison
   - Speedup vs Compression scatter plot

3. **Models**: `models/{model}_{dataset}_{method}_{timestamp}.pth` (if `--save-models` is used)
   - Saved quantized model weights
   - Can be loaded for inference

## Performance Tips

1. **GPU Usage**: Use CUDA when available for faster experiments
2. **Batch Size**: Increase batch size for better GPU utilization
3. **Calibration Size**: Larger calibration sets improve static quantization accuracy
4. **Model Selection**: Start with smaller models (ResNet-18, MobileNet V3 Small) for faster iteration

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size: `--batch-size 16`
   - Use CPU: `--device cpu`

2. **MobileNet V4 Not Found**:
   - Ensure `timm` package is installed and up-to-date
   - Fallback to MobileNet V3 will be used automatically

3. **Dataset Download Issues**:
   - Check internet connection
   - Ensure sufficient disk space in `./data` directory

4. **Quantization Errors**:
   - Some models may not support all quantization methods
   - Check error messages for specific method failures

## Extending the Project

### Adding New Models

1. Add model loading logic to `models/model_loader.py`
2. Update `Config.MODELS` in `config.py`
3. Add model name to argument choices in `main.py`

### Adding New Quantization Methods

1. Create new quantizer class in `quantization/quantizers.py`
2. Inherit from `BaseQuantizer`
3. Implement the `quantize` method
4. Add to `get_available_quantizers()` function

### Adding New Datasets

1. Add dataset loading logic to `utils/data_loader.py`
2. Update transform and normalization parameters
3. Add dataset name to argument choices

## Requirements

- Python 3.7+
- PyTorch 2.0+
- torchvision 0.15+
- Additional packages listed in `requirements.txt`

## License

This project is released under the MIT License. See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{pytorch-quantization-benchmark,
  title={PyTorch Model Quantization Benchmark},
  author={t0saki},
  year={2025},
  url={https://github.com/t0saki/CVQuant}
}
```

## Acknowledgments

- PyTorch team for quantization APIs
- timm library for MobileNet V4 models
- torchvision for pretrained models and datasets