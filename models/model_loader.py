"""
Model loader for ResNet and MobileNet models
"""
import torch
import torch.nn as nn
import torchvision.models as models
import timm
from typing import Dict, Any, Optional, Callable, Union
from functools import partial


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class QuantizedSqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block with FloatFunctional for quantization compatibility"""
    def __init__(
        self,
        input_channels: int,
        squeeze_factor: int = 4,
    ):
        super().__init__()
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.sigmoid = nn.Sigmoid()
        # Use FloatFunctional for quantization compatibility
        self.scale_mul = nn.quantized.FloatFunctional()

    def _scale(self, input: torch.Tensor, inplace: bool) -> torch.Tensor:
        scale = self.fc1(input)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)
        # Use FloatFunctional.mul instead of * operator
        return self.scale_mul.mul(input, scale)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        scale = torch.nn.functional.adaptive_avg_pool2d(input, 1)
        return self._scale(input, False)


class QuantizedInvertedResidual(nn.Module):
    """InvertedResidual block with FloatFunctional for quantization compatibility"""
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        use_se: bool = False,
        use_hs: bool = False,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            layers.append(norm_layer(hidden_dim))
            layers.append(nn.Hardswish() if use_hs else nn.ReLU(inplace=True))

        layers.extend([
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            norm_layer(hidden_dim),
            # Squeeze-and-Excite
            QuantizedSqueezeExcitation(hidden_dim) if use_se else nn.Identity(),
            nn.Hardswish() if use_hs else nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])

        self.conv = nn.Sequential(*layers)
        
        # Use FloatFunctional for quantization compatibility
        if self.use_res_connect:
            self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            # Use FloatFunctional.add instead of + operator for skip connections
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)


class QuantizedMobileNetV3(nn.Module):
    """MobileNetV3 with quantization-compatible blocks"""
    def __init__(
        self,
        inverted_residual_setting: list,
        last_channel: int,
        num_classes: int = 1000,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
    ):
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (isinstance(inverted_residual_setting, list) and 
                  all(isinstance(s, list) for s in inverted_residual_setting)):
            raise TypeError("The inverted_residual_setting should be a list of lists")

        if block is None:
            block = QuantizedInvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0][0]
        layers.append(nn.Conv2d(3, firstconv_output_channels, 3, 2, 1, bias=False))
        layers.append(norm_layer(firstconv_output_channels))
        layers.append(nn.Hardswish())

        # building inverted residual blocks
        for inp, exp, out, k, s, use_se, use_hs in inverted_residual_setting:
            layers.append(block(inp, out, s, exp, use_se, use_hs, norm_layer))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1][2]
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(nn.Conv2d(lastconv_input_channels, lastconv_output_channels, 1, 1, 0, bias=False))
        layers.append(norm_layer(lastconv_output_channels))
        layers.append(nn.Hardswish())

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channels, last_channel),
            nn.Hardswish(),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)


def quantized_mobilenet_v3_large(num_classes: int = 1000, pretrained: bool = False) -> QuantizedMobileNetV3:
    """Quantization-compatible MobileNetV3-Large"""
    # MobileNetV3-Large configuration
    # [inp, exp, out, k, s, use_se, use_hs]
    inverted_residual_setting = [
        [16, 1, 16, 3, 1, False, False],
        [16, 4, 24, 3, 2, False, False],
        [24, 3, 24, 3, 1, False, False],
        [24, 3, 40, 5, 2, True, False],
        [40, 3, 40, 5, 1, True, False],
        [40, 3, 40, 5, 1, True, False],
        [40, 6, 80, 3, 2, False, True],
        [80, 2.5, 80, 3, 1, False, True],
        [80, 2.3, 80, 3, 1, False, True],
        [80, 2.3, 80, 3, 1, False, True],
        [80, 6, 112, 3, 1, True, True],
        [112, 6, 112, 3, 1, True, True],
        [112, 6, 160, 5, 2, True, True],
        [160, 6, 160, 5, 1, True, True],
        [160, 6, 160, 5, 1, True, True],
    ]
    last_channel = 1280

    model = QuantizedMobileNetV3(inverted_residual_setting, last_channel, num_classes)

    if pretrained:
        # Load pretrained weights from standard MobileNetV3-Large and transfer them
        try:
            pretrained_model = models.mobilenet_v3_large(pretrained=True)
            model_dict = model.state_dict()
            pretrained_dict = pretrained_model.state_dict()
            
            # Filter out keys that don't match (due to different block structure)
            filtered_dict = {}
            for k, v in pretrained_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    filtered_dict[k] = v
            
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)
            print(f"Loaded {len(filtered_dict)} layers from pretrained MobileNetV3-Large")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
    
    return model


def quantized_mobilenet_v3_small(num_classes: int = 1000, pretrained: bool = False) -> QuantizedMobileNetV3:
    """Quantization-compatible MobileNetV3-Small"""
    # MobileNetV3-Small configuration
    # [inp, exp, out, k, s, use_se, use_hs]
    inverted_residual_setting = [
        [16, 1, 16, 3, 2, True, False],
        [16, 4.5, 24, 3, 2, False, False],
        [24, 3.67, 24, 3, 1, False, False],
        [24, 4, 40, 5, 2, True, True],
        [40, 6, 40, 5, 1, True, True],
        [40, 6, 40, 5, 1, True, True],
        [40, 3, 48, 5, 1, True, True],
        [48, 3, 48, 5, 1, True, True],
        [48, 6, 96, 5, 2, True, True],
        [96, 6, 96, 5, 1, True, True],
        [96, 6, 96, 5, 1, True, True],
    ]
    last_channel = 1024

    model = QuantizedMobileNetV3(inverted_residual_setting, last_channel, num_classes)

    if pretrained:
        # Load pretrained weights from standard MobileNetV3-Small and transfer them
        try:
            pretrained_model = models.mobilenet_v3_small(pretrained=True)
            model_dict = model.state_dict()
            pretrained_dict = pretrained_model.state_dict()
            
            # Filter out keys that don't match (due to different block structure)
            filtered_dict = {}
            for k, v in pretrained_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    filtered_dict[k] = v
            
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)
            print(f"Loaded {len(filtered_dict)} layers from pretrained MobileNetV3-Small")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
    
    return model


class QuantizedBasicBlock(nn.Module):
    """BasicBlock with FloatFunctional for quantization compatibility"""
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        # Use FloatFunctional for quantization compatibility
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Use FloatFunctional.add instead of += operator
        out = self.skip_add.add(out, identity)
        out = self.relu(out)

        return out


class QuantizedBottleneck(nn.Module):
    """Bottleneck block with FloatFunctional for quantization compatibility"""
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        # Use FloatFunctional for quantization compatibility
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Use FloatFunctional.add instead of += operator
        out = self.skip_add.add(out, identity)
        out = self.relu(out)

        return out


class QuantizedResNet(nn.Module):
    """ResNet with quantization-compatible blocks"""

    def __init__(
        self,
        block: type,
        layers: list,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[list] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, QuantizedBottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, QuantizedBasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: type,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)


def quantized_resnet18(num_classes: int = 1000, pretrained: bool = False) -> QuantizedResNet:
    """Quantization-compatible ResNet-18"""
    model = QuantizedResNet(QuantizedBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    
    if pretrained:
        # Load pretrained weights from standard ResNet18 and transfer them
        pretrained_model = models.resnet18(pretrained=True)
        model_dict = model.state_dict()
        pretrained_dict = pretrained_model.state_dict()
        
        # Filter out keys that don't match (due to different block structure)
        filtered_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                filtered_dict[k] = v
        
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        print(f"Loaded {len(filtered_dict)} layers from pretrained ResNet18")
    
    return model


def quantized_resnet50(num_classes: int = 1000, pretrained: bool = False) -> QuantizedResNet:
    """Quantization-compatible ResNet-50"""
    model = QuantizedResNet(QuantizedBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    
    if pretrained:
        # Load pretrained weights from standard ResNet50 and transfer them
        pretrained_model = models.resnet50(pretrained=True)
        model_dict = model.state_dict()
        pretrained_dict = pretrained_model.state_dict()
        
        # Filter out keys that don't match
        filtered_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                filtered_dict[k] = v
        
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        print(f"Loaded {len(filtered_dict)} layers from pretrained ResNet50")
    
    return model


class ModelLoader:
    """Load and prepare models for quantization experiments"""
    
    def __init__(self, num_classes: int = 1000):
        self.num_classes = num_classes
        self.available_models = {
            'resnet18': self._load_resnet18,
            'resnet50': self._load_resnet50,
            'mobilenet_v3_small': self._load_mobilenet_v3_small,
            'mobilenet_v3_large': self._load_mobilenet_v3_large,
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
        """Load quantization-compatible ResNet-18 model"""
        model = quantized_resnet18(num_classes=self.num_classes, pretrained=pretrained)
        return model
    
    def _load_resnet50(self, pretrained: bool = True) -> nn.Module:
        """Load quantization-compatible ResNet-50 model"""
        model = quantized_resnet50(num_classes=self.num_classes, pretrained=pretrained)
        return model
    
    def _load_mobilenet_v2(self, pretrained: bool = True) -> nn.Module:
        """Load MobileNet V2 model"""
        model = models.mobilenet_v2(pretrained=pretrained)
        if self.num_classes != 1000:
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
        return model
    
    def _load_mobilenet_v3_large(self, pretrained: bool = True) -> nn.Module:
        """Load quantization-compatible MobileNet V3 Large model"""
        model = quantized_mobilenet_v3_large(num_classes=self.num_classes, pretrained=pretrained)
        return model
    
    def _load_mobilenet_v3_small(self, pretrained: bool = True) -> nn.Module:
        """Load quantization-compatible MobileNet V3 Small model"""
        model = quantized_mobilenet_v3_small(num_classes=self.num_classes, pretrained=pretrained)
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