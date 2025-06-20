"""
Quantizable MobileNet implementation with QuantStub and DeQuantStub
Based on the PyTorch MobileNet implementation with modifications for quantization
"""

import torch
from torch import Tensor
import torch.nn as nn
from typing import Any, Callable, List, Optional
from functools import partial
from torch.quantization import QuantStub, DeQuantStub

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


model_urls = {
    'mobilenet_v3_small': 'https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth',
    'mobilenet_v3_large': 'https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth',
}


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


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation layer with quantization-friendly implementation"""
    
    def __init__(
        self,
        input_channels: int,
        squeeze_factor: int = 4,
        activation: Callable[..., nn.Module] = nn.ReLU,
        scale_activation: Callable[..., nn.Module] = nn.Hardsigmoid,
    ):
        super().__init__()
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.activation = activation()
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.scale_activation = scale_activation()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # Use FloatFunctional for quantization-friendly multiplication
        self.mul = nn.quantized.FloatFunctional()

    def _scale(self, input: Tensor, inplace: bool) -> Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        # Use FloatFunctional for quantization-friendly multiplication
        return self.mul.mul(input, scale)

    def forward(self, input: Tensor) -> Tensor:
        return self._scale(input, False)


class InvertedResidualConfig:
    """Configuration for InvertedResidual blocks"""
    
    def __init__(
        self,
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        activation: str,
        stride: int,
        dilation: int,
        width_mult: float,
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual(nn.Module):
    """Inverted residual block with quantization-friendly implementation"""
    
    def __init__(
        self,
        cnf: InvertedResidualConfig,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = SqueezeExcitation,
    ):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                nn.Conv2d(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    kernel_size=1,
                    bias=False,
                )
            )
            layers.append(norm_layer(cnf.expanded_channels))
            layers.append(activation_layer(inplace=True))

        # depthwise
        layers.append(
            nn.Conv2d(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=cnf.expanded_channels,
                bias=False,
                dilation=cnf.dilation,
                padding=(cnf.kernel - 1) // 2 * cnf.dilation,
            )
        )
        layers.append(norm_layer(cnf.expanded_channels))

        if cnf.use_se:
            squeeze_channels = _make_divisible(cnf.expanded_channels // 4, 8)
            layers.append(se_layer(cnf.expanded_channels, squeeze_channels))

        layers.append(activation_layer(inplace=True))

        # project
        layers.append(
            nn.Conv2d(
                cnf.expanded_channels,
                cnf.out_channels,
                kernel_size=1,
                bias=False,
            )
        )
        layers.append(norm_layer(cnf.out_channels))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

        # Use FloatFunctional for quantization-friendly addition
        if self.use_res_connect:
            self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.skip_add.add(result, input)
        return result


class MobileNetV3(nn.Module):
    """MobileNet V3 with quantization stubs"""
    
    def __init__(
        self,
        inverted_residual_setting: List[InvertedResidualConfig],
        last_channel: int,
        num_classes: int = 1000,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, List)
            and all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            nn.Conv2d(
                3,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )
        )
        layers.append(norm_layer(firstconv_output_channels))
        layers.append(nn.Hardswish(inplace=True))

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(
            nn.Conv2d(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                bias=False,
            )
        )
        layers.append(norm_layer(lastconv_output_channels))
        layers.append(nn.Hardswish(inplace=True))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channels, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

        # Add quantization stubs
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

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

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.quant(x)  # add quant
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.dequant(x)  # add dequant
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _mobilenet_v3_conf(
    arch: str,
    width_mult: float = 1.0,
    reduced_tail: bool = False,
    dilated: bool = False,
    **kwargs: Any,
) -> List[InvertedResidualConfig]:
    """Configuration for MobileNet V3"""
    
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    if arch == "mobilenet_v3_large":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
            bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # C1
            bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # C2
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # C3
            bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
            bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
            bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1280 // reduce_divider)  # C5
    elif arch == "mobilenet_v3_small":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
            bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1024 // reduce_divider)  # C5
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel


def _mobilenet_v3(
    arch: str,
    inverted_residual_setting: List[InvertedResidualConfig],
    last_channel: int,
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> MobileNetV3:
    model = MobileNetV3(inverted_residual_setting, last_channel, **kwargs)
    if pretrained:
        if arch in model_urls:
            state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
            # Remove the quant and dequant from state dict if present
            # since they weren't in the original model
            # Also filter out classifier layer if num_classes doesn't match
            state_dict_filtered = {}
            for key, value in state_dict.items():
                if not (key.startswith('quant') or key.startswith('dequant')):
                    # Skip classifier layer if the number of classes doesn't match
                    if key.startswith('classifier.') and kwargs.get('num_classes', 1000) != 1000:
                        continue
                    state_dict_filtered[key] = value
            model.load_state_dict(state_dict_filtered)
    return model


def mobilenet_v3_small_quantizable(
    pretrained: bool = False, 
    progress: bool = True, 
    **kwargs: Any
) -> MobileNetV3:
    """MobileNet V3 Small model from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.
    
    This version is modified for quantization with QuantStub/DeQuantStub and FloatFunctional.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    from functools import partial
    arch = "mobilenet_v3_small"
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch, **kwargs)
    return _mobilenet_v3(arch, inverted_residual_setting, last_channel, pretrained, progress, **kwargs)


def mobilenet_v3_large_quantizable(
    pretrained: bool = False, 
    progress: bool = True, 
    **kwargs: Any
) -> MobileNetV3:
    """MobileNet V3 Large model from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.
    
    This version is modified for quantization with QuantStub/DeQuantStub and FloatFunctional.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    from functools import partial
    arch = "mobilenet_v3_large"
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch, **kwargs)
    return _mobilenet_v3(arch, inverted_residual_setting, last_channel, pretrained, progress, **kwargs)
