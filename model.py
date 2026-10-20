"""Model definitions for the Potato vs Onion image classifier.

This module provides two interchangeable architectures:

* :class:`PotatoOnionCNN` - a compact custom CNN trained from scratch, with
  BatchNorm after every convolution and Dropout before the linear layers.
* :class:`PotatoOnionMobileNet` - a transfer-learning model built on a
  pre-trained MobileNetV2 backbone with a new 3-class classifier head.

Use :func:`build_model` to create either one by name.

Classes (index order must match your dataset's ``class_to_idx``):
    0: both, 1: onions, 2: potatoes

Example:
    >>> model = build_model("mobilenet_v2", pretrained=True)
    >>> logits = model(torch.randn(8, 3, 224, 224))
    >>> logits.shape
    torch.Size([8, 3])
"""

from __future__ import annotations

from typing import Literal, Sequence

import torch
import torch.nn as nn
from torchvision import models

CLASS_NAMES: tuple[str, ...] = ("both", "onions", "potatoes")
NUM_CLASSES: int = len(CLASS_NAMES)

ModelName = Literal["cnn", "mobilenet_v2"]


# --------------------------------------------------------------------------- #
# Option 1: custom CNN trained from scratch
# --------------------------------------------------------------------------- #
class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d building block.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size. Padding is chosen to preserve
            spatial size before pooling.
        pool: Whether to apply 2x2 max pooling at the end of the block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            # bias=False because BatchNorm already has a learnable shift.
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block to a batch of feature maps."""
        return self.block(x)


class PotatoOnionCNN(nn.Module):
    """Custom CNN with BatchNorm and Dropout for 3-class classification.

    Architecture::

        [ConvBlock x N] -> AdaptiveAvgPool2d(1) -> Flatten
            -> Dropout -> Linear -> ReLU -> Dropout -> Linear

    Global adaptive pooling makes the network independent of the input
    resolution, so no manual calculation of the flattened feature size is
    required.

    Args:
        num_classes: Number of output classes.
        in_channels: Number of input image channels (3 for RGB).
        channels: Output channels of each convolutional block.
        hidden_dim: Width of the hidden fully-connected layer.
        dropout: Dropout probability applied before each linear layer.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        in_channels: int = 3,
        channels: Sequence[int] = (32, 64, 128, 256),
        hidden_dim: int = 256,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        blocks: list[nn.Module] = []
        prev = in_channels
        for out in channels:
            blocks.append(ConvBlock(prev, out))
            prev = out

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(prev, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise conv layers with Kaiming and linear layers with Xavier."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits.

        Args:
            x: Input batch of shape ``(N, C, H, W)``.

        Returns:
            Raw (un-normalised) logits of shape ``(N, num_classes)``. Apply
            ``softmax`` for probabilities; ``nn.CrossEntropyLoss`` expects
            logits directly.
        """
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# --------------------------------------------------------------------------- #
# Option 2: transfer learning with MobileNetV2
# --------------------------------------------------------------------------- #
class PotatoOnionMobileNet(nn.Module):
    """MobileNetV2 transfer-learning model with a new classification head.

    Args:
        num_classes: Number of output classes.
        pretrained: If ``True``, load ImageNet-pretrained weights
            (downloaded on first use).
        freeze_backbone: If ``True``, freeze all feature-extractor weights so
            only the new classifier head is trained. Recommended for the
            first few epochs or for very small datasets.
        dropout: Dropout probability in the classifier head.

    Note:
        Pre-trained weights expect inputs normalised with ImageNet statistics
        (``mean=[0.485, 0.456, 0.406]``, ``std=[0.229, 0.224, 0.225]``).
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.mobilenet_v2(weights=weights)

        in_features: int = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Freeze the feature extractor; only the classifier remains trainable."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze the feature extractor for full fine-tuning."""
        for param in self.backbone.features.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits.

        Args:
            x: Input batch of shape ``(N, 3, H, W)`` (224x224 recommended).

        Returns:
            Raw logits of shape ``(N, num_classes)``.
        """
        return self.backbone(x)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_model(
    name: ModelName = "mobilenet_v2",
    num_classes: int = NUM_CLASSES,
    **kwargs: object,
) -> nn.Module:
    """Create a model by name.

    Args:
        name: ``"cnn"`` for :class:`PotatoOnionCNN` or ``"mobilenet_v2"`` for
            :class:`PotatoOnionMobileNet`.
        num_classes: Number of output classes.
        **kwargs: Extra keyword arguments forwarded to the model constructor
            (e.g. ``pretrained=False``, ``dropout=0.3``).

    Returns:
        The instantiated model.

    Raises:
        ValueError: If ``name`` is not a supported architecture.
    """
    if name == "cnn":
        return PotatoOnionCNN(num_classes=num_classes, **kwargs)  # type: ignore[arg-type]
    if name == "mobilenet_v2":
        return PotatoOnionMobileNet(num_classes=num_classes, **kwargs)  # type: ignore[arg-type]
    raise ValueError(f"Unknown model '{name}'. Choose 'cnn' or 'mobilenet_v2'.")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count the parameters of a model.

    Args:
        model: Any ``nn.Module``.
        trainable_only: If ``True``, count only parameters with
            ``requires_grad=True``.

    Returns:
        The total number of parameters.
    """
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad or not trainable_only
    )


if __name__ == "__main__":
    dummy = torch.randn(2, 3, 224, 224)
    for model_name in ("cnn", "mobilenet_v2"):
        net = build_model(model_name, pretrained=False) if model_name == "mobilenet_v2" else build_model(model_name)
        net.eval()
        with torch.no_grad():
            out = net(dummy)
        print(f"{model_name}: output {tuple(out.shape)}, "
              f"trainable params {count_parameters(net):,}")