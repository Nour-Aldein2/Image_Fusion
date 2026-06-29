# config.py

from dataclasses import dataclass, field


@dataclass
class ResidualBlockConfig:
    feature_dim: int = 256

    stride: int = 1
    padding: int = 1
    kernel_size: int = 3

    res_scale_const: float = 0.1


@dataclass
class ModelConfig:
    optical_channels: int = 13  # Sentinel-2 multispectral bands
    sar_channels: int = 2  # Sentinel-1 VV and VH polarizations
    input_dim: int = optical_channels + sar_channels
    out_dim: int = optical_channels   # reconstructed cloud-free Sentinel-2 bands

    stride: int = 1
    padding: int = 1
    kernel_size: int = 3

    res_units_count: int = 16  # Paper calls it `B`


@dataclass
class Config:
    res_block: ResidualBlockConfig = field(default_factory=ResidualBlockConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
