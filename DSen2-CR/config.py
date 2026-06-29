# config.py

from dataclasses import dataclass, field


@dataclass
class ResidualBlockConfig:
    hidden_dim: int = 32
    out_dim: int = 32

    stride: int = 1
    padding: int = 0
    kernel_size: int = 3

    res_scale_const: float = 0.1


@dataclass
class Config:
    res_block: ResidualBlockConfig = field(default_factory=ResidualBlockConfig)