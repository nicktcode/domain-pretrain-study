"""Model configuration dataclass."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    context_length: int = 1024
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ff: int = 3072  # 4 * d_model
    dropout: float = 0.0
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads
