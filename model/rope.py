"""Rotary Position Embeddings (RoPE).

Encodes position information by rotating query and key vectors.
Based on the RoFormer paper and Llama implementation.
"""

import torch


def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute the complex exponentials for rotary embeddings.

    Returns tensor of shape (seq_len, dim // 2) with complex values.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(
    q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors.

    Args:
        q: (batch, seq_len, n_heads, head_dim)
        k: (batch, seq_len, n_heads, head_dim)
        freqs_cis: (seq_len, head_dim // 2)

    Returns:
        Rotated q and k with same shapes.
    """
    # Reshape to complex: (batch, seq, heads, dim/2, 2) -> complex
    q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))

    # Broadcast freqs_cis: (seq, dim/2) -> (1, seq, 1, dim/2)
    freqs = freqs_cis.unsqueeze(0).unsqueeze(2)

    # Apply rotation
    q_rotated = torch.view_as_real(q_complex * freqs).flatten(-2)
    k_rotated = torch.view_as_real(k_complex * freqs).flatten(-2)

    return q_rotated.type_as(q), k_rotated.type_as(k)
