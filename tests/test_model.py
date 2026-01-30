import pytest
import torch


def test_model_config_param_count():
    """Model config should produce ~124M parameters."""
    from model.config import ModelConfig
    from model.transformer import Transformer

    config = ModelConfig()
    model = Transformer(config)
    param_count = sum(p.numel() for p in model.parameters())
    # ~109M with weight tying (embedding shared with output projection)
    assert 100_000_000 < param_count < 140_000_000, f"Got {param_count:,} params"


def test_model_forward_shape():
    """Forward pass should output logits with shape (batch, seq, vocab)."""
    from model.config import ModelConfig
    from model.transformer import Transformer

    config = ModelConfig(n_layers=2, d_model=128, n_heads=4, d_ff=512, vocab_size=1000)
    model = Transformer(config)

    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    logits = model(input_ids)

    assert logits.shape == (batch_size, seq_len, 1000)


def test_model_causal_masking():
    """Model should not attend to future tokens (causal masking)."""
    from model.config import ModelConfig
    from model.transformer import Transformer

    config = ModelConfig(n_layers=1, d_model=64, n_heads=2, d_ff=256, vocab_size=100)
    model = Transformer(config)
    model.eval()

    # Two sequences: same prefix, different suffix
    seq_a = torch.tensor([[1, 2, 3, 4, 5]])
    seq_b = torch.tensor([[1, 2, 3, 9, 9]])

    with torch.no_grad():
        logits_a = model(seq_a)
        logits_b = model(seq_b)

    # Logits at position 2 (predicting token at position 3) should be identical
    # because positions 3+ haven't been seen yet
    torch.testing.assert_close(logits_a[0, 2], logits_b[0, 2])


def test_rope_dimensions():
    """RoPE should not change tensor dimensions."""
    from model.rope import apply_rotary_emb, precompute_freqs_cis

    seq_len, n_heads, head_dim = 32, 4, 16
    freqs = precompute_freqs_cis(head_dim, seq_len)
    q = torch.randn(1, seq_len, n_heads, head_dim)
    k = torch.randn(1, seq_len, n_heads, head_dim)

    q_rot, k_rot = apply_rotary_emb(q, k, freqs)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
