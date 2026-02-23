"""Compute perplexity of a model on a text file."""

import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.transformer import Transformer
from train.dataset import PretrainDataset, tokenize_file


@torch.no_grad()
def compute_perplexity(
    model: Transformer,
    text_path: str,
    tokenizer_path: str,
    context_length: int = 1024,
    batch_size: int = 8,
    device: str = "cpu",
    max_batches: int = 200,
) -> dict:
    """Compute perplexity on a text file.

    Returns dict with perplexity, avg_loss, and token_count.
    """
    tokens_path = text_path.replace(".txt", "_eval.npy")
    tokenize_file(text_path, tokenizer_path, tokens_path)

    tokens = np.load(tokens_path)
    ds = PretrainDataset(tokens, context_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model.eval()
    model = model.to(device)

    total_loss = 0.0
    total_tokens = 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, model.config.vocab_size),
            labels.view(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += labels.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "token_count": total_tokens,
    }
