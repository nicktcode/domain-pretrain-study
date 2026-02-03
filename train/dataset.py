"""Dataset for pre-training: reads tokenized text and serves fixed-length chunks."""

import numpy as np
import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    """Serves fixed-length token sequences from a tokenized corpus.

    The corpus is stored as a flat numpy array of token IDs.
    Each __getitem__ returns a contiguous chunk of context_length + 1 tokens
    (input = chunk[:-1], target = chunk[1:]).
    """

    def __init__(self, token_ids: np.ndarray, context_length: int):
        self.tokens = token_ids
        self.context_length = context_length
        self.n_sequences = (len(self.tokens) - 1) // context_length

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> dict:
        start = idx * self.context_length
        end = start + self.context_length + 1
        chunk = torch.from_numpy(self.tokens[start:end].astype(np.int64))
        return {
            "input_ids": chunk[:-1],
            "labels": chunk[1:],
        }


def tokenize_file(text_path: str, tokenizer_path: str, output_path: str) -> int:
    """Tokenize a text file and save as numpy array. Returns token count."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(tokenizer_path)

    with open(text_path) as f:
        text = f.read()

    encoded = tokenizer.encode(text)
    tokens = np.array(encoded.ids, dtype=np.uint16)
    np.save(output_path, tokens)
    return len(tokens)
