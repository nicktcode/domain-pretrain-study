"""Run a single pre-training experiment.

Usage:
    python -m train.run_experiment --mixture baseline --config config/training.yaml
"""

import argparse
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from model.config import ModelConfig
from model.transformer import Transformer
from train.dataset import PretrainDataset, tokenize_file
from train.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Run pre-training experiment")
    parser.add_argument("--mixture", required=True, help="Mixture name from config")
    parser.add_argument("--config", default="config/training.yaml")
    parser.add_argument("--model-config", default="config/model.yaml")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--data-dir", default="data/mixtures")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--holdout", default="data/corpus/supreme_holdout.txt")
    args = parser.parse_args()

    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)
    config = ModelConfig(**model_cfg)

    print(f"Model: {sum(p.numel() for p in Transformer(config).parameters()):,} params")
    print(f"Mixture: {args.mixture}")

    mixture_text = os.path.join(args.data_dir, f"{args.mixture}.txt")
    mixture_tokens = os.path.join(args.data_dir, f"{args.mixture}.npy")

    if not os.path.exists(mixture_tokens):
        print(f"Tokenizing {mixture_text}...")
        n_tokens = tokenize_file(mixture_text, args.tokenizer, mixture_tokens)
        print(f"  {n_tokens:,} tokens")

    tokens = np.load(mixture_tokens)
    print(f"Training on {len(tokens):,} tokens")

    with open(args.config) as f:
        train_cfg = yaml.safe_load(f)

    train_ds = PretrainDataset(tokens, config.context_length)
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = None
    if os.path.exists(args.holdout):
        holdout_tokens_path = args.holdout.replace(".txt", ".npy")
        if not os.path.exists(holdout_tokens_path):
            tokenize_file(args.holdout, args.tokenizer, holdout_tokens_path)
        val_tokens = np.load(holdout_tokens_path)
        val_ds = PretrainDataset(val_tokens, config.context_length)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = Transformer(config)
    trainer = Trainer(args.config, args.mixture, args.output_dir)
    trainer.train(model, train_loader, val_loader)

    print("Done.")


if __name__ == "__main__":
    main()
