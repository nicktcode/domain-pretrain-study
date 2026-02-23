"""Evaluate all 4 checkpoints on general and domain text.

Usage:
    python -m eval.run_eval --checkpoint-dir checkpoints --tokenizer tokenizer/tokenizer.json
"""

import argparse
import json
import os

import torch
import yaml

from model.config import ModelConfig
from model.transformer import Transformer
from eval.perplexity import compute_perplexity


def load_model_from_checkpoint(checkpoint_path: str, model_config_path: str) -> Transformer:
    with open(model_config_path) as f:
        cfg = yaml.safe_load(f)
    config = ModelConfig(**cfg)
    model = Transformer(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--model-config", default="config/model.yaml")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--general-text", default="data/corpus/general.txt",
                        help="General text for perplexity eval")
    parser.add_argument("--domain-text", default="data/corpus/supreme_holdout.txt",
                        help="Domain text for perplexity eval")
    parser.add_argument("--output", default="eval/results.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}

    mixtures = ["baseline", "light_domain", "medium_domain", "heavy_domain"]

    for mixture in mixtures:
        ckpt_path = os.path.join(args.checkpoint_dir, mixture, "checkpoint_best.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(args.checkpoint_dir, mixture, "checkpoint_final.pt")
        if not os.path.exists(ckpt_path):
            print(f"Skipping {mixture}: no checkpoint found")
            continue

        print(f"\nEvaluating: {mixture}")
        model = load_model_from_checkpoint(ckpt_path, args.model_config)

        result = {"mixture": mixture}

        if os.path.exists(args.general_text):
            print(f"  General perplexity...")
            gen = compute_perplexity(model, args.general_text, args.tokenizer, device=device, max_batches=100)
            result["general_perplexity"] = gen["perplexity"]
            result["general_loss"] = gen["avg_loss"]
            print(f"    PPL: {gen['perplexity']:.2f}")

        if os.path.exists(args.domain_text):
            print(f"  Domain perplexity...")
            dom = compute_perplexity(model, args.domain_text, args.tokenizer, device=device)
            result["domain_perplexity"] = dom["perplexity"]
            result["domain_loss"] = dom["avg_loss"]
            print(f"    PPL: {dom['perplexity']:.2f}")

        results[mixture] = result

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    print(f"\n{'Mixture':<20} {'General PPL':>12} {'Domain PPL':>12}")
    print("-" * 46)
    for name, r in results.items():
        gen = f"{r.get('general_perplexity', 0):.2f}"
        dom = f"{r.get('domain_perplexity', 0):.2f}"
        print(f"{name:<20} {gen:>12} {dom:>12}")


if __name__ == "__main__":
    main()
