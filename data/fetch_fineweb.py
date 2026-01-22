"""Fetch a sample of FineWeb-Edu from HuggingFace for general pre-training text.

FineWeb-Edu is a filtered subset of Common Crawl with educational content.
We sample ~80M tokens worth.

Usage:
    python -m data.fetch_fineweb --output data/processed/fineweb_edu.txt --max-tokens 80000000
"""

import argparse
import os

from datasets import load_dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Fetch FineWeb-Edu sample")
    parser.add_argument("--output", default="data/processed/fineweb_edu.txt")
    parser.add_argument("--max-tokens", type=int, default=80_000_000)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading FineWeb-Edu (streaming, target ~{args.max_tokens:,} tokens)...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    token_count = 0
    doc_count = 0

    with open(args.output, "w") as f:
        for row in tqdm(ds, desc="Downloading"):
            text = row.get("text", "").strip()
            if not text or len(text) < 100:
                continue

            f.write(text + "\n\n")
            token_count += len(text.split())
            doc_count += 1

            if token_count >= args.max_tokens:
                break

    print(f"Done: {doc_count:,} documents, ~{token_count:,} tokens")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
