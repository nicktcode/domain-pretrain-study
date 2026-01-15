"""Fetch fashion product datasets from HuggingFace.

Downloads and filters fashion-related text data from public HF datasets,
outputting clean text files for corpus building.

Usage:
    python -m data.fetch_hf_datasets --output data/processed/hf_fashion.txt --max-tokens 10000000
"""

import argparse
import os
import re

from datasets import load_dataset
from tqdm import tqdm


def clean_text(text: str) -> str:
    """Normalize whitespace and remove junk."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    # skip very short or very long entries
    if len(text) < 50 or len(text) > 5000:
        return ""
    return text


def fetch_amazon_fashion_reviews(max_tokens: int) -> list[str]:
    """Fetch fashion product reviews from Amazon reviews dataset."""
    texts = []
    token_count = 0

    try:
        ds = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            "raw_review_All_Beauty",
            split="full",
            trust_remote_code=True,
        )
    except Exception:
        # fallback: try a simpler fashion dataset
        print("Amazon reviews not available, trying alternative...")
        ds = load_dataset(
            "juliensimon/amazon-shoe-reviews",
            split="train",
        )

    text_field = "text" if "text" in ds.column_names else "reviewText"

    for row in tqdm(ds, desc="Fashion reviews"):
        text = clean_text(row.get(text_field, ""))
        if text:
            texts.append(text)
            token_count += len(text.split())
            if token_count >= max_tokens:
                break

    return texts


def fetch_fashion_products(max_tokens: int) -> list[str]:
    """Fetch fashion product descriptions."""
    texts = []
    token_count = 0

    try:
        ds = load_dataset(
            "TeoCalvo/FlipkartProductsCleaned",
            split="train",
        )
        for row in tqdm(ds, desc="Product descriptions"):
            desc = row.get("description", "")
            name = row.get("product_name", "")
            # Only keep fashion-related
            category = str(row.get("product_category_tree", "")).lower()
            if not any(kw in category for kw in ["clothing", "fashion", "shoe", "accessori", "wear"]):
                continue
            combined = f"{name}\n{clean_text(desc)}" if desc else ""
            if combined.strip():
                texts.append(combined.strip())
                token_count += len(combined.split())
                if token_count >= max_tokens:
                    break
    except Exception as e:
        print(f"Fashion products dataset error: {e}")

    return texts


def main():
    parser = argparse.ArgumentParser(description="Fetch fashion datasets from HuggingFace")
    parser.add_argument("--output", default="data/processed/hf_fashion.txt")
    parser.add_argument("--max-tokens", type=int, default=10_000_000)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    all_texts = []

    reviews = fetch_amazon_fashion_reviews(args.max_tokens // 2)
    print(f"Fetched {len(reviews)} fashion reviews")
    all_texts.extend(reviews)

    remaining = args.max_tokens - sum(len(t.split()) for t in all_texts)
    if remaining > 0:
        products = fetch_fashion_products(remaining)
        print(f"Fetched {len(products)} product descriptions")
        all_texts.extend(products)

    with open(args.output, "w") as f:
        for text in all_texts:
            f.write(text + "\n\n")

    total_tokens = sum(len(t.split()) for t in all_texts)
    print(f"Total: {len(all_texts)} documents, ~{total_tokens:,} tokens")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
