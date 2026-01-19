"""Clean, deduplicate, and combine all text sources into one corpus.

Reads processed text files from data/processed/ and outputs a single
cleaned corpus file. Also splits Supreme data into train/holdout for eval.

Usage:
    python -m data.build_corpus --input-dir data/processed --output-dir data/corpus
"""

import argparse
import hashlib
import os
import re


def dedup_paragraphs(text: str) -> str:
    """Remove duplicate paragraphs within a document."""
    paragraphs = text.split("\n\n")
    seen = set()
    unique = []
    for p in paragraphs:
        p_stripped = p.strip()
        if not p_stripped:
            continue
        h = hashlib.md5(p_stripped.lower().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(p_stripped)
    return "\n\n".join(unique)


def clean_text(text: str) -> str:
    """Basic text cleaning."""
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2014", ", ").replace("\u2013", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def process_file(path: str) -> str:
    """Read, clean, and dedup a text file."""
    with open(path, "r") as f:
        text = f.read()
    text = clean_text(text)
    text = dedup_paragraphs(text)
    return text


def split_by_separator(text: str, sep: str = "---") -> list[str]:
    """Split text into documents by separator."""
    docs = text.split(sep)
    return [d.strip() for d in docs if d.strip()]


def main():
    parser = argparse.ArgumentParser(description="Build training corpus")
    parser.add_argument("--input-dir", default="data/processed")
    parser.add_argument("--output-dir", default="data/corpus")
    parser.add_argument("--holdout-ratio", type=float, default=0.1,
                        help="Fraction of Supreme data to hold out for eval")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Process Supreme files (domain data with holdout split)
    supreme_files = ["supreme_news.txt", "supreme_items.txt", "supreme_droplists.txt"]
    domain_docs = []
    for fname in supreme_files:
        path = os.path.join(args.input_dir, fname)
        if os.path.exists(path):
            text = process_file(path)
            docs = split_by_separator(text)
            domain_docs.extend(docs)
            print(f"  {fname}: {len(docs)} documents")

    # Split Supreme data into train and holdout
    n_holdout = max(1, int(len(domain_docs) * args.holdout_ratio))
    supreme_train = domain_docs[:-n_holdout]
    supreme_holdout = domain_docs[-n_holdout:]

    holdout_path = os.path.join(args.output_dir, "supreme_holdout.txt")
    with open(holdout_path, "w") as f:
        f.write("\n\n".join(supreme_holdout))
    print(f"Supreme holdout: {n_holdout} documents saved to {holdout_path}")

    supreme_train_path = os.path.join(args.output_dir, "domain_supreme.txt")
    with open(supreme_train_path, "w") as f:
        f.write("\n\n".join(supreme_train))
    print(f"Supreme train: {len(supreme_train)} documents")

    # Process other domain files
    other_domain_files = ["hf_fashion.txt", "wikipedia_fashion.txt"]
    other_domain_text = []
    for fname in other_domain_files:
        path = os.path.join(args.input_dir, fname)
        if os.path.exists(path):
            text = process_file(path)
            other_domain_text.append(text)
            print(f"  {fname}: {len(text.split())} words")

    domain_other_path = os.path.join(args.output_dir, "domain_other.txt")
    with open(domain_other_path, "w") as f:
        f.write("\n\n".join(other_domain_text))

    # Process general data (FineWeb-Edu)
    fineweb_path = os.path.join(args.input_dir, "fineweb_edu.txt")
    if os.path.exists(fineweb_path):
        text = process_file(fineweb_path)
        general_path = os.path.join(args.output_dir, "general.txt")
        with open(general_path, "w") as f:
            f.write(text)
        print(f"  fineweb_edu.txt: {len(text.split())} words")
    else:
        print("  WARNING: fineweb_edu.txt not found, run fetch script first")

    # Print summary
    for fname in os.listdir(args.output_dir):
        path = os.path.join(args.output_dir, fname)
        if os.path.isfile(path):
            with open(path) as f:
                words = len(f.read().split())
            print(f"  {fname}: ~{words:,} words (~{words * 4 // 3:,} tokens)")


if __name__ == "__main__":
    main()
