"""Build the 4 data mixture variants from corpus files.

Reads from data/corpus/ and produces 4 combined text files,
one per mixture, ready for tokenization.

Usage:
    python -m data.build_mixtures --corpus-dir data/corpus --output-dir data/mixtures --config config/data_mixtures.yaml
"""

import argparse
import os
import random

import yaml


def read_file(path: str) -> str:
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return ""


def count_tokens_approx(text: str) -> int:
    """Rough token count (words * 1.3)."""
    return int(len(text.split()) * 1.3)


def oversample(text: str, factor: int) -> str:
    """Repeat text N times with shuffled paragraph order."""
    if factor <= 1:
        return text
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    result_paragraphs = []
    for _ in range(factor):
        shuffled = paragraphs.copy()
        random.shuffle(shuffled)
        result_paragraphs.extend(shuffled)
    return "\n\n".join(result_paragraphs)


def main():
    parser = argparse.ArgumentParser(description="Build data mixtures")
    parser.add_argument("--corpus-dir", default="data/corpus")
    parser.add_argument("--output-dir", default="data/mixtures")
    parser.add_argument("--config", default="config/data_mixtures.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load corpus files
    general_text = read_file(os.path.join(args.corpus_dir, "general.txt"))
    supreme_text = read_file(os.path.join(args.corpus_dir, "domain_supreme.txt"))
    other_domain_text = read_file(os.path.join(args.corpus_dir, "domain_other.txt"))

    general_tokens = count_tokens_approx(general_text)
    domain_tokens = count_tokens_approx(supreme_text) + count_tokens_approx(other_domain_text)
    print(f"General: ~{general_tokens:,} tokens")
    print(f"Domain: ~{domain_tokens:,} tokens")
    print()

    # Fixed token budget: all runs train on the same total tokens.
    # Use the general corpus size as the budget (baseline = 100% general).
    token_budget = general_tokens
    print(f"Token budget per run: ~{token_budget:,}")
    print()

    general_words = general_text.split()

    for mix_name, mix_config in config["mixtures"].items():
        print(f"Building mixture: {mix_name}")
        general_ratio = mix_config["general"]
        domain_ratio = mix_config["domain"]
        supreme_os = mix_config.get("supreme_oversample", 1)

        if domain_ratio == 0:
            output_text = general_text
        else:
            # How many tokens of each type we need
            target_domain_tokens = int(token_budget * domain_ratio)
            target_general_tokens = int(token_budget * general_ratio)

            # Build domain portion: oversample Supreme, then repeat the whole
            # domain block as needed to hit the target
            oversampled_supreme = oversample(supreme_text, supreme_os)
            domain_block = oversampled_supreme + "\n\n" + other_domain_text
            domain_block_tokens = count_tokens_approx(domain_block)

            # Repeat domain block if it's smaller than target
            if domain_block_tokens < target_domain_tokens and domain_block_tokens > 0:
                repeats = (target_domain_tokens // domain_block_tokens) + 1
                domain_paragraphs = [p.strip() for p in domain_block.split("\n\n") if p.strip()]
                expanded = []
                for _ in range(repeats):
                    shuffled = domain_paragraphs.copy()
                    random.shuffle(shuffled)
                    expanded.extend(shuffled)
                domain_block = "\n\n".join(expanded)

            # Truncate domain to target
            domain_words = domain_block.split()
            target_domain_words = int(target_domain_tokens / 1.3)
            if len(domain_words) > target_domain_words:
                domain_block = " ".join(domain_words[:target_domain_words])

            # Truncate general to target
            target_general_words = int(target_general_tokens / 1.3)
            truncated_general = " ".join(general_words[:target_general_words])

            output_text = truncated_general + "\n\n" + domain_block

        # Save
        output_path = os.path.join(args.output_dir, f"{mix_name}.txt")
        with open(output_path, "w") as f:
            f.write(output_text)

        token_count = count_tokens_approx(output_text)
        print(f"  {mix_name}: ~{token_count:,} tokens -> {output_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
