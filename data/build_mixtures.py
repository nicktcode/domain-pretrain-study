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

    for mix_name, mix_config in config["mixtures"].items():
        print(f"Building mixture: {mix_name}")
        general_ratio = mix_config["general"]
        domain_ratio = mix_config["domain"]
        supreme_os = mix_config.get("supreme_oversample", 1)

        if domain_ratio == 0:
            # Pure general
            output_text = general_text
        else:
            # Oversample Supreme if configured
            oversampled_supreme = oversample(supreme_text, supreme_os)
            domain_combined = oversampled_supreme + "\n\n" + other_domain_text

            # Calculate how much general text to include
            domain_token_count = count_tokens_approx(domain_combined)
            target_general_tokens = int(domain_token_count * (general_ratio / domain_ratio))

            # Truncate general text to target
            general_words = general_text.split()
            target_words = int(target_general_tokens / 1.3)
            if target_words < len(general_words):
                truncated_general = " ".join(general_words[:target_words])
            else:
                truncated_general = general_text

            output_text = truncated_general + "\n\n" + domain_combined

        # Save
        output_path = os.path.join(args.output_dir, f"{mix_name}.txt")
        with open(output_path, "w") as f:
            f.write(output_text)

        token_count = count_tokens_approx(output_text)
        print(f"  {mix_name}: ~{token_count:,} tokens -> {output_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
