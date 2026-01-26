"""Train a BPE tokenizer on the combined corpus.

Trains a 32K vocab BPE tokenizer using HuggingFace tokenizers library.
Uses all text sources (general + domain) to ensure both vocabularies
are well represented.

Usage:
    python -m tokenizer.train_tokenizer --corpus-dir data/corpus --output tokenizer/tokenizer.json
"""

import argparse
import os
import glob

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument("--corpus-dir", default="data/corpus")
    parser.add_argument("--output", default="tokenizer/tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=16000)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Collect all corpus files
    corpus_files = glob.glob(os.path.join(args.corpus_dir, "*.txt"))
    # Exclude holdout
    corpus_files = [f for f in corpus_files if "holdout" not in f]
    print(f"Training on {len(corpus_files)} files:")
    for f in corpus_files:
        size_mb = os.path.getsize(f) / 1024 / 1024
        print(f"  {f} ({size_mb:.1f} MB)")

    # Build tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train(corpus_files, trainer)

    tokenizer.save(args.output)
    print(f"\nTokenizer saved to {args.output}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    # Quick sanity check
    test_texts = [
        "Supreme Spring/Summer 2026 Box Logo Hoodie in Black",
        "The transformer architecture uses self-attention mechanisms",
        "Gore-Tex waterproof crewneck with embroidered logo",
    ]
    for text in test_texts:
        encoded = tokenizer.encode(text)
        print(f"  '{text}' -> {len(encoded.ids)} tokens")


if __name__ == "__main__":
    main()
