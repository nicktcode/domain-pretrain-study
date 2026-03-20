# Design Notes

## What I'm building

Training a small transformer (~124M params, GPT-2 scale) from scratch to see how domain-specific data affects model quality. I'm using fashion/streetwear as the domain since I have 5+ years of data from a Supreme community platform I built.

The question I want to answer: what's the right ratio of domain text to general text in a pre-training corpus? Too little domain data and the model doesn't pick it up. Too much and it starts forgetting general English (catastrophic forgetting). There should be a sweet spot somewhere.

## Why this matters

I got interested in this because the Llama 2 paper talks about upsampling certain data categories, and Chinchilla showed data quantity matters as much as model size. I wanted to actually measure what happens when you vary the mix, even if it's at small scale.

## Data

I have a PostgreSQL database behind the Supreme platform with:
- 184 news articles (written content about drops and collabs)
- 10,279 items with names, descriptions, prices in 6 currencies, colorways
- 67,383 sellout time records (how fast stuff sold out by region)
- 368 weekly droplists

That's roughly 500K-700K tokens of domain text. Not enough on its own, so I'm adding:
- Fashion product datasets from HuggingFace (~5-10M tokens)
- Wikipedia articles on streetwear, Supreme, fashion brands (~1-2M tokens)
- FineWeb-Edu as the general text baseline (~40-80M tokens)

Aiming for ~50-100M tokens total.

I'm serializing the Supreme data as natural text, not JSON or instruction format:

```
Supreme Fall/Winter 2025, Week 6 Drop, October 16, 2025

Cross Varsity Jacket (Jackets)
Wool blend with cowhide leather sleeves, fill and quilted satin lining.
Colorways: Dark Green, Black. Price: $498 USD / 528 EUR / 448 GBP.
Sold out in EU in 19 seconds (Black XXL fastest).
```

## Architecture

Decoder-only transformer, mostly standard but with a few modern choices:

- 12 layers, 768 hidden dim, 12 attention heads
- RMSNorm instead of LayerNorm (simpler, faster, what Llama uses)
- RoPE for positional embeddings (relative positions > learned absolutes)
- GeLU activations
- 1024 token context window
- ~32K vocab BPE tokenizer trained on the combined corpus

Writing the model from scratch in PyTorch instead of pulling from HuggingFace. It's around 300 lines and I want to understand every piece.

## The experiment

Four training runs, same setup, different data mix:

| Run | General | Fashion/Supreme | Purpose |
|-----|---------|-----------------|---------|
| baseline | 100% | 0% | Control |
| light-domain | 95% | 5% | Small domain signal |
| medium-domain | 90% | 10% | Supreme oversampled 3x |
| heavy-domain | 80% | 20% | Supreme oversampled 6x |

I need to oversample the fashion data since I only have ~10M tokens of it vs ~80M general. That's normal (Llama 2 does this too) but it risks memorization, so I'll keep an eye on it.

## Training config

- AdamW, LR 6e-4 with warmup (2K steps) + cosine decay
- Batch size 64 (64K tokens per batch), gradient accumulation if needed
- BF16 mixed precision
- Gradient clipping at 1.0
- ~50-80K steps per run
- Running on an NVIDIA A10 (24 GiB, $1/hr)
- Should be ~2-4 hours per run, so $8-16 total for all 4

## Evaluation

Measuring three things:

General perplexity on Wikitext-103, to check if domain data hurts general language quality. Expecting baseline wins here.

Domain perplexity on held-out Supreme text (10% kept out of training). Want to see how much domain data actually helps, and when gains stop.

Downstream task: instruction-tune each checkpoint on a small drop summarization dataset (same one for all 4) and compare the outputs. Using ROUGE-L, BERTScore, and checking factual accuracy manually on 20 examples.

The result I'm hoping for is that medium-domain beats both baseline and heavy-domain on the downstream task. That would show there's a real optimal mixture point.

Logging to Weights & Biases: loss curves, gradient norms, LR schedule, throughput.

## Known limitations

124M params and 100M tokens is tiny. This won't produce anything useful as a model. The point is the methodology and analysis. Ideally I'd run each experiment 3x with different seeds to get variance, but that triples compute cost. Also the tokenizer is shared across all runs, so even the baseline gets fashion vocabulary baked in. I'll note all of this in the writeup.
