"""Compare results across all 4 mixture experiments.

Generates tables and plots from eval/results.json and wandb logs.

Usage:
    python -m analysis.compare_runs --results eval/results.json --output-dir analysis/figures
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def plot_perplexity_comparison(results: dict, output_dir: str):
    """Bar chart comparing general and domain perplexity across mixtures."""
    mixtures = list(results.keys())
    general_ppl = [results[m].get("general_perplexity", 0) for m in mixtures]
    domain_ppl = [results[m].get("domain_perplexity", 0) for m in mixtures]

    x = range(len(mixtures))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar([i - width/2 for i in x], general_ppl, width, label="General (Wikitext)", color="#4A90D9")
    bars2 = ax.bar([i + width/2 for i in x], domain_ppl, width, label="Domain (Supreme)", color="#E8744F")

    ax.set_xlabel("Data Mixture")
    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_title("General vs Domain Perplexity by Data Mixture")
    ax.set_xticks(x)
    ax.set_xticklabels(mixtures, rotation=15)
    ax.legend()

    # Constrain y-axis so baseline doesn't squash domain bars
    all_values = [v for v in general_ppl + domain_ppl if v > 0]
    if all_values:
        ax.set_ylim(0, max(all_values) * 1.15)

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "perplexity_comparison.png"), dpi=150)
    plt.close()
    print(f"Saved perplexity_comparison.png")


def print_results_table(results: dict):
    """Print a formatted results table."""
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Mixture':<20} {'General PPL':>12} {'Domain PPL':>12} {'Domain Gain':>12}")
    print("-" * 58)

    baseline_domain = results.get("baseline", {}).get("domain_perplexity", 0)

    for name, r in results.items():
        gen = r.get("general_perplexity", 0)
        dom = r.get("domain_perplexity", 0)
        gain = ((baseline_domain - dom) / baseline_domain * 100) if baseline_domain > 0 else 0
        print(f"{name:<20} {gen:>12.2f} {dom:>12.2f} {gain:>11.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument("--results", default="eval/results.json")
    parser.add_argument("--output-dir", default="analysis/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.results) as f:
        results = json.load(f)

    print_results_table(results)
    plot_perplexity_comparison(results, args.output_dir)

    print(f"\nFigures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
