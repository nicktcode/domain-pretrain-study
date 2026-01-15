"""Fetch Wikipedia articles about streetwear, fashion, and Supreme.

Usage:
    python -m data.fetch_wikipedia --output data/processed/wikipedia_fashion.txt
"""

import argparse
import os
import re

import wikipedia


SEED_TOPICS = [
    "Supreme (brand)",
    "Streetwear",
    "Skateboarding culture",
    "James Jebbia",
    "The North Face",
    "Nike, Inc.",
    "A Bathing Ape",
    "Palace Skateboards",
    "Stussy",
    "Off-White (brand)",
    "Virgil Abloh",
    "Hypebeast",
    "Sneaker collecting",
    "Gore-Tex",
    "Fashion design",
    "Fashion week",
    "Haute couture",
    "Fast fashion",
    "Sustainable fashion",
    "Sportswear (fashion)",
    "Athleisure",
    "Japanese street fashion",
    "Harajuku",
    "Hip hop fashion",
    "Skateboard",
    "Box logo",
    "Limited edition",
    "Collaboration (fashion)",
    "Comme des Garcons",
    "Louis Vuitton",
    "Denim",
    "T-shirt",
    "Hoodie",
    "Sneakers",
    "Resale (fashion)",
    "StockX",
]


def fetch_article(title: str) -> str | None:
    """Fetch a single Wikipedia article, return clean text or None."""
    try:
        page = wikipedia.page(title, auto_suggest=False)
        text = page.content
        # Remove reference markers like [1], [2]
        text = re.sub(r"\[\d+\]", "", text)
        # Remove "== See also ==" and everything after
        text = re.sub(r"\n== See also ==.*", "", text, flags=re.DOTALL)
        text = re.sub(r"\n== References ==.*", "", text, flags=re.DOTALL)
        text = re.sub(r"\n== External links ==.*", "", text, flags=re.DOTALL)
        return text.strip()
    except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError) as e:
        print(f"  Skipping '{title}': {e}")
        return None
    except Exception as e:
        print(f"  Error fetching '{title}': {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Fetch fashion Wikipedia articles")
    parser.add_argument("--output", default="data/processed/wikipedia_fashion.txt")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    articles = []
    for topic in SEED_TOPICS:
        print(f"Fetching: {topic}")
        text = fetch_article(topic)
        if text and len(text) > 500:
            articles.append(f"# {topic}\n\n{text}")
            print(f"  Got {len(text.split())} words")

    with open(args.output, "w") as f:
        for article in articles:
            f.write(article + "\n\n---\n\n")

    total_tokens = sum(len(a.split()) for a in articles)
    print(f"\nTotal: {len(articles)} articles, ~{total_tokens:,} tokens")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
