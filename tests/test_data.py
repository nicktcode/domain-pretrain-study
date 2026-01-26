import os
import tempfile
import pytest


def test_format_item_as_text():
    """Item serialization should produce readable natural text."""
    from data.export_supreme import format_item_as_text

    item = {
        "name": "Box Logo Hoodie",
        "category": "Sweatshirts",
        "description": "Cotton fleece with pouch pocket.",
        "style_name": "Black, Red, Navy",
        "price_usd": 168.00,
        "price_eur": 178.00,
        "price_gbp": 158.00,
    }
    result = format_item_as_text(item)
    assert "Box Logo Hoodie" in result
    assert "Sweatshirts" in result
    assert "$168" in result
    assert "Black, Red, Navy" in result
    assert "{" not in result  # no JSON


def test_format_droplist_as_text():
    """Droplist serialization should include date, season, and items."""
    from data.export_supreme import format_droplist_as_text

    droplist = {
        "date": "2025-10-16",
        "week": 8,
        "season_name": "Fall/Winter 2025",
        "items": [
            {
                "name": "Box Logo Hoodie",
                "category": "Sweatshirts",
                "description": "Cotton fleece.",
                "style_name": "Black",
                "price_usd": 168.00,
                "price_eur": 178.00,
                "price_gbp": 158.00,
            }
        ],
        "sellout_times": [
            {
                "item_name": "Box Logo Hoodie",
                "colorway": "Black - Medium",
                "sellout_seconds": 3.2,
                "region": "EU",
            }
        ],
    }
    result = format_droplist_as_text(droplist)
    assert "Fall/Winter 2025" in result
    assert "Week 8" in result
    assert "October 16, 2025" in result or "2025-10-16" in result
    assert "Box Logo Hoodie" in result
    assert "3.2" in result or "3" in result


def test_strip_html():
    """HTML content from news articles should be stripped to plain text."""
    from data.export_supreme import strip_html

    html = "<p>Supreme's <strong>latest</strong> drop.</p><br/><ul><li>Item 1</li></ul>"
    result = strip_html(html)
    assert "<p>" not in result
    assert "<strong>" not in result
    assert "Supreme" in result
    assert "latest" in result


def test_dedup_paragraphs():
    from data.build_corpus import dedup_paragraphs

    text = "First paragraph.\n\nSecond paragraph.\n\nFirst paragraph.\n\nThird paragraph."
    result = dedup_paragraphs(text)
    assert result.count("First paragraph") == 1
    assert "Second paragraph" in result
    assert "Third paragraph" in result


def test_clean_text_normalizes_unicode():
    from data.build_corpus import clean_text

    text = "Supreme\u2019s \u201clatest\u201d drop \u2014 wow"
    result = clean_text(text)
    assert "\u2019" not in result
    assert "\u201c" not in result
    assert "\u2014" not in result


def test_tokenizer_encodes_fashion_terms():
    """Tokenizer should handle fashion vocabulary efficiently."""
    import os
    from tokenizers import Tokenizer

    # This test only runs after tokenizer is trained
    tok_path = "tokenizer/tokenizer.json"
    if not os.path.exists(tok_path):
        pytest.skip("Tokenizer not trained yet")

    tok = Tokenizer.from_file(tok_path)
    encoded = tok.encode("Supreme Gore-Tex crewneck hoodie in black colorway")
    # Should not need more than ~12 tokens for this (efficient encoding)
    assert len(encoded.ids) <= 15
    decoded = tok.decode(encoded.ids)
    assert "Supreme" in decoded
    assert "Gore" in decoded
