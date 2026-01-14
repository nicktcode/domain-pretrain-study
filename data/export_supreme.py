"""Export Supreme data from PostgreSQL to natural text files.

Connects to the cmtyoneaio database (tenant_id=1 for Supreme)
and exports news articles, items, and droplists with sellout times
as readable natural text for pre-training.

Usage:
    python -m data.export_supreme --output-dir data/processed
"""

import argparse
import os
import re
from datetime import datetime

import psycopg2
from bs4 import BeautifulSoup


TENANT_ID = 1  # SupremeCommunity


def get_connection():
    """Connect using .pgpass or environment variables."""
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "127.0.0.1"),
        port=int(os.environ.get("PGPORT", "5439")),
        user=os.environ.get("PGUSER", "cmtyone_user"),
        dbname=os.environ.get("PGDATABASE", "cmtyoneaio"),
    )


def strip_html(html: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")

    # Replace <br> and block elements with newlines
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for tag in soup.find_all(["p", "h1", "h2", "h3", "h4", "li", "div"]):
        tag.insert_before("\n")
        tag.insert_after("\n")

    text = soup.get_text()
    # Normalize whitespace: collapse multiple spaces, keep single newlines
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def format_item_as_text(item: dict) -> str:
    """Convert an item dict to natural text."""
    parts = []

    name = item["name"]
    category = item.get("category")
    if category:
        parts.append(f"{name} ({category})")
    else:
        parts.append(name)

    desc = item.get("description", "")
    if desc:
        parts.append(desc.strip())

    colorways = item.get("style_name", "")
    if colorways:
        parts.append(f"Colorways: {colorways}.")

    prices = []
    if item.get("price_usd"):
        prices.append(f"${item['price_usd']:.0f} USD")
    if item.get("price_eur"):
        prices.append(f"{item['price_eur']:.0f} EUR")
    if item.get("price_gbp"):
        prices.append(f"{item['price_gbp']:.0f} GBP")
    if prices:
        parts.append(f"Price: {' / '.join(prices)}.")

    return "\n".join(parts)


def format_droplist_as_text(droplist: dict) -> str:
    """Convert a droplist dict with items and sellout times to natural text."""
    date_str = droplist["date"]
    if isinstance(date_str, str):
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    else:
        date_obj = date_str
    formatted_date = date_obj.strftime("%B %d, %Y")

    header = f"Supreme {droplist['season_name']}, Week {droplist['week']} Drop, {formatted_date}"
    sections = [header, ""]

    for item in droplist.get("items", []):
        sections.append(format_item_as_text(item))

        # Add sellout times for this item if available
        item_sellouts = [
            s for s in droplist.get("sellout_times", [])
            if s["item_name"] == item["name"] and s["sellout_seconds"] is not None
        ]
        if item_sellouts:
            fastest = min(item_sellouts, key=lambda s: s["sellout_seconds"])
            sections.append(
                f"Sold out in {fastest['region']} in {fastest['sellout_seconds']:.0f} seconds "
                f"({fastest['colorway']})."
            )

        sections.append("")  # blank line between items

    return "\n".join(sections).strip()


def export_news(conn, output_path: str) -> int:
    """Export news articles as plain text. Returns count."""
    cur = conn.cursor()
    cur.execute("""
        SELECT title, content, published_at
        FROM news
        WHERE tenant_id = %s AND published = true AND content IS NOT NULL AND content != ''
        ORDER BY published_at
    """, (TENANT_ID,))

    count = 0
    with open(output_path, "w") as f:
        for title, content, published_at in cur:
            text = strip_html(content)
            if not text:
                continue
            date_str = published_at.strftime("%B %d, %Y") if published_at else ""
            f.write(f"# {title}\n")
            if date_str:
                f.write(f"{date_str}\n")
            f.write(f"\n{text}\n\n---\n\n")
            count += 1

    cur.close()
    return count


def export_items(conn, output_path: str) -> int:
    """Export all items with descriptions as natural text. Returns count."""
    cur = conn.cursor()
    cur.execute("""
        SELECT i.name, ic.name as category, i.description, i.style_name,
               i.price_usd, i.price_eur, i.price_gbp
        FROM item i
        LEFT JOIN item_category ic ON i.category_id = ic.id
        WHERE i.tenant_id = %s AND i.published = true
        ORDER BY i.created_at
    """, (TENANT_ID,))

    count = 0
    with open(output_path, "w") as f:
        for row in cur:
            item = {
                "name": row[0],
                "category": row[1],
                "description": row[2],
                "style_name": row[3],
                "price_usd": row[4],
                "price_eur": row[5],
                "price_gbp": row[6],
            }
            text = format_item_as_text(item)
            f.write(f"{text}\n\n")
            count += 1

    cur.close()
    return count


def export_droplists(conn, output_path: str) -> int:
    """Export droplists with items and sellout times. Returns count."""
    cur = conn.cursor()

    # Get all droplists
    cur.execute("""
        SELECT d.id, d.date, d.week, s.name as season_name
        FROM droplist d
        JOIN season s ON d.season_id = s.id
        WHERE d.tenant_id = %s AND d.published = true
        ORDER BY d.date
    """, (TENANT_ID,))
    droplists = cur.fetchall()

    count = 0
    with open(output_path, "w") as f:
        for dl_id, dl_date, dl_week, season_name in droplists:
            # Get items for this droplist
            cur.execute("""
                SELECT i.name, ic.name as category, i.description, i.style_name,
                       i.price_usd, i.price_eur, i.price_gbp
                FROM item i
                LEFT JOIN item_category ic ON i.category_id = ic.id
                WHERE i.droplist_id = %s AND i.tenant_id = %s
                ORDER BY ic.sort_order, i.name
            """, (dl_id, TENANT_ID))

            items = []
            for row in cur:
                items.append({
                    "name": row[0],
                    "category": row[1],
                    "description": row[2],
                    "style_name": row[3],
                    "price_usd": row[4],
                    "price_eur": row[5],
                    "price_gbp": row[6],
                })

            if not items:
                continue

            # Get sellout times for this droplist
            cur.execute("""
                SELECT st.item_name, st.colorway, st.sellout_seconds, s.region
                FROM sellout s
                JOIN sellout_time st ON st.sellout_id = s.id
                WHERE s.droplist_id = %s AND s.tenant_id = %s
                ORDER BY st.position
            """, (dl_id, TENANT_ID))

            sellout_times = []
            for row in cur:
                sellout_times.append({
                    "item_name": row[0],
                    "colorway": row[1],
                    "sellout_seconds": row[2],
                    "region": row[3],
                })

            droplist_data = {
                "date": dl_date,
                "week": dl_week or 0,
                "season_name": season_name,
                "items": items,
                "sellout_times": sellout_times,
            }
            text = format_droplist_as_text(droplist_data)
            f.write(f"{text}\n\n---\n\n")
            count += 1

    cur.close()
    return count


def main():
    parser = argparse.ArgumentParser(description="Export Supreme data for pre-training")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    conn = get_connection()
    try:
        n = export_news(conn, os.path.join(args.output_dir, "supreme_news.txt"))
        print(f"Exported {n} news articles")

        n = export_items(conn, os.path.join(args.output_dir, "supreme_items.txt"))
        print(f"Exported {n} items")

        n = export_droplists(conn, os.path.join(args.output_dir, "supreme_droplists.txt"))
        print(f"Exported {n} droplists")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
