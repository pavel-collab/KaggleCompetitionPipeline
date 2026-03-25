"""
Collect competitions from Kaggle API for training dataset.

Usage:
    python collect_data.py
"""

import json
from kaggle.api.kaggle_api_extended import KaggleApi

from config import RAW_COMPETITIONS_FILE, RAW_DATA_DIR, MAX_PAGES, PAGE_SIZE


def collect_competitions() -> list[dict]:
    """Fetch all available competitions from Kaggle API."""
    print("Authenticating with Kaggle API...")
    api = KaggleApi()
    api.authenticate()

    all_competitions = []

    # Collect from different categories to get diverse data
    categories = ["all", "featured", "research", "playground", "getting-started"]

    for category in categories:
        print(f"\nFetching category: {category}")
        page = 1

        while page <= MAX_PAGES:
            try:
                comps = api.competitions_list(
                    page=page,
                    page_size=PAGE_SIZE,
                    search="",
                    category=category,
                    sort_by="latestDeadline",
                )

                if not comps or not comps.competitions:
                    break

                for c in comps.competitions:
                    # Build description from competition info and tags
                    description = c.description or ""
                    if c.tags:
                        tag_descriptions = [
                            tag.description for tag in c.tags
                            if tag.description
                        ]
                        if tag_descriptions:
                            description += "\n\n" + "\n\n".join(tag_descriptions)

                    tags = ", ".join(
                        tag.ref for tag in c.tags if tag.ref
                    ) if c.tags else ""

                    comp_data = {
                        "id": c.ref,
                        "title": c.title,
                        "description": description,
                        "tags": tags,
                        "category": category,
                        "deadline": c.deadline.strftime("%Y-%m-%d") if c.deadline else None,
                        "enabled_date": c.enabled_date.strftime("%Y-%m-%d") if c.enabled_date else None,
                    }

                    # Avoid duplicates
                    if not any(x["id"] == comp_data["id"] for x in all_competitions):
                        all_competitions.append(comp_data)
                        print(f"  [{len(all_competitions)}] {c.title[:50]}...")

                page += 1

            except Exception as e:
                print(f"Error on page {page}: {e}")
                break

    return all_competitions


def save_competitions(competitions: list[dict]) -> None:
    """Save competitions to JSON file."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(RAW_COMPETITIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(competitions, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(competitions)} competitions to {RAW_COMPETITIONS_FILE}")


def main():
    print("=" * 60)
    print("Kaggle Competition Collector")
    print("=" * 60)

    competitions = collect_competitions()

    if competitions:
        save_competitions(competitions)
        print(f"\nTotal unique competitions collected: {len(competitions)}")
    else:
        print("No competitions collected!")


if __name__ == "__main__":
    main()
