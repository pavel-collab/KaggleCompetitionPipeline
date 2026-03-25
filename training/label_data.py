"""
Label collected competitions using LLM.

Usage:
    python label_data.py
"""

import json
import time
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from config import (
    RAW_COMPETITIONS_FILE,
    LABELED_DATA_FILE,
    PROCESSED_DATA_DIR,
    OPENAI_API_KEY,
    OPENAI_API_BASE,
    OPENAI_MODEL,
    LABELS,
)


class CompetitionLabel(BaseModel):
    """Structured output for competition classification."""
    type: Literal["CLASSIC ML", "LLM/NLP", "CV", "Other"] = Field(
        description="Competition type"
    )


CLASSIFICATION_PROMPT = PromptTemplate.from_template(
    """You are an expert Kaggle competition evaluator.

Given a competition, determine its machine learning category based on description and tags.

Competition:
- Title: {title}
- Description: {description}
- Tags: {tags}

Classify as one of:
- CLASSIC ML: Traditional ML (regression, classification, time series, tabular data)
- LLM/NLP: Natural Language Processing, text analysis, language models
- CV: Computer Vision, image classification, object detection
- Other: Anything else (optimization, simulation, etc.)

Return only the type."""
)


def get_llm() -> ChatOpenAI:
    """Create LLM instance."""
    return ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
        temperature=0,
    )


def classify_competition(llm: ChatOpenAI, competition: dict) -> str | None:
    """Classify a single competition."""
    try:
        chain = CLASSIFICATION_PROMPT | llm.with_structured_output(CompetitionLabel)

        result = chain.invoke({
            "title": competition["title"],
            "description": competition["description"][:2000],  # Truncate for cost
            "tags": competition["tags"],
        })

        return result.type

    except Exception as e:
        print(f"  Error: {e}")
        return None


def load_raw_competitions() -> list[dict]:
    """Load raw competitions from file."""
    if not RAW_COMPETITIONS_FILE.exists():
        raise FileNotFoundError(
            f"Raw data not found: {RAW_COMPETITIONS_FILE}\n"
            "Run collect_data.py first!"
        )

    with open(RAW_COMPETITIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_labels() -> dict[str, str]:
    """Load already labeled competitions to resume."""
    if not LABELED_DATA_FILE.exists():
        return {}

    with open(LABELED_DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {item["id"]: item["label"] for item in data}


def save_labeled_data(labeled: list[dict]) -> None:
    """Save labeled data."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(LABELED_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(labeled, f, ensure_ascii=False, indent=2)


def main():
    print("=" * 60)
    print("Competition Labeler")
    print("=" * 60)

    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not set!")
        return

    # Load data
    competitions = load_raw_competitions()
    existing_labels = load_existing_labels()
    print(f"Loaded {len(competitions)} competitions")
    print(f"Already labeled: {len(existing_labels)}")

    # Initialize LLM
    llm = get_llm()

    # Process competitions
    labeled_data = []
    stats = {label: 0 for label in LABELS}

    for i, comp in enumerate(competitions):
        comp_id = comp["id"]

        # Skip if already labeled
        if comp_id in existing_labels:
            label = existing_labels[comp_id]
            labeled_data.append({
                "id": comp_id,
                "title": comp["title"],
                "description": comp["description"],
                "tags": comp["tags"],
                "label": label,
            })
            stats[label] += 1
            continue

        # Classify
        print(f"[{i+1}/{len(competitions)}] {comp['title'][:50]}...", end=" ")
        label = classify_competition(llm, comp)

        if label:
            print(f"-> {label}")
            stats[label] += 1

            labeled_data.append({
                "id": comp_id,
                "title": comp["title"],
                "description": comp["description"],
                "tags": comp["tags"],
                "label": label,
            })

            # Save incrementally (in case of interruption)
            if len(labeled_data) % 10 == 0:
                save_labeled_data(labeled_data)

            # Rate limiting
            time.sleep(0.5)
        else:
            print("-> SKIPPED")

    # Final save
    save_labeled_data(labeled_data)

    # Print statistics
    print("\n" + "=" * 60)
    print("Labeling Statistics:")
    print("=" * 60)
    for label, count in stats.items():
        print(f"  {label}: {count}")
    print(f"  Total: {len(labeled_data)}")
    print(f"\nSaved to: {LABELED_DATA_FILE}")


if __name__ == "__main__":
    main()
