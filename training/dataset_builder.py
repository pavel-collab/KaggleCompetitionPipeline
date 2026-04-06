"""
Unified dataset builder for training pipeline.

Consolidates data collection, labeling, and preparation into a single class
with internal invariants and private methods.

Usage:
    python dataset_builder.py
"""

import json
import random
import time
from dataclasses import dataclass, field
from typing import Literal
import argparse

from pydantic import BaseModel, Field as PydanticField

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_COMPETITIONS_FILE,
    LABELED_DATA_FILE,
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE,
    MAX_PAGES,
    PAGE_SIZE,
    OPENAI_API_KEY,
    OPENAI_API_BASE,
    OPENAI_MODEL,
    LABELS,
    LABEL2ID,
)


class CompetitionLabel(BaseModel):
    """Structured output for competition classification."""
    type: Literal["CLASSIC ML", "LLM/NLP", "CV", "Other"] = PydanticField(
        description="Competition type"
    )


@dataclass
class Competition:
    """Single competition data with invariants."""
    id: str
    title: str
    description: str
    tags: str
    category: str = ""
    deadline: str | None = None
    enabled_date: str | None = None
    label: str | None = None

    def __post_init__(self):
        """Validate invariants."""
        assert self.id, "Competition ID cannot be empty"
        assert self.title, "Competition title cannot be empty"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "tags": self.tags,
            "category": self.category,
            "deadline": self.deadline,
            "enabled_date": self.enabled_date,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Competition":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            title=data["title"],
            description=data.get("description", ""),
            tags=data.get("tags", ""),
            category=data.get("category", ""),
            deadline=data.get("deadline"),
            enabled_date=data.get("enabled_date"),
            label=data.get("label"),
        )


@dataclass
class TrainingSample:
    """Single training sample with invariants."""
    text: str
    label: str
    label_id: int

    def __post_init__(self):
        """Validate invariants."""
        assert self.text, "Sample text cannot be empty"
        assert self.label in LABELS, f"Invalid label: {self.label}"
        assert self.label_id == LABEL2ID[self.label], "Label ID mismatch"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "label": self.label,
            "label_id": self.label_id,
        }


@dataclass
class DatasetBuilder:
    """
    Unified dataset builder with internal invariants.

    Consolidates:
    - Data collection from Kaggle API
    - Labeling with LLM
    - Dataset preparation and splitting
    - Data augmentation for volume increase
    """

    # Configuration
    max_pages: int = MAX_PAGES
    page_size: int = PAGE_SIZE
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    random_seed: int = 42
    augmentation_factor: int = 2  # How many augmented samples per original

    # Internal state
    _competitions: list[Competition] = field(default_factory=list)
    _labeled_competitions: list[Competition] = field(default_factory=list)
    _samples: list[TrainingSample] = field(default_factory=list)
    _llm: object | None = field(default=None, repr=False)
    _api: object | None = field(default=None, repr=False)

    # Invariant: label distribution should be tracked
    _label_counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize internal state."""
        self._label_counts = {label: 0 for label in LABELS}

    # ==================== Public API ====================

    def build(
        self,
        collect: bool = True,
        label: bool = True,
        augment: bool = True,
        prepare: bool = True,
    ) -> None:
        """
        Build complete dataset with all steps.

        Args:
            collect: Whether to collect fresh data from Kaggle
            label: Whether to label data with LLM
            augment: Whether to augment data for increased volume
            prepare: Whether to prepare final train/val/test splits
        """
        #TODO: exchange to logger with 2 loggers: to file and to terminal
        print("=" * 60)
        print("Dataset Builder")
        print("=" * 60)

        if collect:
            self._collect_competitions()
        else:
            self._load_raw_competitions()

        if label:
            self._label_competitions()
        else:
            self._load_labeled_competitions()

        self._validate_labeled_data()

        if augment:
            self._augment_data()

        if prepare:
            self._prepare_samples()
            train, val, test = self._stratified_split()
            self._save_splits(train, val, test)

        self._print_final_stats()

    def collect_only(self) -> None:
        """Only collect data from Kaggle."""
        self._collect_competitions()

    def label_only(self) -> None:
        """Only label existing raw data."""
        self._load_raw_competitions()
        self._label_competitions()

    def prepare_only(self, augment: bool = False) -> None:
        """Only prepare dataset from labeled data."""
        self._load_labeled_competitions()
        if augment:
            self._augment_data()
        self._prepare_samples()
        train, val, test = self._stratified_split()
        self._save_splits(train, val, test)
        self._print_final_stats()

    # ==================== Private: Data Collection ====================

    def _get_api(self):
        """Get authenticated Kaggle API instance."""
        if self._api is None:
            from kaggle.api.kaggle_api_extended import KaggleApi
            self._api = KaggleApi()
            self._api.authenticate()
        return self._api

    def _collect_competitions(self) -> None:
        """Fetch competitions from Kaggle API."""
        print("\n[1/4] Collecting competitions from Kaggle...")

        api = self._get_api()
        seen_ids: set[str] = set()

        categories = ["all", "featured", "research", "playground", "gettingStarted"]

        for category in categories:
            print(f"  Category: {category}")
            page = 1

            while page <= self.max_pages:
                try:
                    comps = api.competitions_list(
                        page=page,
                        page_size=self.page_size,
                        search="",
                        category=category,
                        sort_by="latestDeadline",
                    )

                    if not comps or not comps.competitions:
                        break

                    for c in comps.competitions:
                        if c.ref in seen_ids:
                            continue

                        seen_ids.add(c.ref)
                        competition = self._parse_competition(c, category)
                        self._competitions.append(competition)

                    page += 1

                except Exception as e:
                    print(f"    Error on page {page}: {e}")
                    break

        self._save_raw_competitions()
        print(f"  Collected {len(self._competitions)} unique competitions")

    def _parse_competition(self, c, category: str) -> Competition:
        """Parse Kaggle API competition object."""
        description = c.description or ""
        if c.tags:
            tag_descriptions = [
                tag.description for tag in c.tags if tag.description
            ]
            if tag_descriptions:
                description += "\n\n" + "\n\n".join(tag_descriptions)

        tags = ", ".join(
            tag.ref for tag in c.tags if tag.ref
        ) if c.tags else ""

        return Competition(
            id=c.ref,
            title=c.title,
            description=description,
            tags=tags,
            category=category,
            deadline=c.deadline.strftime("%Y-%m-%d") if c.deadline else None,
            enabled_date=c.enabled_date.strftime("%Y-%m-%d") if c.enabled_date else None,
        )

    def _save_raw_competitions(self) -> None:
        """Save raw competitions to file."""
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

        data = [c.to_dict() for c in self._competitions]
        with open(RAW_COMPETITIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_raw_competitions(self) -> None:
        """Load raw competitions from file."""
        if not RAW_COMPETITIONS_FILE.exists():
            raise FileNotFoundError(
                f"Raw data not found: {RAW_COMPETITIONS_FILE}\n"
                "Run with collect=True first!"
            )

        with open(RAW_COMPETITIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._competitions = [Competition.from_dict(d) for d in data]
        print(f"  Loaded {len(self._competitions)} raw competitions")

    # ==================== Private: Labeling ====================

    def _get_llm(self):
        """Get LLM instance for labeling."""
        if self._llm is None:
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set!")

            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model=OPENAI_MODEL,
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_API_BASE,
                temperature=0,
            )
        return self._llm

    def _label_competitions(self) -> None:
        """Label competitions using LLM."""
        print("\n[2/4] Labeling competitions with LLM...")

        llm = self._get_llm()
        existing = self._load_existing_labels()

        for i, comp in enumerate(self._competitions):
            if comp.id in existing:
                comp.label = existing[comp.id]
                self._labeled_competitions.append(comp)
                self._label_counts[comp.label] += 1
                continue

            print(f"  [{i+1}/{len(self._competitions)}] {comp.title[:40]}...", end=" ")

            label = self._classify_single(llm, comp)
            if label:
                print(f"-> {label}")
                comp.label = label
                self._labeled_competitions.append(comp)
                self._label_counts[label] += 1

                # Incremental save
                if len(self._labeled_competitions) % 10 == 0:
                    self._save_labeled_competitions()

                time.sleep(0.5)  # Rate limiting
            else:
                print("-> SKIPPED")

        self._save_labeled_competitions()
        print(f"  Labeled {len(self._labeled_competitions)} competitions")

    def _classify_single(self, llm, comp: Competition) -> str | None:
        """Classify a single competition."""
        try:
            from langchain_core.prompts import PromptTemplate

            #TODO: move prompt to the separate file
            prompt = PromptTemplate.from_template(
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

            chain = prompt | llm.with_structured_output(CompetitionLabel)
            result = chain.invoke({
                "title": comp.title,
                "description": comp.description[:2000],
                "tags": comp.tags,
            })
            return result.type
        except Exception as e:
            print(f"Error: {e}")
            return None

    def _load_existing_labels(self) -> dict[str, str]:
        """Load already labeled competitions."""
        if not LABELED_DATA_FILE.exists():
            return {}

        with open(LABELED_DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        return {item["id"]: item["label"] for item in data}

    def _save_labeled_competitions(self) -> None:
        """Save labeled competitions."""
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        data = [c.to_dict() for c in self._labeled_competitions]
        with open(LABELED_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_labeled_competitions(self) -> None:
        """Load labeled competitions from file."""
        if not LABELED_DATA_FILE.exists():
            raise FileNotFoundError(
                f"Labeled data not found: {LABELED_DATA_FILE}\n"
                "Run with label=True first!"
            )

        with open(LABELED_DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._labeled_competitions = [Competition.from_dict(d) for d in data]

        # Update label counts
        for comp in self._labeled_competitions:
            if comp.label in self._label_counts:
                self._label_counts[comp.label] += 1

        print(f"  Loaded {len(self._labeled_competitions)} labeled competitions")

    def _validate_labeled_data(self) -> None:
        """Validate labeled data invariants."""
        for comp in self._labeled_competitions:
            assert comp.label in LABELS, f"Invalid label: {comp.label} for {comp.id}"

        # Check for minimum samples per class
        for label, count in self._label_counts.items():
            if count < 5:
                print(f"  Warning: Only {count} samples for label '{label}'")

    # ==================== Private: Augmentation ====================

    def _augment_data(self) -> None:
        """Augment data to increase dataset volume."""
        print(f"\n[3/4] Augmenting data (factor={self.augmentation_factor})...")

        original_count = len(self._labeled_competitions)
        augmented = []

        # Find minority classes for targeted augmentation
        min_count = min(self._label_counts.values())
        target_count = max(self._label_counts.values())

        for comp in self._labeled_competitions:
            # More augmentation for minority classes
            class_count = self._label_counts[comp.label]
            aug_count = self.augmentation_factor

            # Extra augmentation for underrepresented classes
            if class_count < target_count * 0.5:
                aug_count += 1

            for i in range(aug_count):
                aug_comp = self._augment_single(comp, i)
                if aug_comp:
                    augmented.append(aug_comp)

        self._labeled_competitions.extend(augmented)

        # Update counts
        for comp in augmented:
            self._label_counts[comp.label] += 1

        print(f"  Original: {original_count}, Augmented: {len(augmented)}")
        print(f"  Total: {len(self._labeled_competitions)}")

    #TODO: Есть сомнения в эффективности стратегий аугментации без использования LLM. Пока что отключу этот параметр
    def _augment_single(self, comp: Competition, variant: int) -> Competition | None:
        """Create augmented version of a competition."""
        # Simple augmentation strategies (no LLM needed for speed)

        title = comp.title
        description = comp.description
        tags = comp.tags

        if variant == 0:
            # Strategy 1: Shuffle tags
            if tags:
                tag_list = [t.strip() for t in tags.split(",")]
                random.shuffle(tag_list)
                tags = ", ".join(tag_list)
            # Add title variation
            title = f"{title} Competition"

        elif variant == 1:
            # Strategy 2: Truncate description differently
            if description and len(description) > 500:
                start = random.randint(0, min(200, len(description) // 4))
                description = description[start:start + 1200]
            # Reverse tags
            if tags:
                tag_list = [t.strip() for t in tags.split(",")]
                tag_list.reverse()
                tags = ", ".join(tag_list)

        elif variant == 2:
            # Strategy 3: Minimal text (title + tags only)
            description = ""
            title = title.upper()

        else:
            # Strategy 4: Description emphasis
            if description:
                description = f"Competition Description: {description}"
            title = f"Kaggle: {title}"

        return Competition(
            id=f"{comp.id}_aug_{variant}",
            title=title,
            description=description,
            tags=tags,
            category=comp.category,
            label=comp.label,
        )

    # ==================== Private: Dataset Preparation ====================

    def _prepare_samples(self) -> None:
        """Convert labeled competitions to training samples."""
        print("\n[4/4] Preparing training samples...")

        for comp in self._labeled_competitions:
            if comp.label not in LABEL2ID:
                print(f"  Warning: Unknown label '{comp.label}', skipping")
                continue

            text = self._format_text(comp)
            sample = TrainingSample(
                text=text,
                label=comp.label,
                label_id=LABEL2ID[comp.label],
            )
            self._samples.append(sample)

        print(f"  Prepared {len(self._samples)} samples")

    def _format_text(self, comp: Competition) -> str:
        """Format competition as text for model input."""
        parts = [comp.title]

        if comp.tags:
            parts.append(f"Tags: {comp.tags}")

        if comp.description:
            desc = comp.description[:1500]
            parts.append(desc)

        return " [SEP] ".join(parts)

    def _stratified_split(
        self,
    ) -> tuple[list[TrainingSample], list[TrainingSample], list[TrainingSample]]:
        """Split data with stratification by label."""
        random.seed(self.random_seed)

        # Group by label
        by_label: dict[str, list[TrainingSample]] = {}
        for sample in self._samples:
            if sample.label not in by_label:
                by_label[sample.label] = []
            by_label[sample.label].append(sample)

        train, val, test = [], [], []

        for label, items in by_label.items():
            random.shuffle(items)
            n = len(items)

            n_train = int(n * self.train_ratio)
            n_val = int(n * self.val_ratio)

            train.extend(items[:n_train])
            val.extend(items[n_train:n_train + n_val])
            test.extend(items[n_train + n_val:])

        random.shuffle(train)
        random.shuffle(val)
        random.shuffle(test)

        return train, val, test

    def _save_splits(
        self,
        train: list[TrainingSample],
        val: list[TrainingSample],
        test: list[TrainingSample],
    ) -> None:
        """Save train/val/test splits."""
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        for data, filepath in [
            (train, TRAIN_FILE),
            (val, VAL_FILE),
            (test, TEST_FILE),
        ]:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump([s.to_dict() for s in data], f, ensure_ascii=False, indent=2)

        print(f"\n  Saved splits:")
        print(f"    Train: {TRAIN_FILE} ({len(train)} samples)")
        print(f"    Val:   {VAL_FILE} ({len(val)} samples)")
        print(f"    Test:  {TEST_FILE} ({len(test)} samples)")

    def _print_final_stats(self) -> None:
        """Print final dataset statistics."""
        print("\n" + "=" * 60)
        print("Final Statistics:")
        print("=" * 60)

        print("\nLabel distribution:")
        for label, count in sorted(self._label_counts.items()):
            total = sum(self._label_counts.values())
            pct = count / total * 100 if total > 0 else 0
            print(f"  {label}: {count} ({pct:.1f}%)")

        print(f"\nTotal samples: {len(self._samples)}")


def main():
    parser = argparse.ArgumentParser(description="Dataset Builder for Kaggle Competitions")
    parser.add_argument("--collect", action="store_true", help="Collect data from Kaggle API")
    parser.add_argument("--label", action="store_true", help="Label data with LLM")
    args = parser.parse_args()
    
    """Main entry point."""
    builder = DatasetBuilder(
        max_pages=15,  # Increased for more data
        augmentation_factor=3,  # Triple the data
    )

    # Full pipeline: collect=False to skip re-downloading
    # Set collect=True to fetch fresh data from Kaggle
    builder.build(
        collect=args.collect,
        label=args.label,  # Skip if already labeled
        augment=False,
        prepare=True,
    )


if __name__ == "__main__":
    main()
