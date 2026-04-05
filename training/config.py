"""Configuration for training pipeline."""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Data files
RAW_COMPETITIONS_FILE = RAW_DATA_DIR / "competitions.json"
LABELED_DATA_FILE = PROCESSED_DATA_DIR / "labeled_competitions.json"
TRAIN_FILE = PROCESSED_DATA_DIR / "train.json"
VAL_FILE = PROCESSED_DATA_DIR / "val.json"
TEST_FILE = PROCESSED_DATA_DIR / "test.json"

# Kaggle API settings
MAX_PAGES = 10
PAGE_SIZE = 100

# LLM settings (for labeling)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")

# Classification labels
LABELS = ["CLASSIC ML", "LLM/NLP", "CV", "Other"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
NUM_LABELS = len(LABELS)

# Training settings - BERT
BERT_MODEL_NAME = "distilbert-base-uncased"
BERT_OUTPUT_DIR = MODELS_DIR / "bert_classifier"
BERT_BATCH_SIZE = 16
BERT_EPOCHS = 10  # Increased for small dataset
BERT_LEARNING_RATE = 1e-5  # Lower LR for more stable training
BERT_MAX_LENGTH = 512
BERT_WARMUP_RATIO = 0.1  # 10% warmup steps
BERT_EARLY_STOPPING_PATIENCE = 3  # Stop if no improvement for 3 epochs

# Training settings - LoRA
LORA_MODEL_NAME = "unsloth/tinyllama-bnb-4bit"
LORA_OUTPUT_DIR = MODELS_DIR / "lora_classifier"
LORA_BATCH_SIZE = 4
LORA_EPOCHS = 3
LORA_LEARNING_RATE = 2e-4
LORA_R = 16
LORA_ALPHA = 16
LORA_MAX_LENGTH = 512

# Dataset builder settings
DEFAULT_AUGMENTATION_FACTOR = 3  # Number of augmented copies per sample
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_RANDOM_SEED = 42
