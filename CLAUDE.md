# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Stack

```bash
# Full stack (all services)
docker-compose up --build -d

# Scale workers horizontally
docker-compose up --build -d --scale classify-worker=3 --scale notify-worker=2

# Local development (run infra in Docker, app locally)
docker-compose up -d postgres rabbitmq
python -m app.main scheduler     # Terminal 1
python -m app.main classify      # Terminal 2
python -m app.main notify        # Terminal 3
python -m app.main init-db       # One-time DB initialization
```

## Configuration

Copy `env.example` to `.env`. Required variables:
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` — from @BotFather
- `OPENAI_API_KEY` — OpenRouter API key (used for competition classification and translation)
- `KAGGLE_USERNAME`, `KAGGLE_KEY` — from kaggle.com/settings

Settings are loaded via Pydantic (`app/config.py`). The DB URL and RabbitMQ URL are constructed as properties from individual host/port/user/password vars.

## Architecture

The system monitors Kaggle competitions, classifies them with an LLM, and sends curated Telegram notifications.

**Data flow:**

```
Kaggle API → Scheduler → classify_queue (RabbitMQ) → Classify Worker → PostgreSQL
                                                                              ↓
Telegram ← Notify Worker ← notify_queue (RabbitMQ) ←←←←←←←←←←←←←←←←← Scheduler
```

**Scheduler logic** (`app/main.py`): Runs every 24h. If ≥5 competitions pending in DB, pushes them to `notify_queue`. Otherwise fetches up to 50 new competitions from the Kaggle API (filtered to active competitions with 7+ days remaining) and pushes to `classify_queue`.

**Classify Worker** (`app/worker.py`): Receives raw competition data, sends to LLM (OpenRouter, default: `openai/gpt-4o-mini`) for structured classification into `CLASSIC ML`, `LLM/NLP`, `CV`, or `Other`. Only saves to PostgreSQL if type is one of the first three — "Other" is discarded.

**Notify Worker** (`app/worker.py`): Receives competitions from DB, translates descriptions to Russian via LLM, formats HTML, sends via Telegram Bot API, marks DB record as `shown`.

**LLM integration** (`app/llm.py`): LangChain + OpenRouter with Pydantic-structured outputs. Two prompt templates in `app/prompts/prompts.py` — one for classification, one for Russian translation.

## MLOps Layer (`mlops/`)

Two FastAPI services deployed alongside the main stack:

- **Classifier** (`mlops/classifier/`): Serves a fine-tuned BERT model via `/classify` and `/classify/batch`. Exposes Prometheus metrics (request count, latency, predictions per class, confidence scores).
- **Drift Detector** (`mlops/drift_detector/`): Monitors for data drift (PSI, KS test, Jensen-Shannon divergence) and concept drift in prediction distributions. Thresholds: PSI > 0.2, KS p-value < 0.05.

Observability stack: Prometheus + Loki + Promtail + Grafana (dashboards at `:3000`, default `admin/admin`). MLOps services emit structured JSON logs for Loki.

## Training Pipeline (`training/`)

Offline pipeline to train a local classifier as an alternative to cloud LLM calls:

1. `collect_data.py` — Fetch competitions from Kaggle API
2. `label_data.py` — LLM-based labeling of collected data
3. `prepare_dataset.py` — Train/val/test split
4. `train_bert.py` — Fine-tune DistilBERT (~250MB, CPU-compatible, 100–500ms inference)
5. `train_lora.py` — Fine-tune TinyLlama with LoRA (~1–2GB, GPU required)
6. `evaluate.py` — Compare BERT, LoRA, and OpenRouter LLM baseline (accuracy, F1 macro/weighted)
7. `inference.py` — Test predictions on new inputs

Trained models are saved to `training/models/{bert_classifier,lora_classifier}`.

## Key Enums and Status Flow

Competition status in PostgreSQL: `new` → `queued` → `shown`

Competition types: `CLASSIC ML`, `LLM/NLP`, `CV`, `Other` (Other is filtered out at classify step)

## Logging

`app/logging_config.py`: Two handlers — console (for Docker logs) + rotating file (10MB max, 5 backups). MLOps services use structured JSON logging for Loki aggregation.
