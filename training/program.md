# BERT Training Pipeline for Kaggle Competition Classification

Autonomous training pipeline for classifying Kaggle competitions into categories: **CLASSIC ML**, **LLM/NLP**, **CV**, **Other**.

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## Quick Start

```bash
# 1. Prepare data (one-time setup)
python prepare.py

# 2. Train model
python train.py

# 3. (Optional) Refresh data from Kaggle
python prepare.py --refresh
```

## Project Structure

```
training/
├── train.py          # Training script (MODIFIABLE)
├── prepare.py        # Data prep & utilities (READ-ONLY)
├── dataset_builder.py# Data collection & labeling
├── program.md        # This file
├── data/
│   ├── raw/          # Raw competitions from Kaggle
│   └── processed/    # Train/val/test splits
└── models/
    └── bert_classifier/  # Saved model checkpoints
```

## File Roles

| File | Role | Modifiable |
|------|------|------------|
| `train.py` | Training script with hyperparameters | **YES** |
| `prepare.py` | Data loading, evaluation, utilities | NO |
| `dataset_builder.py` | Data collection from Kaggle + LLM labeling | NO |

## Objective

**Maximize validation F1 macro score** on the competition classification task.

The model classifies Kaggle competitions based on:
- Title
- Description
- Tags

## Hyperparameters (train.py)

All hyperparameters are at the top of `train.py` and can be modified:

### Model Architecture
```python
DROPOUT = 0.1                    # Dropout probability
HIDDEN_DIM = 256                 # Classifier hidden dimension
USE_HIDDEN_LAYER = True          # Use hidden layer in classifier
POOL_STRATEGY = "cls"            # Pooling: "cls", "mean", "max"
```

### Optimization
```python
BATCH_SIZE = 16                  # Training batch size
LEARNING_RATE = 2e-5             # Peak learning rate
WEIGHT_DECAY = 0.01              # L2 regularization
ADAM_BETAS = (0.9, 0.999)        # Adam momentum
WARMUP_RATIO = 0.1               # LR warmup fraction
```

### Training Control
```python
MAX_EPOCHS = 10                  # Maximum epochs
EARLY_STOPPING_PATIENCE = 3      # Patience for early stopping
GRADIENT_CLIP = 1.0              # Gradient clipping (0 = off)
USE_CLASS_WEIGHTS = True         # Balance classes
```

### Regularization
```python
LABEL_SMOOTHING = 0.0            # Label smoothing
FREEZE_EMBEDDINGS = False        # Freeze BERT embeddings
FREEZE_ENCODER_LAYERS = 0        # Layers to freeze
```

## Data Pipeline

### 1. Collection (`dataset_builder.py`)

Fetches competitions from Kaggle API:
- Categories: all, featured, research, playground, gettingStarted
- Deduplication by competition ID
- Saved to `data/raw/competitions.json`

### 2. Labeling (`dataset_builder.py`)

Uses LLM (OpenRouter) to classify each competition:
- Structured output with Pydantic
- Incremental saving every 10 records
- Saved to `data/processed/labeled_competitions.json`

### 3. Augmentation (`dataset_builder.py`)

Increases dataset volume with text variations:
- Tag shuffling/reversing
- Description truncation
- Title modifications
- Default: 3x augmentation factor

### 4. Splitting (`dataset_builder.py`)

Stratified split maintaining class distribution:
- Train: 80%
- Validation: 10%
- Test: 10%

## Evaluation Metrics

- **F1 Macro** (primary) - balanced across classes
- **F1 Weighted** - weighted by class frequency
- **Accuracy** - overall correctness
- **Per-class F1** - individual class performance

## Autonomous Experimentation Loop

For autonomous hyperparameter tuning:

```bash
while true; do
    # 1. Run training
    python train.py > results.txt 2>&1

    # 2. Extract metric
    f1=$(grep "val_f1_macro" results.txt | tail -1)
    echo "Result: $f1"

    # 3. Modify train.py hyperparameters
    # (manually or with LLM agent)

    sleep 5
done
```

## Constraints

- **Time budget**: 5 minutes (300s) wall clock per run
- **Device**: CPU, CUDA, or MPS (Apple Silicon)
- **Base model**: DistilBERT (fixed in prepare.py)
- **Labels**: 4 classes (fixed)

## Tips for Optimization

1. **Learning rate** is often the most impactful hyperparameter
2. **Class weights** help with imbalanced data
3. **Pooling strategy** can significantly affect results:
   - `cls` - fast, often sufficient
   - `mean` - captures more context
   - `max` - emphasizes strong signals
4. **Hidden layer** adds capacity but may overfit on small data
5. **Freezing layers** can help with limited data
6. **Label smoothing** can improve generalization

## Requirements

```
torch
transformers
scikit-learn
numpy
```

For data collection:
```
kaggle
langchain
langchain-openai
pydantic
```

## Environment Variables

For data labeling (optional, only if re-labeling):
```bash
export OPENAI_API_KEY="your-openrouter-key"
export OPENAI_API_BASE="https://openrouter.ai/api/v1"
export OPENAI_MODEL="google/gemini-flash-1.5"
```

## Example Output

```
==================================================
BERT Training Script
==================================================
Device: cuda
Time budget: 300s

Loading data...
Train batches: 125
Val batches: 16
Test batches: 16
Class weights: [0.8, 1.2, 1.5, 1.1]

Creating model...
Total params: 66,956,548
Trainable params: 66,956,548

Starting training...
------------------------------------------------------------
Epoch  1/10 | Train Loss: 1.2345 | Val Loss: 0.9876 | Val F1: 0.7234 | Time: 45.2s
  -> New best model saved! F1: 0.7234
Epoch  2/10 | Train Loss: 0.8765 | Val Loss: 0.7654 | Val F1: 0.8012 | Time: 44.8s
  -> New best model saved! F1: 0.8012
...

>>> val_f1_macro = 0.8456
```

## License

MIT
