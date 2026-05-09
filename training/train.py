"""
BERT training script for Kaggle competition classification.

This file CAN BE MODIFIED during autonomous experimentation.
Edit hyperparameters, model architecture, optimizer settings, etc.

Objective: Minimize validation F1 macro error (maximize F1 macro score).

Usage:
    python train.py

Inspired by: https://github.com/karpathy/autoresearch
"""

import gc
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import AutoModel, AutoConfig

from prepare import (
    get_dataloaders,
    load_samples,
    compute_class_weights,
    evaluate_predictions,
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE,
    BASE_MODEL,
    MAX_SEQ_LEN,
    NUM_LABELS,
    LABELS,
    ID2LABEL,
    MODELS_DIR,
    TIME_BUDGET,
)

# ==================== HYPERPARAMETERS (EDITABLE) ====================
# These can be modified by the agent during experimentation.

# Model architecture
DROPOUT = 0.1                       # Dropout probability
HIDDEN_DIM = 256                    # Hidden dimension for classifier head
USE_HIDDEN_LAYER = True             # Whether to use hidden layer in classifier
POOL_STRATEGY = "cls"               # Pooling: "cls", "mean", "max"

# Optimization
BATCH_SIZE = 16                     # Training batch size
LEARNING_RATE = 1e-5                # Peak learning rate
WEIGHT_DECAY = 0.01                 # L2 regularization
ADAM_BETAS = (0.9, 0.999)           # Adam momentum parameters
ADAM_EPS = 1e-8                     # Adam epsilon for numerical stability

# Learning rate schedule
WARMUP_RATIO = 0.1                  # Warmup as fraction of total steps
USE_SCHEDULER = True                # Whether to use LR scheduler
SCHEDULER_TYPE = "onecycle"         # Scheduler: "onecycle", "linear", "cosine"

# Training control
MAX_EPOCHS = 10                     # Maximum training epochs
EARLY_STOPPING_PATIENCE = 3         # Epochs without improvement before stop
GRADIENT_CLIP = 1.0                 # Max gradient norm (0 = no clipping)
USE_CLASS_WEIGHTS = True            # Balance classes with weights

# Regularization
LABEL_SMOOTHING = 0.0               # Label smoothing factor (0 = none)
FREEZE_EMBEDDINGS = False           # Freeze BERT embeddings
FREEZE_ENCODER_LAYERS = 0           # Number of encoder layers to freeze (0 = none)

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"          # Automatic mixed precision (CUDA only)

# ==================== MODEL ====================


@dataclass
class ModelConfig:
    """BERT classifier configuration."""
    base_model: str = BASE_MODEL
    num_labels: int = NUM_LABELS
    dropout: float = DROPOUT
    hidden_dim: int = HIDDEN_DIM
    use_hidden_layer: bool = USE_HIDDEN_LAYER
    pool_strategy: str = POOL_STRATEGY


class BERTClassifier(nn.Module):
    """
    BERT-based sequence classifier.

    Architecture:
        BERT encoder -> pooling -> (optional) hidden layer -> classifier
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Load BERT encoder
        self.bert = AutoModel.from_pretrained(config.base_model)
        hidden_size = self.bert.config.hidden_size

        # Classifier head
        if config.use_hidden_layer:
            self.classifier = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(hidden_size, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.num_labels),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(hidden_size, config.num_labels),
            )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize classifier weights."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Logits [batch, num_labels]
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Pooling strategy
        if self.config.pool_strategy == "cls":
            pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        elif self.config.pool_strategy == "mean":
            # Mean pooling with attention mask
            hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        elif self.config.pool_strategy == "max":
            hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).float()
            hidden = hidden.masked_fill(mask == 0, -1e9)
            pooled = hidden.max(1).values
        else:
            raise ValueError(f"Unknown pool strategy: {self.config.pool_strategy}")

        logits = self.classifier(pooled)
        return logits

    def num_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())

    def num_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==================== TRAINING ====================


def freeze_layers(model: BERTClassifier):
    """Freeze specified layers based on hyperparameters."""
    if FREEZE_EMBEDDINGS:
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False

    if FREEZE_ENCODER_LAYERS > 0:
        # DistilBERT has model.bert.transformer.layer
        # BERT has model.bert.encoder.layer
        if hasattr(model.bert, "transformer"):
            layers = model.bert.transformer.layer
        else:
            layers = model.bert.encoder.layer

        for i, layer in enumerate(layers):
            if i < FREEZE_ENCODER_LAYERS:
                for param in layer.parameters():
                    param.requires_grad = False


def train_epoch(
    model: BERTClassifier,
    train_loader,
    optimizer,
    scheduler,
    class_weights: torch.Tensor | None,
    scaler,
    device: str,
) -> dict:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # Forward pass (with AMP if enabled)
        if USE_AMP and scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(input_ids, attention_mask)
                loss = compute_loss(logits, labels, class_weights)

            scaler.scale(loss).backward()

            if GRADIENT_CLIP > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)

            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids, attention_mask)
            loss = compute_loss(logits, labels, class_weights)

            loss.backward()

            if GRADIENT_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)

            optimizer.step()

        if scheduler is not None and USE_SCHEDULER:
            scheduler.step()

        # Metrics
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None,
) -> torch.Tensor:
    """Compute cross-entropy loss."""
    if class_weights is not None and USE_CLASS_WEIGHTS:
        weight = class_weights.to(logits.device)
    else:
        weight = None

    loss = F.cross_entropy(
        logits,
        labels,
        weight=weight,
        label_smoothing=LABEL_SMOOTHING,
    )

    return loss


@torch.no_grad()
def evaluate(
    model: BERTClassifier,
    data_loader,
    class_weights: torch.Tensor | None,
    device: str,
) -> dict:
    """Evaluate model on data."""
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = compute_loss(logits, labels, class_weights)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    # Compute metrics
    metrics = evaluate_predictions(all_labels, all_preds, verbose=False)
    metrics["loss"] = total_loss / len(all_labels)

    return metrics


def save_model(model: BERTClassifier, path: Path):
    """Save model checkpoint."""
    path.mkdir(parents=True, exist_ok=True)

    # Save model state
    torch.save(model.state_dict(), path / "model.pt")

    # Save config
    import json
    config_dict = {
        "base_model": model.config.base_model,
        "num_labels": model.config.num_labels,
        "dropout": model.config.dropout,
        "hidden_dim": model.config.hidden_dim,
        "use_hidden_layer": model.config.use_hidden_layer,
        "pool_strategy": model.config.pool_strategy,
    }
    with open(path / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Save label mapping
    with open(path / "labels.json", "w") as f:
        json.dump({"labels": LABELS, "id2label": ID2LABEL}, f, indent=2)


def load_model(path: Path, device: str) -> BERTClassifier:
    """Load model from checkpoint."""
    import json

    with open(path / "config.json") as f:
        config_dict = json.load(f)

    config = ModelConfig(**config_dict)
    model = BERTClassifier(config)
    model.load_state_dict(torch.load(path / "model.pt", map_location=device))
    model.to(device)

    return model


# ==================== MAIN ====================


def main():
    """Main training loop."""
    print("=" * 60)
    print("BERT Training Script")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Time budget: {TIME_BUDGET}s")

    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)
    train_samples = load_samples(TRAIN_FILE)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Compute class weights
    class_weights = None
    if USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(train_samples)
        print(f"Class weights: {class_weights.tolist()}")

    # Create model
    print("\nCreating model...")
    config = ModelConfig()
    model = BERTClassifier(config)
    freeze_layers(model)
    model.to(DEVICE)

    print(f"Total params: {model.num_params():,}")
    print(f"Trainable params: {model.num_trainable_params():,}")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
    )

    # Scheduler
    total_steps = len(train_loader) * MAX_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = None
    if USE_SCHEDULER:
        if SCHEDULER_TYPE == "onecycle":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=LEARNING_RATE,
                total_steps=total_steps,
                pct_start=WARMUP_RATIO,
                anneal_strategy="cos",
            )
        elif SCHEDULER_TYPE == "linear":
            from transformers import get_linear_schedule_with_warmup
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        elif SCHEDULER_TYPE == "cosine":
            from transformers import get_cosine_schedule_with_warmup
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None

    # Training loop
    print("\nStarting training...")
    print("-" * 60)

    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    train_start = time.time()

    # Disable GC during training for performance
    gc.disable()

    for epoch in range(MAX_EPOCHS):
        epoch_start = time.time()

        # Check time budget
        elapsed = time.time() - train_start
        if elapsed > TIME_BUDGET:
            print(f"\nTime budget exceeded ({elapsed:.1f}s > {TIME_BUDGET}s)")
            break

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, class_weights, scaler, DEVICE
        )

        # Evaluate
        val_metrics = evaluate(model, val_loader, class_weights, DEVICE)

        epoch_time = time.time() - epoch_start

        # Print progress
        print(
            f"Epoch {epoch+1:2d}/{MAX_EPOCHS} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val F1: {val_metrics['f1_macro']:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Early stopping check
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_epoch = epoch + 1
            patience_counter = 0

            # Save best model
            save_model(model, MODELS_DIR / "bert_classifier")
            print(f"  -> New best model saved! F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Re-enable GC
    gc.enable()

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    # Load best model
    model = load_model(MODELS_DIR / "bert_classifier", DEVICE)

    # Evaluate on test set
    print("\nTest Set Results:")
    test_metrics = evaluate(model, test_loader, class_weights, DEVICE)
    _ = evaluate_predictions(
        [s.label_id for s in load_samples(TEST_FILE)],
        [logits_to_pred(model, s, DEVICE) for s in load_samples(TEST_FILE)],
        verbose=True,
    )

    # Summary
    total_time = time.time() - train_start
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Best epoch: {best_epoch}")
    print(f"Best val F1 macro: {best_f1:.4f}")
    print(f"Test F1 macro: {test_metrics['f1_macro']:.4f}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Model saved to: {MODELS_DIR / 'bert_classifier'}")

    # Return metric for automated evaluation
    return {
        "val_f1_macro": best_f1,
        "test_f1_macro": test_metrics["f1_macro"],
        "test_accuracy": test_metrics["accuracy"],
        "total_time": total_time,
        "best_epoch": best_epoch,
    }


def logits_to_pred(model: BERTClassifier, sample, device: str) -> int:
    """Get prediction for a single sample."""
    from prepare import get_tokenizer, MAX_SEQ_LEN

    tokenizer = get_tokenizer()
    encoding = tokenizer(
        sample.text,
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LEN,
        return_tensors="pt",
    )

    model.eval()
    with torch.no_grad():
        logits = model(
            encoding["input_ids"].to(device),
            encoding["attention_mask"].to(device),
        )
    return logits.argmax(dim=-1).item()


if __name__ == "__main__":
    results = main()
    print(f"\n>>> val_f1_macro = {results['val_f1_macro']:.4f}")
