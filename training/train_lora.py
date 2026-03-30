"""
Fine-tune LLM with LoRA using unsloth for competition classification.

This approach uses instruction-tuning to make the model output class labels.

Usage:
    python train_lora.py

Requirements:
    - GPU with CUDA support
    - pip install unsloth
"""

import json
import torch
from datasets import Dataset

from config import (
    TRAIN_FILE,
    VAL_FILE,
    LORA_MODEL_NAME,
    LORA_OUTPUT_DIR,
    LORA_BATCH_SIZE,
    LORA_EPOCHS,
    LORA_LEARNING_RATE,
    LORA_R,
    LORA_ALPHA,
    LORA_MAX_LENGTH,
    LABELS,
)
from training_plots import plot_all_metrics


# Prompt template for classification
PROMPT_TEMPLATE = """Below is a Kaggle competition. Classify it into one of these categories: {labels}

### Competition:
{text}

### Classification:
{label}"""

INFERENCE_TEMPLATE = """Below is a Kaggle competition. Classify it into one of these categories: {labels}

### Competition:
{text}

### Classification:
"""


def load_data(filepath) -> list[dict]:
    """Load dataset from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def format_for_training(sample: dict) -> dict:
    """Format sample for instruction tuning."""
    return {
        "text": PROMPT_TEMPLATE.format(
            labels=", ".join(LABELS),
            text=sample["text"][:1500],  # Truncate to save memory
            label=sample["label"],
        )
    }


def main():
    print("=" * 60)
    print("LoRA Fine-tuning with Unsloth")
    print("=" * 60)

    # Check CUDA
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. LoRA training requires GPU.")
        print("Consider using train_bert.py instead for CPU training.")
        return

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Import unsloth (requires GPU)
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
    except ImportError:
        print("Error: unsloth not installed.")
        print("Install with: pip install unsloth")
        return

    # Load model
    print(f"\nLoading model: {LORA_MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=LORA_MODEL_NAME,
        max_seq_length=LORA_MAX_LENGTH,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )

    # Add LoRA adapters
    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Load and format data
    print("\nLoading datasets...")
    train_data = load_data(TRAIN_FILE)
    val_data = load_data(VAL_FILE)

    print(f"  Train: {len(train_data)} samples")
    print(f"  Val:   {len(val_data)} samples")

    # Format for training
    train_formatted = [format_for_training(s) for s in train_data]
    val_formatted = [format_for_training(s) for s in val_data]

    train_dataset = Dataset.from_list(train_formatted)
    val_dataset = Dataset.from_list(val_formatted)

    # Training arguments
    LORA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(LORA_OUTPUT_DIR),
        num_train_epochs=LORA_EPOCHS,
        per_device_train_batch_size=LORA_BATCH_SIZE,
        per_device_eval_batch_size=LORA_BATCH_SIZE,
        learning_rate=LORA_LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=10,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=False,
        bf16=True,
        report_to="none",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=LORA_MAX_LENGTH,
        args=training_args,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    trainer.train()

    # Generate training plots
    plot_all_metrics(trainer, LORA_OUTPUT_DIR)

    # Save model
    print("\n" + "=" * 60)
    print(f"Saving model to: {LORA_OUTPUT_DIR}")
    print("=" * 60)

    model.save_pretrained(str(LORA_OUTPUT_DIR))
    tokenizer.save_pretrained(str(LORA_OUTPUT_DIR))

    # Test inference
    print("\nTesting inference...")
    test_text = "Predict house prices based on features like size, location, and age."

    FastLanguageModel.for_inference(model)

    inputs = tokenizer(
        INFERENCE_TEMPLATE.format(labels=", ".join(LABELS), text=test_text),
        return_tensors="pt",
    ).to("cuda")

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=10,
            do_sample=False,  # Greedy decoding
            pad_token_id=tokenizer.eos_token_id,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input: {test_text}")
    print(f"Output: {result.split('### Classification:')[-1].strip()}")

    print("\nDone! Model saved.")
    print("\nTo use the model:")
    print("  from unsloth import FastLanguageModel")
    print(f"  model, tokenizer = FastLanguageModel.from_pretrained('{LORA_OUTPUT_DIR}')")


if __name__ == "__main__":
    main()
