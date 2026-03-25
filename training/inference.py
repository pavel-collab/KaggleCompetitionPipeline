"""
Inference script for trained models.

Usage:
    python inference.py --model bert "Your competition text here"
    python inference.py --model lora "Your competition text here"
"""

import argparse
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

from config import BERT_OUTPUT_DIR, LORA_OUTPUT_DIR, ID2LABEL


def load_bert_classifier():
    """Load fine-tuned BERT classifier."""
    return pipeline(
        "text-classification",
        model=str(BERT_OUTPUT_DIR),
        tokenizer=str(BERT_OUTPUT_DIR),
    )


def classify_bert(text: str) -> dict:
    """Classify using BERT model."""
    classifier = load_bert_classifier()
    result = classifier(text, truncation=True, max_length=512)[0]
    return {
        "label": result["label"],
        "confidence": result["score"],
    }


def classify_lora(text: str) -> dict:
    """Classify using LoRA model."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise ImportError("unsloth not installed. Run: pip install unsloth")

    from config import LABELS

    INFERENCE_TEMPLATE = """Below is a Kaggle competition. Classify it into one of these categories: {labels}

### Competition:
{text}

### Classification:
"""

    model, tokenizer = FastLanguageModel.from_pretrained(
        str(LORA_OUTPUT_DIR),
        max_seq_length=512,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)

    prompt = INFERENCE_TEMPLATE.format(labels=", ".join(LABELS), text=text[:1500])
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        temperature=0.1,
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    label = result.split("### Classification:")[-1].strip().split()[0]

    # Clean up label
    for valid_label in LABELS:
        if valid_label.lower() in label.lower():
            return {"label": valid_label, "confidence": None}

    return {"label": label, "confidence": None}


def main():
    parser = argparse.ArgumentParser(description="Classify Kaggle competition")
    parser.add_argument("--model", choices=["bert", "lora"], default="bert",
                        help="Model to use for classification")
    parser.add_argument("text", help="Competition text to classify")

    args = parser.parse_args()

    print(f"Using model: {args.model}")
    print(f"Text: {args.text[:100]}...")

    if args.model == "bert":
        result = classify_bert(args.text)
    else:
        result = classify_lora(args.text)

    print(f"\nResult:")
    print(f"  Label: {result['label']}")
    if result.get('confidence'):
        print(f"  Confidence: {result['confidence']:.4f}")


if __name__ == "__main__":
    main()
