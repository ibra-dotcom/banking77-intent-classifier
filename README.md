# Banking77 Intent Classifier

A multi-class intent classification system for banking customer queries, achieving **90.8% accuracy** across 77 intent categories using fine-tuned DistilBERT.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Transformers](https://img.shields.io/badge/🤗_Transformers-4.30+-yellow.svg)

## Overview

This project implements an intent classification model for the [Banking77 dataset](https://huggingface.co/datasets/PolyAI/banking77), a challenging benchmark containing 13,083 customer service queries across 77 fine-grained banking intents.

Beyond achieving strong baseline performance, this work includes error analysis revealing that **label ambiguity — not model capacity — is the primary source of errors**, suggesting directions for future improvement.

## Results

### Baseline Performance

We establish a supervised baseline using a pretrained Transformer encoder fine-tuned end-to-end with a lightweight classification head.

| Metric | Score |
|--------|-------|
| **Accuracy** | 90.8% |
| **Macro F1** | 90.8% |
| **Micro F1** | 90.8% |

Macro F1 is emphasized to account for class imbalance and ensure consistent performance across all 77 intents.

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | `distilbert-base-uncased` |
| Max Sequence Length | 128 |
| Batch Size | 32 |
| Learning Rate | 2e-5 |
| Weight Decay | 0.01 |
| Dropout | 0.1 |
| Epochs | 3 |

## Error Analysis

### Top Confusions

An analysis of the most frequent misclassifications with high prediction confidence reveals that errors cluster around semantically overlapping intents:

| Confused Pair | Pattern |
|---------------|---------|
| `pending_transfer` ↔ `balance_not_updated_after_bank_transfer` | Transfer status ambiguity |
| `pin_blocked` ↔ `card_not_working` | Card access issues |
| `get_disposable_virtual_card` ↔ `getting_virtual_card` | Virtual card requests |
| `exchange_charge` ↔ `exchange_rate` | Currency conversion queries |

### Key Insight: Label Ambiguity

A significant portion of model errors are attributable to **label ambiguity rather than model capacity**. Several Banking77 intents overlap in real user language, making certain distinctions difficult even for humans without additional context.

For example:
- *"Why is my payment still pending?"* — Is this `pending_transfer` or `pending_card_payment`?
- *"Where do I find the exchange rate?"* — Is this `exchange_rate` or `exchange_charge`?

This suggests that future improvements may benefit more from:

1. **Intent consolidation or hierarchical labeling** — Grouping semantically similar intents
2. **Context-aware modeling** — Multi-turn conversation understanding
3. **Confidence-based fallback strategies** — Requesting clarification when prediction confidence is low

...rather than purely increasing model complexity.

## Quick Start

### Installation

```bash
git clone https://github.com/ibra-dotcom/banking77-intent-classifier.git
cd banking77-intent-classifier
pip install -r requirements.txt
```

### Training

```bash
python src/train.py
```

### Evaluation

```bash
python src/eval.py --ckpt results/model_epoch3.pt --out_dir results
```

### Inference

```python
from src.predict import IntentClassifier

classifier = IntentClassifier("results/model_epoch3.pt")
result = classifier.predict("I need to cancel my card")

print(f"Intent: {result['intent']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Project Structure

```
banking77-intent-classifier/
├── src/
│   ├── train.py          # Training script
│   ├── eval.py           # Evaluation with metrics and error analysis
│   ├── predict.py        # Inference module
│   └── data.py           # Dataset loading and preprocessing
├── results/
│   ├── model_epoch3.pt   # Trained model checkpoint
│   ├── metrics.json      # Evaluation metrics
│   ├── confusion_matrix.csv
│   └── errors_top50.csv  # Top misclassifications for analysis
├── requirements.txt
└── README.md
```

## Dataset

The [Banking77](https://arxiv.org/abs/2003.04807) dataset (Casanueva et al., 2020) contains:

- **Training set:** 10,003 examples
- **Test set:** 3,080 examples
- **Classes:** 77 fine-grained banking intents

## Future Work

- [ ] Confidence calibration for uncertainty estimation
- [ ] Hierarchical classification with intent grouping
- [ ] Multi-turn context integration
- [ ] Deployment as REST API

## References

- [Banking77 Dataset Paper](https://arxiv.org/abs/2003.04807) — Casanueva et al., 2020
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108) — Sanh et al., 2019
- [Hugging Face Transformers](https://github.com/huggingface/transformers)

## Author

**Ibra** — [GitHub](https://github.com/ibra-dotcom) | [LinkedIn](https://www.linkedin.com/in/ibrahim-ba)
