# src/model.py
# -----------------------------
# Transformer-based classifier for intent classification.
# This module defines:
# - A pretrained encoder (e.g., DistilBERT)
# - A lightweight classification head
# -----------------------------

import torch
import torch.nn as nn
from transformers import AutoModel


class TransformerClassifier(nn.Module):
    """
    Generic transformer classifier for multi-class text classification.
    Designed to be simple, interpretable, and easy to fine-tune.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Load the pretrained transformer encoder (no task-specific head)
        self.encoder = AutoModel.from_pretrained(model_name)

        # Hidden size depends on the chosen model architecture
        hidden_size = self.encoder.config.hidden_size

        # Dropout for regularization during fine-tuning
        self.dropout = nn.Dropout(dropout)

        # Linear classification head mapping encoder output to intent logits
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass:
        - Encode text with transformer
        - Extract [CLS] representation
        - Apply classification head
        """

        # Run the transformer encoder
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Use the first token representation ([CLS] for BERT-like models)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Apply dropout and linear classifier
        logits = self.classifier(self.dropout(cls_embedding))

        return logits
