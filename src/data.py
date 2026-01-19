# src/data.py - Data loading and preprocessing functions

# src/data.py
# -----------------------------
# Data loading and preprocessing utilities for the Banking77 dataset.
# This module is responsible for:
# - Loading the dataset
# - Tokenizing text inputs
# - Creating PyTorch DataLoaders
# -----------------------------

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


# -----------------------------
# Configuration for data-related parameters
# -----------------------------
@dataclass
class DataConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 128
    batch_size: int = 32
    num_workers: int = 2


# -----------------------------
# Custom collate function
# Ensures that already-tokenized tensors are stacked correctly
# -----------------------------
def _collate_fn(batch):
    out = {}
    for key in batch[0].keys():
        out[key] = torch.stack([b[key] for b in batch])
    return out


# -----------------------------
# Build training and test DataLoaders for Banking77
# Returns:
# - train DataLoader
# - test DataLoader
# - id -> label name mapping
# -----------------------------
def get_dataloaders(cfg: DataConfig) -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
    # Load the Banking77 dataset from Hugging Face
    ds = load_dataset("banking77")

    # Initialize tokenizer corresponding to the chosen transformer model
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    # Tokenization function applied to dataset batches
    def tokenize(examples):
        enc = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=cfg.max_length,
        )
        enc["labels"] = examples["label"]
        return enc

    # Apply tokenization to the entire dataset
    tokenized = ds.map(
        tokenize,
        batched=True,
        remove_columns=ds["train"].column_names,
    )

    # Convert selected fields to PyTorch tensors
    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    # Create DataLoader for training data
    train_loader = DataLoader(
        tokenized["train"],
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=_collate_fn,
    )

    # Create DataLoader for test data
    test_loader = DataLoader(
        tokenized["test"],
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=_collate_fn,
    )

    # Build mapping from label IDs to human-readable intent names
    id2label = ds["train"].features["label"].int2str
    num_classes = ds["train"].features["label"].num_classes
    id2label_map = {i: id2label(i) for i in range(num_classes)}

    return train_loader, test_loader, id2label_map
