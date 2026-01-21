# src/train.py
# -----------------------------
# Train script for Banking77 intent classification.
# Responsibilities:
# - Build dataloaders
# - Initialize model + optimizer + loss
# - Run training loop with periodic evaluation
# - Save checkpoints + training history for reproducibility
# -----------------------------

import argparse
import json
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from data import DataConfig, get_dataloaders
from model import TransformerClassifier


# -----------------------------
# Training configuration
# -----------------------------
@dataclass
class TrainConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 128
    batch_size: int = 32
    lr: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 3
    dropout: float = 0.1
    out_dir: str = "results"
    seed: int = 42


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    """Create output directory if needed."""
    import os
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict, path: str) -> None:
    """Save a dict as pretty JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> Dict[str, float]:
    """Evaluate on a dataloader and return loss + accuracy."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total = 0
    correct = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, labels)

        preds = torch.argmax(logits, dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)

    return {
        "eval_loss": total_loss / max(total, 1),
        "eval_accuracy": correct / max(total, 1),
    }


def main():
    # -----------------------------
    # Parse CLI arguments for flexible experimentation
    # -----------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    cfg = TrainConfig(
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        out_dir=args.out_dir,
    )

    # -----------------------------
    # Environment setup
    # -----------------------------
    ensure_dir(cfg.out_dir)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Data
    # -----------------------------
    data_cfg = DataConfig(
        model_name=cfg.model_name,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
    )
    train_loader, test_loader, id2label = get_dataloaders(data_cfg)
    num_labels = len(id2label)

    # -----------------------------
    # Model + optimization
    # -----------------------------
    model = TransformerClassifier(
        model_name=cfg.model_name,
        num_labels=num_labels,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    history = {
        "config": cfg.__dict__,
        "epochs": [],
    }

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(1, cfg.epochs + 1):
        model.train()

        running_loss = 0.0
        n = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            n += labels.size(0)
            pbar.set_postfix(train_loss=running_loss / max(n, 1))

        # -----------------------------
        # Eval + checkpointing
        # -----------------------------
        eval_stats = evaluate(model, test_loader, device)
        train_loss = running_loss / max(n, 1)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **eval_stats,
        }
        history["epochs"].append(epoch_record)

        ckpt_path = f"{cfg.out_dir}/model_epoch{epoch}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "id2label": id2label,
                "config": cfg.__dict__,
            },
            ckpt_path,
        )

        save_json(history, f"{cfg.out_dir}/train_history.json")

        print(f"[epoch {epoch}] train_loss={train_loss:.4f} "
              f"eval_loss={eval_stats['eval_loss']:.4f} "
              f"eval_acc={eval_stats['eval_accuracy']:.4f} | saved {ckpt_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
