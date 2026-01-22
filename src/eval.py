# src/eval.py
# -----------------------------
# Evaluation script for Banking77 intent classification.
# Produces:
# - accuracy, macro-F1, micro-F1
# - confusion matrix (CSV)
# - top misclassified examples (CSV)
# -----------------------------

import argparse
from typing import Dict, List

import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from .model import TransformerClassifier


def ensure_dir(path: str) -> None:
    """Create output directory if needed."""
    import os
    os.makedirs(path, exist_ok=True)


def load_checkpoint(ckpt_path: str, device: torch.device):
    """Load model + metadata from a training checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    id2label = ckpt["id2label"]
    num_labels = len(id2label)

    model = TransformerClassifier(
        model_name=cfg["model_name"],
        num_labels=num_labels,
        dropout=cfg.get("dropout", 0.1),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)

    return model, tokenizer, cfg, id2label


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    tokenizer,
    texts: List[str],
    device: torch.device,
    max_length: int,
) -> Dict[str, List]:
    """Run batched predictions and return predictions + confidences."""
    y_pred: List[int] = []
    y_conf: List[float] = []

    for t in tqdm(texts, desc="Predicting"):
        enc = tokenizer(
            t,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=-1).squeeze(0)

        pred_id = int(torch.argmax(probs).item())
        conf = float(probs[pred_id].item())

        y_pred.append(pred_id)
        y_conf.append(conf)

    return {"y_pred": y_pred, "y_conf": y_conf}


def main():
    # -----------------------------
    # Args
    # -----------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--out_dir", type=str, default="results", help="Directory to save evaluation artifacts")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Load checkpoint + dataset
    # -----------------------------
    model, tokenizer, cfg, id2label = load_checkpoint(args.ckpt, device)

    ds_test = load_dataset("banking77")["test"]
    texts: List[str] = ds_test["text"]
    y_true: List[int] = ds_test["label"]

    # -----------------------------
    # Predictions
    # -----------------------------
    preds = predict(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        max_length=cfg["max_length"],
    )
    y_pred = preds["y_pred"]
    y_conf = preds["y_conf"]

    # -----------------------------
    # Metrics
    # -----------------------------
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro")),
    }

    with open(f"{args.out_dir}/metrics.json", "w", encoding="utf-8") as f:
        import json
        json.dump({"metrics": metrics, "ckpt": args.ckpt, "config": cfg}, f, indent=2)

    # -----------------------------
    # Confusion matrix (CSV)
    # Note: 77x77 is large; CSV is best for inspection/filtering.
    # -----------------------------
    labels_sorted = sorted(id2label.keys())
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    label_names = [id2label[i] for i in labels_sorted]
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    cm_df.to_csv(f"{args.out_dir}/confusion_matrix.csv")

    # -----------------------------
    # Top misclassifications (CSV)
    # -----------------------------
    rows = []
    for t, yt, yp, c in zip(texts, y_true, y_pred, y_conf):
        if yt != yp:
            rows.append(
                {
                    "text": t,
                    "true_label": id2label[int(yt)],
                    "pred_label": id2label[int(yp)],
                    "pred_confidence": float(c),
                }
            )
    errors_df = pd.DataFrame(rows).sort_values("pred_confidence", ascending=False).head(50)
    errors_df.to_csv(f"{args.out_dir}/errors_top50.csv", index=False)

    print("Saved evaluation artifacts:")
    print(f"- {args.out_dir}/metrics.json")
    print(f"- {args.out_dir}/confusion_matrix.csv")
    print(f"- {args.out_dir}/errors_top50.csv")
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
