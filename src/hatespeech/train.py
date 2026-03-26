from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import mlflow
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from hatespeech.config import load_config


def _compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }


def main() -> None:
    cfg = load_config()

    if cfg.tracking_uri:
        mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    train_df = pd.read_csv(cfg.train_path)
    test_df = pd.read_csv(cfg.test_path)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    def tokenize(batch):
        return tokenizer(
            batch[cfg.text_col],
            truncation=True,
            max_length=cfg.max_length,
        )

    train_ds = Dataset.from_pandas(train_df)
    test_ds = Dataset.from_pandas(test_df)

    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    train_ds = train_ds.rename_column(cfg.label_col, "labels")
    test_ds = test_ds.rename_column(cfg.label_col, "labels")

    cols_to_remove = [c for c in train_ds.column_names if c not in {"input_ids", "attention_mask", "token_type_ids", "labels"}]
    train_ds = train_ds.remove_columns(cols_to_remove)
    test_ds = test_ds.remove_columns(cols_to_remove)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = TrainingArguments(
        output_dir="outputs",
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        num_train_epochs=cfg.num_train_epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        evaluation_strategy=cfg.eval_strategy,
        save_strategy=cfg.save_strategy,
        load_best_model_at_end=True,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        logging_steps=50,
        report_to=[],
    )

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "model_name": cfg.model_name,
                "max_length": cfg.max_length,
                "learning_rate": cfg.learning_rate,
                "train_batch_size": cfg.per_device_train_batch_size,
                "eval_batch_size": cfg.per_device_eval_batch_size,
                "epochs": cfg.num_train_epochs,
                "weight_decay": cfg.weight_decay,
                "warmup_ratio": cfg.warmup_ratio,
                "device": device,
            }
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=_compute_metrics,
        )

        trainer.train()
        metrics = trainer.evaluate()

        metrics_path = Path(cfg.metrics_path)
        metrics_path.write_text(json.dumps({k: float(v) for k, v in metrics.items()}, indent=2), encoding="utf-8")

        model_dir = Path(cfg.model_dir)
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))

        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifacts(str(model_dir), artifact_path="model")

        mlflow.set_tag("run_id", run.info.run_id)


if __name__ == "__main__":
    main()
