#+#+#+#+############################################################
# File: src/hatespeech/config.py
# Purpose: Load params.yaml into a typed Config object
#+#+#+#+############################################################

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class Config:
    raw_path: str
    train_path: str
    test_path: str
    text_col: str
    label_col: str
    test_size: float
    random_seed: int

    model_name: str
    max_length: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    learning_rate: float
    num_train_epochs: float
    weight_decay: float
    warmup_ratio: float
    eval_strategy: str
    save_strategy: str
    metric_for_best_model: str
    greater_is_better: bool

    experiment_name: str
    tracking_uri: str

    model_dir: str
    metrics_path: str


def load_config(path: str | Path = "params.yaml") -> Config:
    path = Path(path)
    data: Dict[str, Any]
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return Config(
        raw_path=data["data"]["raw_path"],
        train_path=data["data"]["train_path"],
        test_path=data["data"]["test_path"],
        text_col=data["data"]["text_col"],
        label_col=data["data"]["label_col"],
        test_size=float(data["data"]["test_size"]),
        random_seed=int(data["data"]["random_seed"]),
        model_name=data["train"]["model_name"],
        max_length=int(data["train"]["max_length"]),
        per_device_train_batch_size=int(data["train"]["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(data["train"]["per_device_eval_batch_size"]),
        learning_rate=float(data["train"]["learning_rate"]),
        num_train_epochs=float(data["train"]["num_train_epochs"]),
        weight_decay=float(data["train"]["weight_decay"]),
        warmup_ratio=float(data["train"]["warmup_ratio"]),
        eval_strategy=str(data["train"]["eval_strategy"]),
        save_strategy=str(data["train"]["save_strategy"]),
        metric_for_best_model=str(data["train"]["metric_for_best_model"]),
        greater_is_better=bool(data["train"]["greater_is_better"]),
        experiment_name=str(data["mlflow"]["experiment_name"]),
        tracking_uri=str(data["mlflow"].get("tracking_uri") or ""),
        model_dir=str(data["artifacts"]["model_dir"]),
        metrics_path=str(data["artifacts"]["metrics_path"]),
    )
