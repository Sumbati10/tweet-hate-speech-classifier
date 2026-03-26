from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from hatespeech.config import load_config


def main() -> None:
    cfg = load_config()

    raw_path = Path(cfg.raw_path)
    train_path = Path(cfg.train_path)
    test_path = Path(cfg.test_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path)

    df = df[[cfg.text_col, cfg.label_col]].copy()
    df[cfg.text_col] = df[cfg.text_col].astype(str)
    df[cfg.label_col] = df[cfg.label_col].astype(int)

    df = df.dropna(subset=[cfg.text_col, cfg.label_col])

    train_df, test_df = train_test_split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.random_seed,
        stratify=df[cfg.label_col] if df[cfg.label_col].nunique() > 1 else None,
    )

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)


if __name__ == "__main__":
    main()
