"""Train and evaluate the shared forecasting backbone across RNN, LSTM, and GRU models."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from artificial_life_trading_ecosystem.config import settings
from artificial_life_trading_ecosystem.data import build_feature_set, chronological_split, create_sliding_windows, create_target, load_raw_market_data, validate_market_data
from artificial_life_trading_ecosystem.models.forecasting import GRU, LSTM, SimpleRNN
from artificial_life_trading_ecosystem.models.forecasting.base_forecaster import ForecastingModelConfig
from artificial_life_trading_ecosystem.models.manifest import create_dataset_manifest
from artificial_life_trading_ecosystem.utils.seed_utils import set_deterministic_seed


def prepare_training_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, object]]:
    set_deterministic_seed(settings.seed)
    raw_df = load_raw_market_data()
    validated = validate_market_data(raw_df)
    features = build_feature_set(validated)
    targets = create_target(validated, target_mode=settings.target_mode)
    splits = chronological_split(
        validated,
        train_proportion=settings.train_proportion,
        validation_proportion=settings.validation_proportion,
        evolution_proportion=settings.evolution_proportion,
        test_proportion=settings.test_proportion,
    )
    train_df = splits["train"]
    validation_df = splits["validation"]
    feature_names = list(features.columns)

    X_train, y_train = create_sliding_windows(train_df, create_target(train_df, target_mode=settings.target_mode), window_length=settings.window_length)
    X_val, y_val = create_sliding_windows(validation_df, create_target(validation_df, target_mode=settings.target_mode), window_length=settings.window_length)

    raw_filename = f"{settings.ticker}_{settings.start_date}_{settings.end_date}_{settings.interval}.csv"
    manifest = create_dataset_manifest(
        ticker=settings.ticker,
        start_date=settings.start_date,
        end_date=settings.end_date,
        interval=settings.interval,
        source="yfinance",
        raw_filename=raw_filename,
        raw_path=Path(settings.data_dir) / "raw" / raw_filename,
        row_count=len(validated),
    )

    return X_train, y_train, X_val, y_val, feature_names, manifest


def _build_model(model_name: str, config: ForecastingModelConfig, feature_count: int) -> object:
    model_cls = {"SimpleRNN": SimpleRNN, "LSTM": LSTM, "GRU": GRU}[model_name]
    return model_cls(config=config, input_shape=(config.sequence_length, feature_count), feature_names=config.feature_names)


def main() -> None:
    X_train, y_train, X_val, y_val, feature_names, manifest = prepare_training_data()
    feature_count = X_train.shape[2]
    base_cfg = ForecastingModelConfig(
        model_type="SimpleRNN",
        sequence_length=settings.window_length,
        hidden_units=settings.hidden_units,
        dropout=settings.dropout,
        learning_rate=settings.learning_rate,
        optimizer=settings.optimizer,
        loss=settings.loss,
        epochs=settings.epochs,
        batch_size=settings.batch_size,
        early_stopping_patience=settings.early_stopping_patience,
        random_seed=settings.seed,
        target_mode=settings.target_mode,
        feature_names=feature_names,
    )

    results = {}
    for model_name in ["SimpleRNN", "LSTM", "GRU"]:
        temp_cfg = ForecastingModelConfig(**{**base_cfg.__dict__, "model_type": model_name})
        model = _build_model(model_name, temp_cfg, feature_count)
        model.build((temp_cfg.sequence_length, feature_count))
        start = time.perf_counter()
        history = model.train(X_train, y_train, X_val, y_val)
        fit_seconds = time.perf_counter() - start
        metrics = model.evaluate(X_val, y_val)
        params = model.parameter_count()
        save_dir = Path(settings.results_dir) / "forecasts" / model_name.lower()
        save_dir.mkdir(parents=True, exist_ok=True)
        payload = model.save(save_dir)
        results[model_name] = {
            "training_duration_seconds": fit_seconds,
            "history": history,
            "metrics": metrics,
            "parameter_count": params,
            "save_path": str(save_dir),
            "metadata": payload,
            "dataset_manifest": manifest,
        }

    output = Path(settings.results_dir) / "forecasts" / "comparison.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({
        "dataset_manifest": manifest,
        "feature_names": feature_names,
        "window_length": settings.window_length,
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "models": {name: {"parameter_count": info["parameter_count"], "metrics": info["metrics"]} for name, info in results.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
