"""Minimal reproducible script for loading and preparing NIFTY market data."""

from __future__ import annotations

from artificial_life_trading_ecosystem.config import settings
from artificial_life_trading_ecosystem.data import (
    build_feature_set,
    chronological_split,
    create_sliding_windows,
    create_target,
    create_walk_forward_epochs,
    download_market_data,
    fit_scaler,
    load_raw_market_data,
    validate_market_data,
)
from artificial_life_trading_ecosystem.utils import set_deterministic_seed


def main() -> None:
    set_deterministic_seed(settings.seed)
    raw_df = load_raw_market_data(
        ticker=settings.ticker,
        start_date=settings.start_date,
        end_date=settings.end_date,
        interval=settings.interval,
    )
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
    val_df = splits["validation"]
    evo_df = splits["evolution"]
    test_df = splits["test"]

    scaler = fit_scaler(train_df, feature_columns=list(features.columns))
    X_train, y_train = create_sliding_windows(train_df, create_target(train_df, target_mode=settings.target_mode), window_length=settings.window_length)
    X_val, y_val = create_sliding_windows(val_df, create_target(val_df, target_mode=settings.target_mode), window_length=settings.window_length)
    X_test, y_test = create_sliding_windows(test_df, create_target(test_df, target_mode=settings.target_mode), window_length=settings.window_length)
    epochs = create_walk_forward_epochs(evo_df, epoch_length=settings.epoch_length)

    processed_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(processed_dir / "market_data.csv", index=False)

    print(f"raw rows: {len(validated)}")
    print(f"feature count: {features.shape[1]}")
    print(f"training samples: {len(X_train)}")
    print(f"validation samples: {len(X_val)}")
    print(f"evolution rows/epochs: {len(evo_df)} / {len(epochs)}")
    print(f"final test samples: {len(X_test)}")
    print(f"window length: {settings.window_length}")
    print(f"target mode: {settings.target_mode}")
    print(f"feature names: {list(features.columns)}")


if __name__ == "__main__":
    main()
