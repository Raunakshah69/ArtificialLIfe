from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from artificial_life_trading_ecosystem.data import (
    build_feature_set,
    chronological_split,
    clean_market_data,
    create_sliding_windows,
    create_target,
    create_walk_forward_epochs,
    fit_scaler,
    validate_market_data,
)


@pytest.fixture
def sample_market_df() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    close = np.linspace(100.0, 200.0, len(dates))
    open_prices = close - 1.0
    high = close + 2.0
    low = close - 2.0
    volume = np.linspace(1000, 2000, len(dates))
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": open_prices,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )


def test_chronological_sorting() -> None:
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"]),
            "Open": [10, 11, 12],
            "High": [12, 13, 14],
            "Low": [9, 10, 11],
            "Close": [11, 12, 13],
            "Volume": [100, 200, 300],
        }
    )
    cleaned = validate_market_data(df)
    assert cleaned["Date"].is_monotonic_increasing


def test_duplicate_detection() -> None:
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
            "Open": [10, 10, 11],
            "High": [12, 12, 13],
            "Low": [9, 9, 10],
            "Close": [11, 11, 12],
            "Volume": [100, 100, 200],
        }
    )
    with pytest.raises(ValueError, match="Duplicate timestamps"):
        validate_market_data(df)


def test_invalid_ohlc_detection() -> None:
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "Open": [10, 11],
            "High": [8, 12],
            "Low": [9, 10],
            "Close": [11, 13],
            "Volume": [100, 200],
        }
    )
    with pytest.raises(ValueError, match="OHLC"):
        validate_market_data(df)


def test_target_alignment(sample_market_df: pd.DataFrame) -> None:
    targets = create_target(sample_market_df, target_mode="regression")
    assert len(targets) == len(sample_market_df)
    assert np.isclose(targets.iloc[0], (sample_market_df["Close"].iloc[1] - sample_market_df["Close"].iloc[0]) / sample_market_df["Close"].iloc[0])


def test_window_shape(sample_market_df: pd.DataFrame) -> None:
    feature_df = build_feature_set(sample_market_df)
    targets = create_target(feature_df, target_mode="regression")
    X, y = create_sliding_windows(feature_df, targets, window_length=30)
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 30
    assert X.shape[2] == feature_df.shape[1]


def test_no_future_values_enter_X(sample_market_df: pd.DataFrame) -> None:
    feature_df = build_feature_set(sample_market_df)
    targets = create_target(feature_df, target_mode="regression")
    X, y = create_sliding_windows(feature_df, targets, window_length=10)
    first_window = X[0, :, 0]
    assert np.all(first_window == feature_df["Open"].iloc[:10].to_numpy())
    assert y[0] == targets.iloc[9]


def test_chronological_train_validation_evolution_test_separation(sample_market_df: pd.DataFrame) -> None:
    splits = chronological_split(sample_market_df, train_proportion=0.60, validation_proportion=0.15, evolution_proportion=0.15, test_proportion=0.10)
    train_df, val_df, evo_df, test_df = splits["train"], splits["validation"], splits["evolution"], splits["test"]
    assert train_df["Date"].max() < val_df["Date"].min()
    assert val_df["Date"].max() < evo_df["Date"].min()
    assert evo_df["Date"].max() < test_df["Date"].min()


def test_scaler_fit_only_on_training_data() -> None:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=100, freq="D"),
            "Open": np.arange(100, 200, dtype=float),
            "High": np.arange(100, 200, dtype=float) + 2,
            "Low": np.arange(100, 200, dtype=float) - 2,
            "Close": np.arange(100, 200, dtype=float),
            "Volume": np.ones(100, dtype=float) * 1000,
        }
    )
    df.iloc[60:, 1:] += 50
    train_df, val_df, _, _ = chronological_split(df, train_proportion=0.60, validation_proportion=0.15, evolution_proportion=0.15, test_proportion=0.10)["train"], chronological_split(df, train_proportion=0.60, validation_proportion=0.15, evolution_proportion=0.15, test_proportion=0.10)["validation"], None, None
    feature_cols = ["Open", "Close"]
    scaler = fit_scaler(train_df, feature_cols=feature_cols)
    train_scaled = scaler.transform(train_df[feature_cols].to_numpy())
    val_scaled = scaler.transform(val_df[feature_cols].to_numpy())
    assert np.allclose(np.mean(train_scaled, axis=0), 0.0, atol=1e-8)
    assert not np.allclose(np.mean(val_scaled, axis=0), 0.0)


def test_epoch_non_overlap(sample_market_df: pd.DataFrame) -> None:
    evo_df = sample_market_df.iloc[75:105].copy()
    epochs = create_walk_forward_epochs(evo_df, epoch_length=10)
    assert len(epochs) >= 2
    for left, right in zip(epochs, epochs[1:]):
        assert left["Date"].max() < right["Date"].min()


def test_deterministic_output(sample_market_df: pd.DataFrame) -> None:
    first = create_target(sample_market_df, target_mode="regression")
    second = create_target(sample_market_df.copy(), target_mode="regression")
    pd.testing.assert_series_equal(first, second)
