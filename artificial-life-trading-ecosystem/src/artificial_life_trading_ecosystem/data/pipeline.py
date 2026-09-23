"""Market data acquisition and preparation pipeline for Milestone 1."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - optional until the data source is used
    yf = None

from artificial_life_trading_ecosystem.config import settings
from artificial_life_trading_ecosystem.models.manifest import create_dataset_manifest


RAW_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
BASELINE_FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "daily_return",
    "high_low_range",
    "open_close_range",
    "rolling_volatility",
]


def _normalize_yfinance_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Flatten Yahoo Finance multi-index columns and enforce the OHLCV contract."""

    normalized = frame.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized = normalized.droplevel(0, axis=1)

    if "Date" not in normalized.columns and "index" in normalized.columns:
        normalized = normalized.rename(columns={"index": "Date"})

    if not {"Open", "High", "Low", "Close"}.issubset(normalized.columns):
        candidate_cols = [col for col in normalized.columns if str(col) not in {"Date", "index"}]
        if len(candidate_cols) >= 6:
            rename_map = {
                candidate_cols[0]: "Adj Close",
                candidate_cols[1]: "Close",
                candidate_cols[2]: "High",
                candidate_cols[3]: "Low",
                candidate_cols[4]: "Open",
                candidate_cols[5]: "Volume",
            }
            normalized = normalized.rename(columns=rename_map)

    if "Date" in normalized.columns:
        normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce")

    return normalized


def _get_settings(config: Any | None = None) -> Any:
    return config if config is not None else settings


def _resolve_raw_dir(raw_dir: str | Path | None = None, config: Any | None = None) -> Path:
    cfg = _get_settings(config)
    target = Path(raw_dir) if raw_dir is not None else Path(cfg.data_dir) / "raw"
    target.mkdir(parents=True, exist_ok=True)
    return target


def download_market_data(
    ticker: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    interval: str | None = None,
    raw_dir: str | Path | None = None,
    config: Any | None = None,
    save_to_disk: bool = True,
) -> pd.DataFrame:
    """Download NIFTY market data from Yahoo Finance and save it to the raw-data directory."""

    cfg = _get_settings(config)
    if yf is None:
        raise ImportError("yfinance is required for market data acquisition.")

    ticker_name = ticker or cfg.ticker
    start = start_date or cfg.start_date
    end = end_date or cfg.end_date
    freq = interval or cfg.interval

    raw_df = yf.download(
        ticker_name,
        start=start,
        end=end,
        interval=freq,
        progress=False,
        auto_adjust=False,
        threads=True,
    )

    if raw_df.empty:
        raise ValueError(f"No market data returned for ticker={ticker_name!r} in {start} to {end}.")

    raw_df = _normalize_yfinance_columns(raw_df)
    raw_df = raw_df.reset_index().rename(columns={"index": "Date"}) if "Date" not in raw_df.columns else raw_df.copy()
    if "Date" in raw_df.columns:
        raw_df["Date"] = pd.to_datetime(raw_df["Date"])

    if save_to_disk:
        raw_path = _resolve_raw_dir(raw_dir, cfg)
        file_name = f"{ticker_name}_{start}_{end}_{freq}.csv"
        raw_file = raw_path / file_name
        raw_df.to_csv(raw_file, index=False)

        manifest = create_dataset_manifest(
            ticker=ticker_name,
            start_date=start,
            end_date=end,
            interval=freq,
            source="yfinance",
            raw_filename=file_name,
            raw_path=raw_file,
            row_count=len(raw_df),
        )
        manifest_path = raw_path / f"{ticker_name}_{start}_{end}_{freq}_manifest.json"
        manifest_path.write_text(__import__("json").dumps(manifest, indent=2), encoding="utf-8")

    return raw_df


def load_raw_market_data(
    path: str | Path | None = None,
    *,
    ticker: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    interval: str | None = None,
    raw_dir: str | Path | None = None,
    config: Any | None = None,
) -> pd.DataFrame:
    """Load raw market data from disk or download it when no local file is available."""

    cfg = _get_settings(config)
    if path is not None:
        frame = pd.read_csv(path)
        return _normalize_yfinance_columns(frame)

    raw_path = Path(raw_dir) if raw_dir is not None else Path(cfg.data_dir) / "raw"
    ticker_name = ticker or cfg.ticker
    start = start_date or cfg.start_date
    end = end_date or cfg.end_date
    freq = interval or cfg.interval
    candidate = raw_path / f"{ticker_name}_{start}_{end}_{freq}.csv"

    if candidate.exists():
        frame = pd.read_csv(candidate)
        return _normalize_yfinance_columns(frame)

    return download_market_data(
        ticker=ticker_name,
        start_date=start,
        end_date=end,
        interval=freq,
        raw_dir=raw_path,
        config=cfg,
        save_to_disk=True,
    )


def validate_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """Check that raw OHLCV data is valid, sorted, and free from obvious leakage-prone issues."""

    if df is None or df.empty:
        raise ValueError("Market data is empty.")

    validated = df.copy()
    required_columns = ["Open", "High", "Low", "Close"]
    missing = [name for name in required_columns if name not in validated.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "Date" in validated.columns:
        validated["Date"] = pd.to_datetime(validated["Date"], errors="raise")
        validated = validated.sort_values("Date", kind="mergesort").reset_index(drop=True)

        duplicate_dates = validated["Date"][validated["Date"].duplicated()].tolist()
        if duplicate_dates:
            raise ValueError(f"Duplicate timestamps detected: {duplicate_dates[:10]}")

    missing_ohlc = validated[["Open", "High", "Low", "Close"]].isna().any(axis=1)
    if missing_ohlc.any():
        row_display = validated.loc[missing_ohlc, ["Date", "Open", "High", "Low", "Close"]] if "Date" in validated.columns else validated.loc[missing_ohlc, ["Open", "High", "Low", "Close"]]
        raise ValueError(f"Missing OHLC values detected in rows: {row_display.head().to_dict(orient='records')}")

    numeric_cols = ["Open", "High", "Low", "Close"]
    for col in numeric_cols:
        validated[col] = pd.to_numeric(validated[col], errors="raise")

    if "Volume" in validated.columns:
        validated["Volume"] = pd.to_numeric(validated["Volume"], errors="coerce")
        negative = validated["Volume"] < 0
        if negative.any():
            row_display = validated.loc[negative, ["Date", "Volume"]] if "Date" in validated.columns else validated.loc[negative, ["Volume"]]
            raise ValueError(f"Negative volume values detected: {row_display.head().to_dict(orient='records')}")
        zero_volume = validated["Volume"] == 0
        if zero_volume.any():
            warnings.warn("Zero-volume rows are retained; they may represent exchange-quiet periods and are not treated as invalid market data.", UserWarning)

    row_level_ok = (
        (validated["Low"] <= validated["High"])
        & (validated["Low"] <= validated["Open"])
        & (validated["Low"] <= validated["Close"])
        & (validated["High"] >= validated["Open"])
        & (validated["High"] >= validated["Close"])
    )
    if not row_level_ok.all():
        row_display = validated.loc[~row_level_ok, ["Date", "Open", "High", "Low", "Close"]] if "Date" in validated.columns else validated.loc[~row_level_ok, ["Open", "High", "Low", "Close"]]
        raise ValueError(f"Impossible OHLC relationships detected: {row_display.head().to_dict(orient='records')}")

    if "Adj Close" in validated.columns:
        validated["Adj Close"] = pd.to_numeric(validated["Adj Close"], errors="coerce")
        if validated["Adj Close"].isna().any():
            warnings.warn("Adj Close contains nulls; preserving the column but leaving the data otherwise unchanged.", UserWarning)

    return validated


def clean_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw data and ensure the market table is valid for downstream processing."""

    validated = validate_market_data(df)
    for col in ["Open", "High", "Low", "Close"]:
        validated[col] = pd.to_numeric(validated[col], errors="raise")
    if "Volume" in validated.columns:
        validated["Volume"] = pd.to_numeric(validated["Volume"], errors="coerce")
    return validated


def build_feature_set(
    df: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    rolling_window: int = 5,
) -> pd.DataFrame:
    """Create a compact baseline feature set for a market dataset."""

    cleaned = clean_market_data(df)
    if feature_columns is None:
        feature_columns = ["Open", "High", "Low", "Close", "Volume"]

    features = cleaned[feature_columns].copy()
    close = cleaned["Close"].astype(float)
    close_safe = close.replace(0.0, np.nan)

    features["daily_return"] = close.pct_change().fillna(0.0)
    features["high_low_range"] = ((cleaned["High"] - cleaned["Low"]) / close_safe).fillna(0.0)
    features["open_close_range"] = ((cleaned["Open"] - close) / close_safe).fillna(0.0)
    features["rolling_volatility"] = close.pct_change().rolling(window=rolling_window, min_periods=1).std().fillna(0.0)

    return features


def create_target(df: pd.DataFrame, target_mode: str = "regression") -> pd.Series:
    """Create the next-period target for regression or classification tasks."""

    if "Close" not in df.columns:
        raise ValueError("The supplied frame does not contain a Close column required for target generation.")

    if target_mode not in {"regression", "classification"}:
        raise ValueError("target_mode must be either 'regression' or 'classification'.")

    working = df.copy()
    if {"Open", "High", "Low", "Close"}.issubset(working.columns):
        working = clean_market_data(working)

    close = working["Close"].astype(float)
    next_close = close.shift(-1)

    if target_mode == "regression":
        pct_change = (next_close - close) / close.replace(0.0, np.nan)
        target = pct_change.fillna(0.0)
    else:
        target = (next_close > close).astype(int)

    target.name = "target"
    return target


def chronological_split(
    df: pd.DataFrame,
    *,
    train_proportion: float = 0.60,
    validation_proportion: float = 0.15,
    evolution_proportion: float = 0.15,
    test_proportion: float = 0.10,
    config: Any | None = None,
) -> dict[str, pd.DataFrame]:
    """Split a chronological dataset into train, validation, development, and test partitions."""

    cfg = _get_settings(config)
    cleaned = clean_market_data(df)
    if not np.isclose(train_proportion + validation_proportion + evolution_proportion + test_proportion, 1.0):
        raise ValueError("Split proportions must sum to 1.0.")

    train_pct = train_proportion or cfg.train_proportion if hasattr(cfg, "train_proportion") else train_proportion
    validation_pct = validation_proportion or cfg.validation_proportion if hasattr(cfg, "validation_proportion") else validation_proportion
    evolution_pct = evolution_proportion or cfg.evolution_proportion if hasattr(cfg, "evolution_proportion") else evolution_proportion
    test_pct = test_proportion or cfg.test_proportion if hasattr(cfg, "test_proportion") else test_proportion

    total = len(cleaned)
    train_end = int(total * train_pct)
    validation_end = int(total * (train_pct + validation_pct))
    evolution_end = int(total * (train_pct + validation_pct + evolution_pct))

    train_df = cleaned.iloc[:train_end].copy()
    validation_df = cleaned.iloc[train_end:validation_end].copy()
    evolution_df = cleaned.iloc[validation_end:evolution_end].copy()
    test_df = cleaned.iloc[evolution_end:].copy()

    if train_df.empty or validation_df.empty or evolution_df.empty or test_df.empty:
        raise ValueError("One or more chronological splits are empty. Adjust the split proportions.")

    return {
        "train": train_df,
        "validation": validation_df,
        "evolution": evolution_df,
        "test": test_df,
    }


def fit_scaler(
    train_df: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    feature_cols: list[str] | None = None,
    scaler_type: str = "StandardScaler",
    config: Any | None = None,
):
    """Fit a scaler using the forecasting training segment only."""

    cfg = _get_settings(config)
    selected = feature_cols if feature_cols is not None else feature_columns
    if selected is None:
        selected = ["Open", "High", "Low", "Close", "Volume"]
    scaler_name = scaler_type or getattr(cfg, "scaler_type", "StandardScaler")

    if scaler_name == "StandardScaler":
        scaler = StandardScaler()
    elif scaler_name == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif scaler_name == "RobustScaler":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unsupported scaler type: {scaler_name}")

    train_values = train_df[selected].to_numpy(dtype=float)
    scaler.fit(train_values)
    return scaler


def transform_datasets(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame | None,
    evolution_df: pd.DataFrame | None,
    test_df: pd.DataFrame | None,
    *,
    scaler,
    feature_columns: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Apply a fitted scaler to time-series partitions without re-fitting on any hold-out segment."""

    selected = feature_columns or ["Open", "High", "Low", "Close", "Volume"]
    transformed: dict[str, pd.DataFrame] = {}

    for name, frame in {
        "train": train_df,
        "validation": validation_df,
        "evolution": evolution_df,
        "test": test_df,
    }.items():
        if frame is None:
            continue
        scaled_values = scaler.transform(frame[selected].to_numpy(dtype=float))
        scaled_df = pd.DataFrame(scaled_values, columns=selected, index=frame.index)
        transformed[name] = frame.copy()
        for col in selected:
            transformed[name][col] = scaled_df[col].to_numpy()

    return transformed


def create_sliding_windows(
    df: pd.DataFrame,
    targets: pd.Series,
    *,
    window_length: int = 30,
    feature_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding windows from chronological data with target alignment based on the period after the window."""

    if window_length <= 0:
        raise ValueError("window_length must be a positive integer.")

    cleaned = clean_market_data(df)
    if len(cleaned) != len(targets):
        raise ValueError("The number of rows in the feature frame and target series must match.")

    feature_frame = build_feature_set(cleaned if feature_columns is None else cleaned, feature_columns=feature_columns)
    if feature_columns is not None:
        feature_frame = feature_frame[feature_columns].copy()

    valid_windows: list[np.ndarray] = []
    valid_targets: list[float] = []
    max_start = len(feature_frame) - window_length + 1

    for start_index in range(max_start):
        end_index = start_index + window_length - 1
        target_value = targets.iloc[end_index]
        if pd.isna(target_value):
            continue
        valid_windows.append(feature_frame.iloc[start_index : start_index + window_length].to_numpy(dtype=float))
        valid_targets.append(float(target_value))

    if not valid_windows:
        raise ValueError("No valid sliding windows could be created for the supplied data.")

    X = np.stack(valid_windows)
    y = np.asarray(valid_targets, dtype=float)
    return X, y


def create_walk_forward_epochs(
    df: pd.DataFrame,
    *,
    epoch_length: int = 20,
) -> list[pd.DataFrame]:
    """Create successive, non-overlapping chronological epochs for walk-forward evaluation."""

    cleaned = clean_market_data(df)
    if epoch_length <= 0:
        raise ValueError("epoch_length must be a positive integer.")

    if len(cleaned) < epoch_length:
        return [cleaned.copy()]

    epochs: list[pd.DataFrame] = []
    for start_index in range(0, len(cleaned), epoch_length):
        epochs.append(cleaned.iloc[start_index : start_index + epoch_length].copy())
    return epochs


def _config_from_settings(config: Any | None = None) -> Any:
    return _get_settings(config)


__all__ = [
    "BASELINE_FEATURES",
    "RAW_COLUMNS",
    "build_feature_set",
    "clean_market_data",
    "chronological_split",
    "create_sliding_windows",
    "create_target",
    "create_walk_forward_epochs",
    "download_market_data",
    "fit_scaler",
    "load_raw_market_data",
    "transform_datasets",
    "validate_market_data",
]
