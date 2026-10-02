from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

tensorflow_required = pytest.mark.skipif(importlib.util.find_spec("tensorflow") is None, reason="TensorFlow is not installed in this environment.")

from artificial_life_trading_ecosystem.models.forecasting import GRU, LSTM, SimpleRNN
from artificial_life_trading_ecosystem.models.forecasting.base_forecaster import ForecastingModelConfig
from artificial_life_trading_ecosystem.models.manifest import compute_sha256, create_dataset_manifest


@pytest.fixture
def tiny_window_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(64, 30, 3)).astype(np.float32)
    y = rng.normal(size=(64,)).astype(np.float32)
    return X, y


@tensorflow_required
def test_simple_rnn_builds_successfully() -> None:
    cfg = ForecastingModelConfig(model_type="SimpleRNN", sequence_length=30, hidden_units=8, dropout=0.20, learning_rate=0.001, epochs=2, batch_size=8)
    model = SimpleRNN(config=cfg, input_shape=(30, 3), feature_names=["a", "b", "c"])
    model.build((30, 3))
    assert model.model is not None


@tensorflow_required
def test_lstm_builds_successfully() -> None:
    cfg = ForecastingModelConfig(model_type="LSTM", sequence_length=30, hidden_units=8, dropout=0.20, learning_rate=0.001, epochs=2, batch_size=8)
    model = LSTM(config=cfg, input_shape=(30, 3), feature_names=["a", "b", "c"])
    model.build((30, 3))
    assert model.model is not None


@tensorflow_required
def test_gru_builds_successfully() -> None:
    cfg = ForecastingModelConfig(model_type="GRU", sequence_length=30, hidden_units=8, dropout=0.20, learning_rate=0.001, epochs=2, batch_size=8)
    model = GRU(config=cfg, input_shape=(30, 3), feature_names=["a", "b", "c"])
    model.build((30, 3))
    assert model.model is not None


@tensorflow_required
def test_all_models_accept_identical_input_tensors(tiny_window_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, _ = tiny_window_data
    cfg = ForecastingModelConfig(sequence_length=30, hidden_units=8, dropout=0.20, learning_rate=0.001, epochs=2, batch_size=8)
    models = [SimpleRNN(cfg), LSTM(cfg), GRU(cfg)]
    for model in models:
        model.build((30, 3))
        preds = model.predict(X[:8])
        assert preds.shape == (8,)


@tensorflow_required
def test_all_models_produce_identical_output_shape_conventions(tiny_window_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, _ = tiny_window_data
    cfg = ForecastingModelConfig(sequence_length=30, hidden_units=8, dropout=0.20, learning_rate=0.001, epochs=2, batch_size=8)
    for cls in (SimpleRNN, LSTM, GRU):
        model = cls(config=cfg, input_shape=(30, 3), feature_names=["a", "b", "c"])
        model.build((30, 3))
        preds = model.predict(X[:4])
        assert preds.ndim == 1
        assert preds.shape[0] == 4


@tensorflow_required
def test_parameter_count_is_obtained_from_actual_model(tiny_window_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, _ = tiny_window_data
    cfg = ForecastingModelConfig(sequence_length=30, hidden_units=8, dropout=0.20, learning_rate=0.001, epochs=2, batch_size=8)
    model = SimpleRNN(cfg, input_shape=(30, 3), feature_names=["a", "b", "c"])
    model.build((30, 3))
    params = model.parameter_count()
    assert params["total"] > 0
    assert params["trainable"] > 0
    assert params["model_type"] == "SimpleRNN"


@tensorflow_required
def test_training_runs_on_tiny_synthetic_dataset(tiny_window_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = tiny_window_data
    cfg = ForecastingModelConfig(sequence_length=30, hidden_units=8, dropout=0.20, learning_rate=0.001, epochs=2, batch_size=8, early_stopping_patience=1)
    model = SimpleRNN(cfg, input_shape=(30, 3), feature_names=["a", "b", "c"])
    model.build((30, 3))
    history = model.train(X[:32], y[:32], X[32:], y[32:])
    assert "loss" in history
    assert len(history["loss"]) >= 1


@tensorflow_required
def test_prediction_works_after_training(tiny_window_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = tiny_window_data
    cfg = ForecastingModelConfig(sequence_length=30, hidden_units=8, dropout=0.20, learning_rate=0.001, epochs=2, batch_size=8, early_stopping_patience=1)
    model = GRU(cfg, input_shape=(30, 3), feature_names=["a", "b", "c"])
    model.build((30, 3))
    model.train(X[:32], y[:32], X[32:], y[32:])
    preds = model.predict(X[:4])
    assert preds.shape == (4,)
    assert np.isfinite(preds).all()


@tensorflow_required
def test_save_reload_preserves_predictions(tiny_window_data: tuple[np.ndarray, np.ndarray], tmp_path: Path) -> None:
    X, y = tiny_window_data
    cfg = ForecastingModelConfig(sequence_length=30, hidden_units=8, dropout=0.20, learning_rate=0.001, epochs=2, batch_size=8, early_stopping_patience=1)
    model = LSTM(cfg, input_shape=(30, 3), feature_names=["a", "b", "c"])
    model.build((30, 3))
    model.train(X[:32], y[:32], X[32:], y[32:])
    save_path = tmp_path / "lstm_saved"
    model.save(save_path)
    reloaded = LSTM(cfg, input_shape=(30, 3), feature_names=["a", "b", "c"])
    reloaded.load(save_path)
    original_preds = model.predict(X[:6])
    reloaded_preds = reloaded.predict(X[:6])
    assert np.allclose(original_preds, reloaded_preds, rtol=1e-4, atol=1e-4)


@tensorflow_required
def test_validation_evaluation_returns_mae_and_rmse(tiny_window_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = tiny_window_data
    cfg = ForecastingModelConfig(sequence_length=30, hidden_units=8, dropout=0.20, learning_rate=0.001, epochs=2, batch_size=8, early_stopping_patience=1)
    model = SimpleRNN(cfg, input_shape=(30, 3), feature_names=["a", "b", "c"])
    model.build((30, 3))
    model.train(X[:32], y[:32], X[32:], y[32:])
    metrics = model.evaluate(X[32:], y[32:])
    assert "mae" in metrics
    assert "rmse" in metrics
    assert metrics["mae"] >= 0.0
    assert metrics["rmse"] >= 0.0


@tensorflow_required
def test_final_test_data_is_never_required_by_training_api(tiny_window_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = tiny_window_data
    cfg = ForecastingModelConfig(sequence_length=30, hidden_units=8, dropout=0.20, learning_rate=0.001, epochs=2, batch_size=8, early_stopping_patience=1)
    model = GRU(cfg, input_shape=(30, 3), feature_names=["a", "b", "c"])
    model.build((30, 3))
    model.train(X[:32], y[:32], X[32:], y[32:])
    assert model.model is not None


@tensorflow_required
def test_fixed_seeds_produce_reproducible_model_construction(tiny_window_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = tiny_window_data
    cfg = ForecastingModelConfig(sequence_length=30, hidden_units=8, dropout=0.20, learning_rate=0.001, epochs=2, batch_size=8)
    model1 = SimpleRNN(cfg, input_shape=(30, 3), feature_names=["a", "b", "c"])
    model2 = SimpleRNN(cfg, input_shape=(30, 3), feature_names=["a", "b", "c"])
    model1.build((30, 3))
    model2.build((30, 3))
    p1 = model1.predict(X[:4])
    p2 = model2.predict(X[:4])
    assert np.allclose(p1, p2, rtol=1e-4, atol=1e-4)


def test_dataset_manifest_correctly_records_hash(tmp_path: Path) -> None:
    path = tmp_path / "raw_market.csv"
    payload = pd.DataFrame({"Date": ["2024-01-01", "2024-01-02"], "Open": [1.0, 2.0], "High": [2.0, 3.0], "Low": [0.5, 1.5], "Close": [1.5, 2.5], "Volume": [100, 200]})
    payload.to_csv(path, index=False)
    manifest = create_dataset_manifest(
        ticker="^NSEI",
        start_date="2024-01-01",
        end_date="2024-01-02",
        interval="1d",
        source="yfinance",
        raw_filename=path.name,
        raw_path=path,
        row_count=len(payload),
    )
    assert manifest["ticker"] == "^NSEI"
    assert manifest["sha256"] == compute_sha256(path)
    assert manifest["row_count"] == 2
