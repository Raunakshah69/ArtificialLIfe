"""Deterministic decision-head genomes for trading agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DecisionGenome:
    """Genome that encodes the compact decision head and trading parameters."""

    context_length: int = 5
    hidden_units: int = 8
    activation: str = "tanh"
    input_to_hidden: np.ndarray = field(default_factory=lambda: np.zeros((6, 8), dtype=np.float32))
    hidden_bias: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float32))
    hidden_to_output: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float32))
    output_bias: float = 0.0
    signal_threshold: float = 0.20
    position_size: float = 0.25
    stop_loss: float = 0.03
    take_profit: float = 0.06
    transaction_cost: float = 0.001
    mutation_rate: float = 0.0
    weights: dict[str, np.ndarray | float] = field(default_factory=dict)
    trading_params: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.context_length = int(self.context_length)
        self.hidden_units = int(self.hidden_units)
        self.activation = str(self.activation).lower()
        self.input_to_hidden = np.asarray(self.input_to_hidden, dtype=np.float32)
        self.hidden_bias = np.asarray(self.hidden_bias, dtype=np.float32)
        self.hidden_to_output = np.asarray(self.hidden_to_output, dtype=np.float32)
        self.output_bias = float(self.output_bias)

        if self.input_to_hidden.shape != (self.context_length + 1, self.hidden_units):
            self.input_to_hidden = np.zeros((self.context_length + 1, self.hidden_units), dtype=np.float32)
        if self.hidden_bias.shape != (self.hidden_units,):
            self.hidden_bias = np.zeros(self.hidden_units, dtype=np.float32)
        if self.hidden_to_output.shape != (self.hidden_units,):
            self.hidden_to_output = np.zeros(self.hidden_units, dtype=np.float32)

        self.weights = {
            "input_to_hidden": self.input_to_hidden.copy(),
            "hidden_bias": self.hidden_bias.copy(),
            "hidden_to_output": self.hidden_to_output.copy(),
            "output_bias": float(self.output_bias),
        }
        self.trading_params = {
            "signal_threshold": float(self.signal_threshold),
            "position_size": float(self.position_size),
            "stop_loss": float(self.stop_loss),
            "take_profit": float(self.take_profit),
            "transaction_cost": float(self.transaction_cost),
        }

    @property
    def flattened_length(self) -> int:
        return int(self.input_to_hidden.size + self.hidden_bias.size + self.hidden_to_output.size + 1 + 5)

    def flatten(self) -> np.ndarray:
        values = [
            self.input_to_hidden.reshape(-1),
            self.hidden_bias.reshape(-1),
            self.hidden_to_output.reshape(-1),
            np.asarray([self.output_bias], dtype=np.float32),
            np.asarray(
                [
                    self.signal_threshold,
                    self.position_size,
                    self.stop_loss,
                    self.take_profit,
                    self.transaction_cost,
                ],
                dtype=np.float32,
            ),
        ]
        return np.concatenate([v.reshape(-1) for v in values]).astype(np.float32)

    @classmethod
    def from_default(
        cls,
        *,
        context_length: int = 5,
        hidden_units: int = 8,
        activation: str = "tanh",
        signal_threshold: float = 0.20,
        position_size: float = 0.25,
        stop_loss: float = 0.03,
        take_profit: float = 0.06,
        transaction_cost: float = 0.001,
        seed: int = 42,
    ) -> "DecisionGenome":
        rng = np.random.default_rng(seed)
        input_dim = context_length + 1
        input_scale = np.sqrt(2.0 / (input_dim + hidden_units))
        output_scale = np.sqrt(2.0 / (hidden_units + 1))
        input_to_hidden = rng.normal(0.0, input_scale, size=(input_dim, hidden_units)).astype(np.float32)
        hidden_bias = np.zeros(hidden_units, dtype=np.float32)
        hidden_to_output = rng.normal(0.0, output_scale, size=(hidden_units,)).astype(np.float32)
        output_bias = 0.0
        return cls(
            context_length=context_length,
            hidden_units=hidden_units,
            activation=activation,
            input_to_hidden=input_to_hidden,
            hidden_bias=hidden_bias,
            hidden_to_output=hidden_to_output,
            output_bias=output_bias,
            signal_threshold=float(signal_threshold),
            position_size=float(position_size),
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            transaction_cost=float(transaction_cost),
            mutation_rate=0.0,
        )

    @classmethod
    def from_flattened(cls, flat: np.ndarray, *, context_length: int = 5, hidden_units: int = 8) -> "DecisionGenome":
        flat_arr = np.asarray(flat, dtype=np.float32).reshape(-1)
        input_dim = context_length + 1
        w_in_size = input_dim * hidden_units
        hidden_bias_size = hidden_units
        hidden_to_output_size = hidden_units
        idx = 0
        w_in = flat_arr[idx : idx + w_in_size].reshape(input_dim, hidden_units)
        idx += w_in_size
        hb = flat_arr[idx : idx + hidden_bias_size]
        idx += hidden_bias_size
        h2o = flat_arr[idx : idx + hidden_to_output_size]
        idx += hidden_to_output_size
        output_bias = float(flat_arr[idx])
        idx += 1
        threshold, position_size, stop_loss, take_profit, tx = flat_arr[idx : idx + 5]
        return cls(
            context_length=context_length,
            hidden_units=hidden_units,
            activation="tanh",
            input_to_hidden=w_in,
            hidden_bias=hb,
            hidden_to_output=h2o,
            output_bias=output_bias,
            signal_threshold=float(threshold),
            position_size=float(position_size),
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            transaction_cost=float(tx),
        )


__all__ = ["DecisionGenome"]
