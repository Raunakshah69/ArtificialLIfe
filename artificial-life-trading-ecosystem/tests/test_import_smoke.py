"""Minimal smoke test ensuring the package imports successfully."""

from artificial_life_trading_ecosystem import settings
from artificial_life_trading_ecosystem.utils import get_logger, set_deterministic_seed


def test_package_imports() -> None:
    assert settings.project_name == "artificial-life-trading-ecosystem"
    assert callable(get_logger)
    assert callable(set_deterministic_seed)
