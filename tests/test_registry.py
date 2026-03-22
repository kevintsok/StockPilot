"""
Unit tests for the strategy registry (predict/strategies/registry.py).

Tests StrategyRegistry and make_strategy for loading and
validating JSON strategy configs.
"""

import json
from pathlib import Path

import pytest

from auto_select_stock.predict.strategies.registry import (
    StrategyRegistry,
    make_strategy,
    _STRATEGY_SCHEMA,
)
from auto_select_stock.predict.strategies import (
    TopKStrategy,
    ThresholdStrategy,
    LongShortStrategy,
    MomentumFilterStrategy,
    RiskParityStrategy,
    MeanReversionStrategy,
    ConfidenceStrategy,
    SectorNeutralStrategy,
    TrailingStopStrategy,
    DualThreshStrategy,
)
import jsonschema


class TestStrategySchema:
    """Tests for the JSON schema used to validate strategy configs."""

    def test_valid_topk_config(self):
        """Valid topk config passes schema validation."""
        config = {
            "name": "TestTopK",
            "type": "topk",
            "params": {"top_k": 5, "allow_short": False},
        }
        jsonschema.validate(config, _STRATEGY_SCHEMA)

    def test_valid_dual_thresh_config(self):
        """Valid dual_thresh config passes schema validation."""
        config = {
            "name": "TestDualThresh",
            "type": "dual_thresh",
            "params": {
                "upper_thresh": 0.02,
                "lower_thresh": -0.01,
                "top_k": 5,
            },
        }
        jsonschema.validate(config, _STRATEGY_SCHEMA)

    def test_missing_required_field_raises(self):
        """Config missing 'name' raises ValidationError."""
        config = {
            "type": "topk",
            "params": {"top_k": 5},
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(config, _STRATEGY_SCHEMA)

    def test_invalid_type_passes_schema_validation(self):
        """Schema validation passes for unknown type (schema only checks type is string)."""
        config = {
            "name": "BadType",
            "type": "unknown_strategy",
            "params": {},
        }
        # Schema doesn't validate enum, only checks type is string
        jsonschema.validate(config, _STRATEGY_SCHEMA)  # Should not raise


class TestStrategyRegistryInit:
    """Tests for StrategyRegistry initialization."""

    def test_init_with_nonexistent_dir(self, tmp_path):
        """StrategyRegistry handles nonexistent directory gracefully."""
        registry = StrategyRegistry(tmp_path / "nonexistent")
        result = registry.list_strategies()
        assert result == []


class TestStrategyRegistryListStrategies:
    """Tests for StrategyRegistry.list_strategies()."""

    def test_list_strategies_finds_json_files(self, tmp_path):
        """list_strategies discovers strategy configs in directory."""
        strategy_dir = tmp_path / "strategies"
        strategy_dir.mkdir()
        config_path = strategy_dir / "test_strategy.json"
        config = [{
            "name": "TestTopK",
            "description": "Test strategy",
            "type": "topk",
            "params": {"top_k": 5},
        }]
        config_path.write_text(json.dumps(config), encoding="utf-8")

        registry = StrategyRegistry(strategy_dir)
        result = registry.list_strategies()

        assert len(result) == 1
        assert result[0]["name"] == "TestTopK"
        assert result[0]["type"] == "topk"

    def test_list_strategies_handles_invalid_json(self, tmp_path):
        """list_strategies gracefully skips invalid JSON files."""
        strategy_dir = tmp_path / "strategies"
        strategy_dir.mkdir()
        bad_path = strategy_dir / "bad.json"
        bad_path.write_text("not valid json{{", encoding="utf-8")

        registry = StrategyRegistry(strategy_dir)
        result = registry.list_strategies()
        assert result == []

    def test_list_strategies_handles_malformed_json(self, tmp_path):
        """list_strategies skips files with malformed JSON."""
        strategy_dir = tmp_path / "strategies"
        strategy_dir.mkdir()
        config_path = strategy_dir / "bad.json"
        config_path.write_text("not valid json{{", encoding="utf-8")

        registry = StrategyRegistry(strategy_dir)
        result = registry.list_strategies()
        assert result == []


class TestStrategyRegistryGet:
    """Tests for StrategyRegistry.get()."""

    def test_get_existing_strategy(self, tmp_path):
        """get() returns config for an existing strategy."""
        strategy_dir = tmp_path / "strategies"
        strategy_dir.mkdir()
        config_path = strategy_dir / "strategies.json"
        config = [{
            "name": "MyTopK",
            "description": "My test strategy",
            "type": "topk",
            "params": {"top_k": 10},
        }]
        config_path.write_text(json.dumps(config), encoding="utf-8")

        registry = StrategyRegistry(strategy_dir)
        result = registry.get("MyTopK")

        assert result["name"] == "MyTopK"
        assert result["params"]["top_k"] == 10

    def test_get_nonexistent_raises(self, tmp_path):
        """get() raises KeyError for unknown strategy name."""
        strategy_dir = tmp_path / "strategies"
        strategy_dir.mkdir()

        registry = StrategyRegistry(strategy_dir)
        with pytest.raises(KeyError, match="No strategy named"):
            registry.get("NonExistent")


class TestMakeStrategy:
    """Tests for make_strategy() factory function."""

    def test_make_topk_strategy(self):
        """make_strategy creates TopKStrategy from config."""
        config = {
            "name": "TestTopK",
            "type": "topk",
            "params": {"top_k": 5, "allow_short": False},
        }
        strategy = make_strategy(config)
        assert isinstance(strategy, TopKStrategy)
        assert strategy.top_k == 5
        assert strategy.allow_short is False

    def test_make_threshold_strategy(self):
        """make_strategy creates ThresholdStrategy from config."""
        config = {
            "name": "TestThresh",
            "type": "threshold",
            "params": {"top_k": 10, "threshold": 0.01},
        }
        strategy = make_strategy(config)
        assert isinstance(strategy, ThresholdStrategy)
        assert strategy.threshold == 0.01

    def test_make_long_short_strategy(self):
        """make_strategy creates LongShortStrategy from config."""
        config = {
            "name": "TestLS",
            "type": "long_short",
            "params": {"top_pct": 0.1, "allow_short": True},
        }
        strategy = make_strategy(config)
        assert isinstance(strategy, LongShortStrategy)
        assert strategy.top_pct == 0.1

    def test_make_momentum_filter_strategy(self):
        """make_strategy creates MomentumFilterStrategy from config."""
        config = {
            "name": "TestMom",
            "type": "momentum_filter",
            "params": {"top_k": 5, "lookback": 10},
        }
        strategy = make_strategy(config)
        assert isinstance(strategy, MomentumFilterStrategy)
        assert strategy.lookback == 10

    def test_make_risk_parity_strategy(self):
        """make_strategy creates RiskParityStrategy from config."""
        config = {
            "name": "TestRP",
            "type": "risk_parity",
            "params": {"top_k": 5, "vol_lookback": 20},
        }
        strategy = make_strategy(config)
        assert isinstance(strategy, RiskParityStrategy)
        assert strategy.vol_lookback == 20

    def test_make_mean_reversion_strategy(self):
        """make_strategy creates MeanReversionStrategy from config."""
        config = {
            "name": "TestMR",
            "type": "mean_reversion",
            "params": {"top_k": 5},
        }
        strategy = make_strategy(config)
        assert isinstance(strategy, MeanReversionStrategy)

    def test_make_confidence_strategy(self):
        """make_strategy creates ConfidenceStrategy from config."""
        config = {
            "name": "TestConf",
            "type": "confidence",
            "params": {"top_k": 5, "min_confidence": 0.005},
        }
        strategy = make_strategy(config)
        assert isinstance(strategy, ConfidenceStrategy)
        assert strategy.min_confidence == 0.005

    def test_make_sector_neutral_strategy(self):
        """make_strategy creates SectorNeutralStrategy from config."""
        config = {
            "name": "TestSN",
            "type": "sector_neutral",
            "params": {"top_pct": 0.1},
        }
        strategy = make_strategy(config)
        assert isinstance(strategy, SectorNeutralStrategy)

    def test_make_trailing_stop_strategy(self):
        """make_strategy creates TrailingStopStrategy from config."""
        config = {
            "name": "TestTS",
            "type": "trailing_stop",
            "params": {"top_k": 5, "stop_loss": 0.05},
        }
        strategy = make_strategy(config)
        assert isinstance(strategy, TrailingStopStrategy)
        assert strategy.stop_loss == 0.05

    def test_make_dual_thresh_strategy(self):
        """make_strategy creates DualThreshStrategy from config."""
        config = {
            "name": "TestDT",
            "type": "dual_thresh",
            "params": {"upper_thresh": 0.02, "lower_thresh": -0.01, "top_k": 5},
        }
        strategy = make_strategy(config)
        assert isinstance(strategy, DualThreshStrategy)
        assert strategy.upper_thresh == 0.02
        assert strategy.lower_thresh == -0.01

    def test_make_strategy_with_horizon(self):
        """make_strategy respects horizon parameter from config."""
        config = {
            "name": "TestHorizon",
            "type": "topk",
            "horizon": "5d",
            "params": {"top_k": 5},
        }
        strategy = make_strategy(config)
        assert strategy.horizon == "5d"

    def test_make_unknown_type_raises(self):
        """make_strategy raises ValueError for unknown type."""
        config = {
            "name": "Bad",
            "type": "not_a_real_type",
            "params": {},
        }
        with pytest.raises(ValueError, match="Unknown strategy type"):
            make_strategy(config)

    def test_make_strategy_generates_tag(self):
        """make_strategy generates a tag from config hash."""
        config = {
            "name": "TestTag",
            "type": "topk",
            "params": {"top_k": 5},
        }
        strategy = make_strategy(config)
        assert strategy.tag is not None
        assert len(strategy.tag) == 5  # MD5 digest truncated to 5 chars
