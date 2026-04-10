import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import jsonschema

from .base import _STRATEGY_TYPE_REGISTRY, _validate_params

logger = logging.getLogger(__name__)

_STRATEGY_SCHEMA = {
    "type": "object",
    "required": ["name", "type", "params"],
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "type": {"type": "string"},
        "params": {"type": "object"},
        "horizon": {"type": "string"},
    },
}


class StrategyRegistry:
    """Loads and validates strategy JSON config files.

    Scans a directory for ``.json`` files, validates each against the schema,
    and returns an instantiated ``BaseStrategy`` subclass.

    All configs are pre-loaded on first access and cached for the lifetime
    of the registry instance. ``list_strategies()`` reports broken configs
    via logging instead of silently skipping them.
    """

    def __init__(self, strategies_dir: Path):
        self.strategies_dir = Path(strategies_dir)
        self._cache: Dict[str, Dict[str, Any]] = {}   # name -> validated config
        self._path_cache: Dict[str, Path] = {}        # name -> source file path
        self._loaded: bool = False

    def _ensure_loaded(self) -> None:
        """Scan and validate all JSON files once, populating the cache."""
        if self._loaded:
            return
        self._cache.clear()
        self._path_cache.clear()
        for path in sorted(self.strategies_dir.glob("*.json")):
            try:
                cfgs = self._load_one(path)
                items = cfgs if isinstance(cfgs, list) else [cfgs]
                for cfg in items:
                    try:
                        jsonschema.validate(cfg, _STRATEGY_SCHEMA)
                        _validate_params(cfg.get("params", {}))
                    except Exception as exc:
                        logger.warning("Skipping invalid strategy config %s: %s", path, exc)
                        continue
                    self._cache[cfg["name"]] = cfg
                    self._path_cache[cfg["name"]] = path
            except Exception as exc:
                logger.warning("Failed to load strategy file %s: %s", path, exc)
        self._loaded = True

    def list_strategies(self) -> List[Dict[str, Any]]:
        """Return metadata for all valid strategy configs found.

        Logs a warning for each config that fails to load or validate.
        """
        self._ensure_loaded()
        return [
            {
                "name": cfg["name"],
                "description": cfg.get("description", ""),
                "type": cfg["type"],
                "path": str(self._path_cache.get(cfg["name"], self.strategies_dir / "~.json")),
            }
            for cfg in self._cache.values()
        ]

    def get(self, name: str) -> Dict[str, Any]:
        """Return the raw config dict for *name* (cached after first load)."""
        self._ensure_loaded()
        if name not in self._cache:
            raise KeyError(f"No strategy named '{name}' found in {self.strategies_dir}")
        return self._cache[name]

    def _load_one(self, path: Path) -> Dict[str, Any]:
        with open(path, encoding="utf-8") as f:
            return json.load(f)


def make_strategy(cfg: Dict[str, Any]) -> "BaseStrategy":
    """Instantiate a BaseStrategy subclass from a validated config dict.

    Strategy type is resolved via auto-discovery: subclasses of BaseStrategy
    are registered automatically by the ``_AutoDiscoveryMeta`` metaclass,
    keyed by their lowercase class name with 'Strategy' removed
    (e.g. ``TopKStrategy`` -> ``"topk"``).

    Custom strategies with ``custom_*`` type prefix are loaded from
    ``custom_strategies`` module (e.g. ``custom_adaptive_k`` ->
    ``AdaptiveKStrategy``).
    """
    # Trigger import of all strategy subclasses so the registry is populated.
    # Importing the package runs the module-level subclass registration.
    from . import (  # noqa: F401 - side-effect: registers all subclasses
        TopKStrategy,
        ThresholdStrategy,
        LongShortStrategy,
        MomentumFilterStrategy,
        RiskParityStrategy,
        MeanReversionStrategy,
        ConfidenceStrategy,
        ConfidenceStopStrategy,
        SectorNeutralStrategy,
        TrailingStopStrategy,
        DualThreshStrategy,
    )

    strategy_type = cfg["type"]
    strategy_cls = _STRATEGY_TYPE_REGISTRY.get(strategy_type)

    # Handle custom_* strategies: load from custom_strategies module
    if strategy_cls is None and strategy_type.startswith("custom_"):
        from . import custom_strategies as cs
        # Derive class name: custom_adaptive_k -> AdaptiveKStrategy
        suffix = strategy_type[len("custom_"):]  # e.g. "adaptive_k"
        parts = suffix.split("_")
        # Handle known acronyms: rsi -> RSI
        acronym_map = {"rsi": "RSI", "atr": "ATR", "bb": "BB", "ma": "MA"}
        class_parts = []
        for part in parts:
            if part.lower() in acronym_map:
                class_parts.append(acronym_map[part.lower()])
            else:
                class_parts.append(part.capitalize())
        class_name = "".join(class_parts) + "Strategy"
        strategy_cls = getattr(cs, class_name, None)
        if strategy_cls is None:
            known = sorted(_STRATEGY_TYPE_REGISTRY.keys())
            raise ValueError(
                f"Custom strategy class {class_name!r} not found in custom_strategies. "
                f"Known custom types need corresponding {class_name} class."
            )
        # Register it so future lookups work
        _STRATEGY_TYPE_REGISTRY[strategy_type] = strategy_cls

    if strategy_cls is None:
        known = sorted(_STRATEGY_TYPE_REGISTRY.keys())
        raise ValueError(
            f"Unknown strategy type: {strategy_type!r}. "
            f"Known types: {known}"
        )

    horizon = cfg.get("horizon", "1d")
    params = cfg.get("params", {})
    _validate_params(params)

    # Generate a 5-char unique tag from name + type + horizon + params
    digest = hashlib.md5(
        f"{cfg['name']}:{cfg['type']}:{horizon}:{json.dumps(params, sort_keys=True)}".encode()
    ).hexdigest()[:5]

    return strategy_cls(name=cfg["name"], horizon=horizon, tag=digest, **params)
