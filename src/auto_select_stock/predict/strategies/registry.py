import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsonschema

_STRATEGY_SCHEMA = {
    "type": "object",
    "required": ["name", "type", "params"],
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "type": {
            "type": "string",
            "enum": [
                "topk",
                "threshold",
                "long_short",
                "momentum_filter",
                "risk_parity",
                "mean_reversion",
                "confidence",
                "sector_neutral",
                "trailing_stop",
                "dual_thresh",
            ],
        },
        "params": {"type": "object"},
        "horizon": {"type": "string"},
    },
}


class StrategyRegistry:
    """Loads and validates strategy JSON config files.

    Scans a directory for ``.json`` files, validates each against the schema,
    and returns an instantiated ``BaseStrategy`` subclass.
    """

    def __init__(self, strategies_dir: Path):
        self.strategies_dir = Path(strategies_dir)
        self._cache: Dict[str, Dict[str, Any]] = {}

    def list_strategies(self) -> List[Dict[str, Any]]:
        """Return metadata for all valid strategy configs found."""
        results = []
        for path in sorted(self.strategies_dir.glob("*.json")):
            try:
                cfgs = self._load_one(path)
                # Handle both single-config and list-of-configs files
                if isinstance(cfgs, list):
                    for cfg in cfgs:
                        jsonschema.validate(cfg, _STRATEGY_SCHEMA)
                        results.append({"name": cfg["name"], "description": cfg.get("description", ""), "type": cfg["type"], "path": str(path)})
                else:
                    results.append({"name": cfgs["name"], "description": cfgs.get("description", ""), "type": cfgs["type"], "path": str(path)})
            except Exception:  # noqa: BLE001
                continue
        return results

    def get(self, name: str) -> Dict[str, Any]:
        """Return the raw config dict for *name* (cached after first load)."""
        if name not in self._cache:
            found = False
            for path in self.strategies_dir.glob("*.json"):
                cfgs = self._load_one(path)
                if isinstance(cfgs, list):
                    for cfg in cfgs:
                        try:
                            jsonschema.validate(cfg, _STRATEGY_SCHEMA)
                        except Exception:  # noqa: BLE001
                            continue
                        if cfg["name"] == name:
                            self._cache[name] = cfg
                            found = True
                            break
                else:
                    try:
                        jsonschema.validate(cfgs, _STRATEGY_SCHEMA)
                    except Exception:  # noqa: BLE001
                        continue
                    if cfgs["name"] == name:
                        self._cache[name] = cfgs
                        found = True
                        break
                if found:
                    break
            if not found:
                raise KeyError(f"No strategy named '{name}' found in {self.strategies_dir}")
        return self._cache[name]

    def _load_one(self, path: Path) -> Dict[str, Any]:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        # Don't validate here - handle lists vs single objects
        return cfg


def make_strategy(cfg: Dict[str, Any]) -> "BaseStrategy":
    """Instantiate a BaseStrategy subclass from a validated config dict."""
    from . import (  # noqa: F401 - dynamically register all strategy types
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

    strategy_map = {
        "topk": TopKStrategy,
        "threshold": ThresholdStrategy,
        "long_short": LongShortStrategy,
        "momentum_filter": MomentumFilterStrategy,
        "risk_parity": RiskParityStrategy,
        "mean_reversion": MeanReversionStrategy,
        "confidence": ConfidenceStrategy,
        "sector_neutral": SectorNeutralStrategy,
        "trailing_stop": TrailingStopStrategy,
        "dual_thresh": DualThreshStrategy,
    }
    cls = strategy_map.get(cfg["type"])
    if cls is None:
        raise ValueError(f"Unknown strategy type: {cfg['type']}")

    horizon = cfg.get("horizon", "1d")
    params = cfg.get("params", {})
    # Generate a 5-char unique tag from name + type + horizon + params
    digest = hashlib.md5(
        f"{cfg['name']}:{cfg['type']}:{horizon}:{json.dumps(params, sort_keys=True)}".encode()
    ).hexdigest()[:5]

    return cls(name=cfg["name"], horizon=horizon, tag=digest, **params)
