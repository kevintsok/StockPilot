"""
Virtual Portfolio: daily simulation of all strategies.

Tracks per-strategy positions, equity curves, and metrics, persisting state to JSON.
Runs daily after market close (15:05 Mon-Fri via cron).
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import DATA_DIR
from ..data.storage import _connect, list_symbols
from ..predict.backtest import filter_a_share_symbols
from ..predict.inference import PricePredictor
from ..predict.strategies.base import Signal
from ..predict.strategies.registry import StrategyRegistry, make_strategy

logger = logging.getLogger(__name__)

_MAX_WORKERS = 16   # Parallel inference threads

_INITIAL_CAPITAL = 100_000.0
_LOT_SIZE = 100
_STATE_FILE = Path("data/virtual_portfolio_state.json")

# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------


@dataclass
class VirtualPosition:
    symbol: str
    shares: int
    entry_price: float
    entry_date: str


@dataclass
class DailySnapshot:
    date: str
    total_value: float
    cash: float
    holdings_value: float


@dataclass
class Trade:
    date: str
    symbol: str
    action: str
    price: float
    shares: int
    amount: float


@dataclass
class PortfolioMetrics:
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    ann_return: float
    num_trades: int
    current_positions: int


# ----------------------------------------------------------------------
# StrategyPortfolio: state for one strategy
# ----------------------------------------------------------------------


class StrategyPortfolio:
    def __init__(
        self,
        name: str,
        tag: str = "",
        positions: Optional[Dict[str, VirtualPosition]] = None,
        equity_curve: Optional[List[DailySnapshot]] = None,
        trade_log: Optional[List[Trade]] = None,
        metrics: Optional[PortfolioMetrics] = None,
    ):
        self.name = name
        self.tag = tag
        self.positions: Dict[str, VirtualPosition] = positions or {}
        self.equity_curve: List[DailySnapshot] = equity_curve or []
        self.trade_log: List[Trade] = trade_log or []
        self._metrics: Optional[PortfolioMetrics] = metrics

    @property
    def metrics(self) -> PortfolioMetrics:
        if self._metrics is not None:
            return self._metrics
        return self._compute_metrics()

    def _compute_metrics(self) -> PortfolioMetrics:
        if not self.equity_curve:
            return PortfolioMetrics(
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                ann_return=0.0,
                num_trades=len(self.trade_log),
                current_positions=len(self.positions),
            )
        values = [s.total_value for s in self.equity_curve]
        rets = []
        for i in range(1, len(values)):
            if values[i - 1] > 0:
                rets.append((values[i] / values[i - 1]) - 1.0)
            else:
                rets.append(0.0)

        total_ret = (values[-1] / _INITIAL_CAPITAL - 1.0) if values[-1] > 0 else 0.0

        ann_return = 0.0
        if rets and len(rets) > 0:
            ann_return = (1 + total_ret) ** (252.0 / len(rets)) - 1

        sharpe = 0.0
        if rets:
            r_arr = np.array(rets)
            r_mean = r_arr.mean()
            r_std = r_arr.std(ddof=0)
            if r_std > 0:
                sharpe = (r_mean / r_std) * np.sqrt(252)

        peak = np.maximum.accumulate(values)
        dd_arr = (np.array(values) / peak - 1.0)
        max_dd = float(dd_arr.min()) if len(dd_arr) > 0 else 0.0

        return PortfolioMetrics(
            total_return=total_ret,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            ann_return=ann_return,
            num_trades=len(self.trade_log),
            current_positions=len(self.positions),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tag": self.tag,
            "positions": [
                {"symbol": p.symbol, "shares": p.shares, "entry_price": p.entry_price, "entry_date": p.entry_date}
                for p in self.positions.values()
            ],
            "equity_curve": [
                {"date": s.date, "total_value": s.total_value, "cash": s.cash, "holdings_value": s.holdings_value}
                for s in self.equity_curve
            ],
            "trade_log": [
                {"date": t.date, "symbol": t.symbol, "action": t.action, "price": t.price, "shares": t.shares, "amount": t.amount}
                for t in self.trade_log
            ],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategyPortfolio":
        positions = {
            p["symbol"]: VirtualPosition(
                symbol=p["symbol"],
                shares=int(p["shares"]),
                entry_price=float(p["entry_price"]),
                entry_date=p["entry_date"],
            )
            for p in d.get("positions", [])
        }
        equity_curve = [
            DailySnapshot(
                date=s["date"],
                total_value=float(s["total_value"]),
                cash=float(s["cash"]),
                holdings_value=float(s["holdings_value"]),
            )
            for s in d.get("equity_curve", [])
        ]
        trade_log = [
            Trade(
                date=t["date"],
                symbol=t["symbol"],
                action=t["action"],
                price=float(t["price"]),
                shares=int(t["shares"]),
                amount=float(t["amount"]),
            )
            for t in d.get("trade_log", [])
        ]
        sp = cls(name=d["name"], tag=d.get("tag", ""))
        sp.positions = positions
        sp.equity_curve = equity_curve
        sp.trade_log = trade_log
        return sp


# ----------------------------------------------------------------------
# VirtualPortfolio: manages all strategy portfolios
# ----------------------------------------------------------------------


class VirtualPortfolio:
    def __init__(
        self,
        portfolios: Optional[Dict[str, StrategyPortfolio]] = None,
        last_update_date: str = "",
        initial_capital: float = _INITIAL_CAPITAL,
    ):
        self.portfolios: Dict[str, StrategyPortfolio] = portfolios or {}
        self.last_update_date = last_update_date
        self.initial_capital = initial_capital
        self._strategies_dir = Path(__file__).parent.parent / "predict" / "strategies" / "configs"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_json(self, path: Path) -> None:
        data = {
            "last_update_date": self.last_update_date,
            "initial_capital": self.initial_capital,
            "portfolios": {k: v.to_dict() for k, v in self.portfolios.items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Virtual portfolio state saved to %s", path)

    @classmethod
    def from_json(cls, path: Path) -> "VirtualPortfolio":
        if not path.exists():
            return cls()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        portfolios = {}
        for name, pdata in data.get("portfolios", {}).items():
            portfolios[name] = StrategyPortfolio.from_dict(pdata)
        return cls(
            portfolios=portfolios,
            last_update_date=data.get("last_update_date", ""),
            initial_capital=data.get("initial_capital", _INITIAL_CAPITAL),
        )

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self, update_date: str, checkpoint: str) -> None:
        """Run daily update for all strategies.

        Args:
            update_date: today's date string (YYYY-MM-DD)
            checkpoint: path to model checkpoint
        """
        from ..predict.strategies import ConfidenceStrategy

        logger.info("Starting virtual portfolio update for %s", update_date)

        # Skip if already updated today
        if self.last_update_date == update_date:
            logger.info("Already updated today (%s), skipping", update_date)
            return

        # Load strategies
        registry = StrategyRegistry(self._strategies_dir)
        strategy_metas = registry.list_strategies()
        if not strategy_metas:
            logger.warning("No strategies found, skipping update")
            return

        # Load or create StrategyPortfolio for each strategy
        for meta in strategy_metas:
            name = meta["name"]
            if name not in self.portfolios:
                cfg_dict = registry.get(name)
                self.portfolios[name] = StrategyPortfolio(
                    name=name,
                    tag=cfg_dict.get("tag", ""),
                )

        # Batch inference
        logger.info("Running batch inference for %d strategies...", len(self.portfolios))
        signals, price_map = self._collect_signals(checkpoint, update_date)

        # Update each strategy portfolio
        for meta in strategy_metas:
            name = meta["name"]
            sp = self.portfolios[name]
            cfg_dict = registry.get(name)

            try:
                strat = make_strategy(cfg_dict)
            except Exception as exc:
                logger.warning("Failed to make strategy %s: %s", name, exc)
                continue

            self._update_one_strategy(sp, strat, signals, price_map, update_date)

        self.last_update_date = update_date
        logger.info("Virtual portfolio update complete for %s", update_date)

    def _collect_signals(
        self,
        checkpoint: str,
        update_date: str,
    ) -> tuple[list[Signal], Dict[str, float]]:
        """Run parallel batch inference and return signals + latest prices."""
        predictor = PricePredictor(checkpoint)
        symbols = filter_a_share_symbols(list_symbols())
        logger.info("Collecting signals for %d symbols...", len(symbols))

        price_map: Dict[str, float] = {}
        conn = _connect()
        placeholders = ",".join("?" * len(symbols))
        rows = conn.execute(
            f"""SELECT symbol, close FROM price
                WHERE symbol IN ({placeholders})
                AND date = (SELECT MAX(date) FROM price WHERE symbol = price.symbol)""",
            symbols,
        ).fetchall()
        for sym, close in rows:
            price_map[sym] = float(close)

        signals: list[Signal] = []

        def _predict_one(sym: str) -> Optional[Signal]:
            try:
                result = predictor.predict(sym, horizon=None)
                if isinstance(result, dict):
                    pred_rets = result
                    pred_ret_1d = pred_rets.get("1d", 0.0)
                else:
                    pred_rets = {"1d": float(result)}
                    pred_ret_1d = float(result)
                return Signal(
                    symbol=sym,
                    predicted_ret=float(pred_ret_1d),
                    realized_ret=0.0,
                    industry=None,
                    predicted_rets=pred_rets,
                )
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
            futures = {executor.submit(_predict_one, sym): sym for sym in symbols}
            for i, future in enumerate(as_completed(futures)):
                sig = future.result()
                if sig is not None:
                    signals.append(sig)
                if (i + 1) % 500 == 0:
                    logger.info("  ... collected %d/%d signals", i + 1, len(symbols))

        logger.info("Collected %d signals", len(signals))
        return signals, price_map

    def _update_one_strategy(
        self,
        sp: StrategyPortfolio,
        strat,
        signals: list[Signal],
        price_map: Dict[str, float],
        update_date: str,
    ) -> None:
        """Update a single strategy's portfolio for today."""
        prev_weights: Dict[str, float] = {}
        cache: Dict[str, Any] = {}

        # Compute previous day's weight from equity curve
        if sp.equity_curve:
            last_snapshot = sp.equity_curve[-1]
            total = last_snapshot.total_value
            if total > 0:
                for sym, pos in sp.positions.items():
                    holdings_val = pos.shares * price_map.get(sym, pos.entry_price)
                    prev_weights[sym] = holdings_val / total

        # Get target weights from strategy
        weights = strat.select_positions(signals, prev_weights, cache)
        pos_weights = {s: w for s, w in weights.items() if w > 0}
        target_syms = set(pos_weights.keys())

        today_str = update_date
        today_date = date.fromisoformat(update_date)

        # ── Sell positions no longer in target ──────────────────────────
        for sym in list(sp.positions.keys()):
            if sym in target_syms:
                continue
            # T+1 check: can't sell today if bought today
            pos = sp.positions[sym]
            if pos.entry_date == today_str:
                continue
            sell_price = price_map.get(sym, pos.entry_price)
            if sell_price <= 0:
                continue
            sell_amount = pos.shares * sell_price
            sp.trade_log.append(Trade(
                date=today_str, symbol=sym, action="sell",
                price=sell_price, shares=pos.shares, amount=sell_amount,
            ))
            del sp.positions[sym]

        # ── Update existing position entry_price with latest close ──────
        for sym, pos in sp.positions.items():
            if sym in price_map:
                pos.entry_price = price_map[sym]

        # ── Buy new target positions ───────────────────────────────────
        available_cash = self.initial_capital
        if sp.equity_curve:
            # Use latest total value as reference for sizing
            last_val = sp.equity_curve[-1].total_value
            available_cash = max(last_val, self.initial_capital)

        for sym in list(target_syms):
            if sym in sp.positions:
                continue
            if sym not in price_map or price_map[sym] <= 0:
                continue
            buy_price = price_map[sym]
            weight = pos_weights.get(sym, 0.0)
            cash_for_pos = weight * available_cash
            max_possible = int(cash_for_pos // buy_price // _LOT_SIZE) * _LOT_SIZE
            if max_possible < _LOT_SIZE:
                continue
            cost = max_possible * buy_price
            sp.positions[sym] = VirtualPosition(
                symbol=sym,
                shares=max_possible,
                entry_price=buy_price,
                entry_date=today_str,
            )
            sp.trade_log.append(Trade(
                date=today_str, symbol=sym, action="buy",
                price=buy_price, shares=max_possible, amount=cost,
            ))

        # ── Compute today's equity snapshot ─────────────────────────────
        holdings_value = 0.0
        for sym, pos in sp.positions.items():
            p = price_map.get(sym, pos.entry_price)
            holdings_value += pos.shares * p
        total_value = holdings_value  # simplified: all-in portfolio, no cash tracking
        sp.equity_curve.append(DailySnapshot(
            date=today_str,
            total_value=total_value,
            cash=0.0,
            holdings_value=holdings_value,
        ))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_top_n(self, n: int = 3) -> List[StrategyPortfolio]:
        """Return top-N strategies by total return."""
        sorted_portfolios = sorted(
            self.portfolios.values(),
            key=lambda p: p.metrics.total_return,
            reverse=True,
        )
        return sorted_portfolios[:n]

    def summary(self) -> str:
        lines = [f"Virtual Portfolio ({self.last_update_date})"]
        lines.append(f"{'Strategy':<30} {'Return':>8} {'Sharpe':>7} {'DD':>8} {'Trades':>7} {'Pos':>5}")
        lines.append("-" * 70)
        for sp in sorted(self.portfolios.values(), key=lambda p: p.metrics.total_return, reverse=True):
            m = sp.metrics
            lines.append(
                f"{sp.name:<30} {m.total_return*100:>+7.2f}% {m.sharpe_ratio:>7.3f} "
                f"{m.max_drawdown*100:>+7.2f}% {m.num_trades:>7} {m.current_positions:>5}"
            )
        return "\n".join(lines)
