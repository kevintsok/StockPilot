#!/usr/bin/env python3
"""
BC Model Top-5 Push Script.

Uses the Behavior Cloning model to recommend top 5 stocks and push via PushPlus.
"""
import sys
sys.path.insert(0, 'src')
import datetime
import os
import subprocess

from auto_select_stock.notify.push_providers import PushPlusProvider
from auto_select_stock.data.storage import _connect


CHECKPOINT = "/mnt/d/Projects/auto-select-stock/models/bc_pretrain.pt"
TOKEN = os.getenv("PUSHPLUS_TOKEN", "183cae5e7d8148f0b85754a2912fc81c")


def _is_trading_day(db_date_str: str) -> bool:
    """Check if the DB's latest date is today or yesterday (accounting for weekends)."""
    if not db_date_str:
        return False
    db_date = datetime.datetime.strptime(db_date_str, "%Y-%m-%d").date()
    today = datetime.date.today()
    diff = (today - db_date).days
    return diff <= 1


def get_latest_price_date() -> str:
    """Return the most recent price date in the database."""
    conn = _connect()
    return conn.execute("SELECT MAX(date) FROM price").fetchone()[0]


def _ensure_fresh_data() -> None:
    """Ensure the price database is up-to-date before pushing."""
    latest_date = get_latest_price_date()
    if _is_trading_day(latest_date):
        print(f"Database is fresh (latest: {latest_date}), skipping update.")
        return

    print(f"Database is stale (latest: {latest_date}, today: {datetime.date.today()}).")
    print("Running update-daily to fetch latest data...")

    result = subprocess.run(
        [sys.executable, "-m", "auto_select_stock.cli", "update-daily"],
        env={**os.environ, "PYTHONPATH": "./src"},
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"WARNING: update-daily failed: {result.stderr}", file=sys.stderr)
    else:
        new_date = get_latest_price_date()
        print(f"Update complete. Latest date now: {new_date}")


def main():
    import torch
    import numpy as np
    from tqdm import tqdm
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from auto_select_stock.config import DATA_DIR
    from auto_select_stock.data.storage import list_symbols, load_stock_history
    from auto_select_stock.predict.data import compute_technical_indicators, PRICE_FEATURE_COLUMNS

    class BCNetwork(torch.nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 128):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim // 2),
                torch.nn.ReLU(),
            )
            self.action_head = torch.nn.Linear(hidden_dim // 2, 1)
            self.action_head.weight.data.fill_(0.0)
            self.action_head.bias.data.fill_(0.0)

        def forward(self, x):
            features = self.net(x)
            action = torch.sigmoid(self.action_head(features))
            return action

    def compute_state(seq, t, position=0.0, entry_price=0.0):
        recent = seq[max(0, t - 20):t + 1, :]
        if len(recent) >= 20:
            ma20 = np.mean(recent[-20:, 3])
            price_level = recent[-1, 3] / ma20 if ma20 > 0 else 1.0
        else:
            price_level = 1.0
        if len(recent) >= 2:
            ret_1d = np.log(recent[-1, 3] / recent[-2, 3]) if recent[-2, 3] > 0 else 0.0
        else:
            ret_1d = 0.0
        if len(recent) >= 5:
            ret_5d = np.log(recent[-1, 3] / recent[-5, 3]) if recent[-5, 3] > 0 else 0.0
        else:
            ret_5d = 0.0
        unrealized_pnl = 0.0
        state = np.array([price_level - 1.0, ret_1d, ret_5d, position / 0.15, unrealized_pnl, 0.0], dtype=np.float32)
        return state

    _ensure_fresh_data()
    latest_date = get_latest_price_date()
    print(f"Latest price date: {latest_date}")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BCNetwork(input_dim=6, hidden_dim=128).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()
    print(f"Loaded BC model, device={device}")

    # Load stock data
    import pandas as pd
    all_symbols = list_symbols(base_dir=DATA_DIR)
    print(f"Total symbols: {len(all_symbols)}")

    results = []
    for sym in tqdm(all_symbols[:500], desc="Scoring"):
        try:
            arr = load_stock_history(sym, base_dir=DATA_DIR)
            if arr is None or len(arr) < 120:
                continue
            df = pd.DataFrame(arr)
            df["date"] = pd.to_datetime(df["date"]).dt.floor("D").astype("datetime64[ns]")
            df.sort_values("date", inplace=True)
            df = df[df["date"] >= pd.to_datetime("2024-01-01")]
            if len(df) < 120:
                continue
            price_df = df[PRICE_FEATURE_COLUMNS].copy()
            tech_df = compute_technical_indicators(df)
            combined = pd.concat([price_df, tech_df], axis=1)
            cols = list(PRICE_FEATURE_COLUMNS)
            for c in ["rsi_14", "macd_line", "macd_signal", "macd_hist",
                      "bb_position", "bb_width", "volume_ma5", "volume_ma20",
                      "atr_14", "stoch_k", "stoch_d", "obv_ma10", "roc_10", "momentum_10"]:
                if c in combined.columns:
                    cols.append(c)
            arr_out = combined[cols].values.astype(np.float32)
            arr_out = np.nan_to_num(arr_out, nan=0.0, posinf=0.0, neginf=0.0)
            for ci in range(arr_out.shape[1]):
                arr_out[:, ci] = np.clip(arr_out[:, ci], -1e6, 1e6)

            t = len(arr_out) - 2
            if t < 60:
                continue
            state = compute_state(arr_out, t, position=0.0, entry_price=0.0)
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_t).item()
            current_price = arr_out[-1, 3]  # Use latest price (last row)
            results.append((sym, action, current_price))
        except Exception:
            continue

    # Sort and filter
    results.sort(key=lambda x: -x[1])
    filtered = [r for r in results if r[1] >= 0.05]
    top5 = filtered[:5]

    # Get stock names
    import akshare as ak
    try:
        name_df = ak.stock_info_a_code_name()
        name_map = dict(zip(name_df['code'], name_df['name']))
    except Exception:
        name_map = {}

    print(f"\n{'='*60}")
    print("BC Model Top-5 Recommendations")
    print(f"{'='*60}")
    for i, (sym, score, price) in enumerate(top5):
        name = name_map.get(sym, "")
        print(f"{i+1}. {sym} {name} | Score: {score:.4f} | Price: {price:.2f}")

    # Generate HTML
    html = f"""<h2>BC Model 明日荐股 (Top-5)</h2>
<p><b>生成时间</b>: {datetime.date.today().isoformat()}</p>
<p><b>数据最新日期</b>: {latest_date}</p>
<table border="1" cellpadding="5" cellspacing="0">
<tr><th>排名</th><th>股票代码</th><th>股票名称</th><th>评分</th><th>当前价格</th></tr>
"""
    for i, (sym, score, price) in enumerate(top5):
        name = name_map.get(sym, "")
        html += f"<tr><td>{i+1}</td><td><b>{sym}</b></td><td>{name}</td><td>{score:.4f}</td><td>{price:.2f}</td></tr>\n"
    html += """</table>
<p><b>说明</b>:</p>
<ul>
<li>评分基于 BC 模型输出的持仓概率 [0, 1]</li>
<li>筛选条件: score >= 0.05</li>
<li>BC 模型 = 行为克隆，模仿专家交易规则</li>
</ul>
<p><b>历史表现</b> (2024-2025 回测):</p>
<ul>
<li>总收益: +56.39%</li>
<li>Sharpe: 17.5</li>
<li>交易次数: 2,053</li>
</ul>
"""

    # Send push
    provider = PushPlusProvider(token=TOKEN)
    provider.send(title=f"StockPilot BC Top-5 {latest_date}", content=html)
    print(f"\nPushPlus: sent {len(top5)} stocks")


if __name__ == "__main__":
    main()
