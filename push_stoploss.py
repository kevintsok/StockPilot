"""Direct push for StopLoss-3pct-14d strategy without update-daily subprocess."""
import sys
sys.path.insert(0, 'src')

from auto_select_stock.notify.pipeline import get_top_k_stocks
from auto_select_stock.notify.push_providers import PushPlusProvider
from auto_select_stock.notify.config import PUSHPLUS_TOKEN

CHECKPOINT = "models/price_transformer_full.pt"
STRATEGY = "StopLoss-3pct-14d"
TOKEN = PUSHPLUS_TOKEN or "183cae5e7d8148f0b85754a2912fc81c"

print("Fetching top stocks...")
results = get_top_k_stocks(
    checkpoint=CHECKPOINT,
    strategy=STRATEGY,
    top_k=10,
    horizon="14d",
)

print(f"Got {len(results)} stocks:")
for sym, pred_rets, weight in results:
    ret_14d = pred_rets.get("14d", 0) * 100
    ret_1d = pred_rets.get("1d", 0) * 100
    ret_3d = pred_rets.get("3d", 0) * 100
    ret_5d = pred_rets.get("5d", 0) * 100
    print(f"  {sym}: 1d={ret_1d:+.2f}% 3d={ret_3d:+.2f}% 5d={ret_5d:+.2f}% 14d={ret_14d:+.2f}% weight={weight:.4f}")

# Build simple HTML
rows = ""
for sym, pred_rets, weight in results:
    r14 = pred_rets.get("14d", 0) * 100
    r1 = pred_rets.get("1d", 0) * 100
    r3 = pred_rets.get("3d", 0) * 100
    r5 = pred_rets.get("5d", 0) * 100
    color = "#2e7d32" if r14 > 0 else "#c62828"
    pos = weight * 100_000
    rows += f"<tr><td><b>{sym}</b></td><td style='color:{color}'>{r14:+.2f}%</td>"
    rows += f"<td>{r1:+.1f}%</td><td>{r3:+.1f}%</td><td>{r5:+.1f}%</td>"
    rows += f"<td>{weight:.2%}<br><span style='color:#888;font-size:11px'>~{pos:.0f}元</span></td></tr>\n"

html = f"""<html><body style='font-family:Arial;font-size:14px'>
<h3>StockPilot 每日推荐 | StopLoss-3pct-14d | 2026-03-27数据</h3>
<p style='color:#888'>基于14日预测排序，等权持仓，3%止损线</p>
<table border='1' cellpadding='6' cellspacing='0' style='border-collapse:collapse;width:100%'>
<tr style='background:#f5f5f5'><td><b>股票</b></td><td><b>14d预测</b></td><td>1d</td><td>3d</td><td>5d</td><td><b>权重/仓位</b></td></tr>
{rows}
</table>
<p style='color:#aaa;font-size:11px'>模型：price_transformer_full.pt | 数据截至 2026-03-27</p>
</body></html>"""

print("\nSending to PushPlus...")
provider = PushPlusProvider(token=TOKEN)
provider.send(title="StockPilot 每日推荐 StopLoss-3pct-14d 2026-03-27", content=html)
print("Done!")
