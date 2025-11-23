from pathlib import Path
from typing import List

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .stock_types import StockScore


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(searchpath=Path(__file__).resolve().parent),
        autoescape=select_autoescape(["html"]),
    )


def render_report(scores: List[StockScore], top_n: int, output_path: Path) -> Path:
    env = _env()
    template = env.from_string(REPORT_TEMPLATE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(
            template.render(
                items=scores[:top_n],
            )
        )
    return output_path


REPORT_TEMPLATE = """<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <title>低估值股票榜单</title>
  <style>
    :root { --bg:#0e1116; --card:#161b22; --text:#e6edf3; --accent:#58a6ff; }
    body { background: radial-gradient(circle at 20% 20%, #162036 0, #0e1116 40%, #0b0f18 100%); color: var(--text); font-family: "Segoe UI", Arial, sans-serif; margin:0; padding:32px; }
    h1 { margin:0 0 16px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }
    .card { background: var(--card); border:1px solid #1f2630; border-radius: 12px; padding:16px; box-shadow: 0 10px 30px rgba(0,0,0,.45); transition: transform .12s ease, border-color .12s ease; }
    .card:hover { transform: translateY(-3px); border-color: var(--accent); }
    .score { font-size: 32px; font-weight: 700; color: var(--accent); }
    .pill { display:inline-block; padding:4px 10px; border-radius: 999px; background:#1d2633; margin-right:6px; font-size:12px; }
    details { margin-top:12px; }
    summary { cursor: pointer; color: var(--accent); }
    table { width:100%; border-collapse: collapse; margin-top:8px; }
    th, td { text-align:left; padding:4px 0; font-size: 13px; }
  </style>
</head>
<body>
  <h1>低估值股票榜单</h1>
  <div class="grid">
    {% for item in items %}
    <div class="card">
      <div class="score">{{ "%.2f"|format(item.score) }}</div>
      <div class="pill">代码 {{ item.symbol }}</div>
      <div class="pill">估值评分</div>
      <details>
        <summary>展开详情</summary>
        <table>
          {% for k,v in item.factors.items() %}
          <tr><th>{{ k }}</th><td>{{ v }}</td></tr>
          {% endfor %}
        </table>
        <p style="margin-top:8px; color:#9fb8d3;">{{ item.rationale }}</p>
      </details>
    </div>
    {% endfor %}
  </div>
</body>
</html>
"""
