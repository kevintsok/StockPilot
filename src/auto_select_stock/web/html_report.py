from pathlib import Path
from typing import List

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..core.types import StockScore


def _env() -> Environment:
    """Get Jinja2 environment with FileSystemLoader pointing to templates directory."""
    template_dir = Path(__file__).resolve().parent / "templates"
    return Environment(
        loader=FileSystemLoader(searchpath=str(template_dir)),
        autoescape=select_autoescape(["html"]),
    )


def render_report(scores: List[StockScore], top_n: int, output_path: Path) -> Path:
    env = _env()
    template = env.get_template("report.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(
            template.render(
                items=scores[:top_n],
            )
        )
    return output_path
