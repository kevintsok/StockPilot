import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("AUTO_SELECT_STOCK_DATA_DIR", PROJECT_ROOT / "data"))
REPORT_DIR = Path(os.getenv("AUTO_SELECT_STOCK_REPORT_DIR", PROJECT_ROOT / "reports"))
MODEL_DIR = Path(os.getenv("AUTO_SELECT_MODEL_DIR", PROJECT_ROOT / "models"))
PREPROCESSED_DIR = Path(os.getenv("AUTO_SELECT_STOCK_PREPROCESSED_DIR", DATA_DIR / "preprocessed"))

# 默认 LLM 配置，可通过环境变量覆盖
DEFAULT_LLM_PROVIDER = os.getenv("AUTO_SELECT_LLM_PROVIDER", "openai")
DEFAULT_LLM_MODEL = os.getenv("AUTO_SELECT_LLM_MODEL", "gpt-4o-mini")

# 请求并发或节流参数
MAX_CONCURRENT_REQUESTS = int(os.getenv("AUTO_SELECT_MAX_CONCURRENCY", "4"))
REQUEST_SLEEP_SECONDS = float(os.getenv("AUTO_SELECT_REQUEST_SLEEP", "0.0"))

# 抓取使用的起始日期
DEFAULT_START_DATE = os.getenv("AUTO_SELECT_START_DATE", "2018-01-01")
