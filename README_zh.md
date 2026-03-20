# StockPilot

A 股自动选股工具，结合 LLM 估值评分与 Transformer 模型预测次日收益。包含 Web 控制面板和内置回测引擎。

## 功能

- **数据**：抓取每日行情和财务数据存入 SQLite，缓存为 npz 预处理张量
- **模型**：因果 Transformer 编码器，含回归头和分类头；检查点包含 scaler 和配置，支持可复现推理
- **推理**：`PricePredictor` API 和 CLI，支持单股预测
- **回测**：纯多（top-K）或做多/做空（前后 N%）策略，支持交易成本/滑点、换手率和行业集中度统计
- **Ops 控制面板**：本地 Web UI，可抓取数据、训练、推理、回测，带实时日志
- **报告**：HTML 排行榜和可排序的价格/财务看板

## 快速开始

```bash
# 安装
pip install -r requirements.txt

# 抓取历史数据
python -m auto_select_stock.cli fetch-all --start 2018-01-01

# 训练模型
python -m auto_select_stock.cli train-transformer \
  --seq-len 60 --epochs 20 --batch-size 64 --device cuda \
  --save-path models/price_transformer.pt

# 预测次日收盘价
python -m auto_select_stock.cli predict-transformer 600000 \
  --seq-len 60 --checkpoint models/price_transformer.pt

# 回测（做多/做空）
python -m auto_select_stock.cli backtest-transformer \
  --start 2023-01-01 --end 2023-06-30 \
  --top-pct 0.1 --checkpoint models/price_transformer.pt

# 回测（纯多 top-K）
python -m auto_select_stock.cli backtest-transformer --mode topk --top-k 5 \
  --checkpoint models/price_transformer.pt

# Web 控制面板
python -m auto_select_stock.ops_dashboard
# 打开 http://127.0.0.1:8000
```

## 回测命令

| 命令 | 说明 |
|------|------|
| `backtest-transformer` | 基于预测收益的做多/做空组合（前后 N%） |
| `backtest-transformer --mode topk` | 每日批量推理，按预测收益排序，买入 top-K |
| `backtest-strategy` | 同 `--mode topk`，结果保存到检查点同目录 JSON |
| `backtest-per-symbol --workers N` | 按股票分别回测，输出 CSV；workers > 1 时多进程 |

## LLM 评分

```bash
export OPENAI_API_KEY=your_key
python -m auto_select_stock.cli score --top 50 --provider openai
python -m auto_select_stock.cli render --top 50 --output reports/undervalued.html
```

## 目录结构

```
src/auto_select_stock/
  config.py              - 路径和环境变量默认值
  storage.py             - SQLite 读写（行情表和财务表）
  data_fetcher.py       - 每日行情数据抓取（akshare）
  financials_fetcher.py  - 季度财报抓取
  predict/
    data.py              - 预处理、特征缓存、数据集
    torch_model.py       - PriceTransformer 架构
    train.py             - 训练循环，支持日期窗口切分
    inference.py         - PricePredictor 批量推理
    backtest.py          - 回测策略（多空、topk）
    checkpoints.py       - 检查点保存/加载
    strategy.py          - 组合构建辅助函数
  cli.py                - 所有 CLI 命令
  ops_dashboard.py       - Web 控制面板（端口 8000）
  scoring.py            - LLM 股票评分
  llm/                  - LLM 提供商适配器
  html_report.py         - HTML 排行榜渲染
  dashboard.py           - 可排序价格/财务看板
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `AUTO_SELECT_STOCK_DATA_DIR` | `data/` | 行情/财务数据目录 |
| `AUTO_SELECT_MODEL_DIR` | `models/` | 检查点存储目录 |
| `AUTO_SELECT_STOCK_PREPROCESSED_DIR` | `data/preprocessed/` | 缓存特征目录 |
| `AUTO_SELECT_LLM_PROVIDER` | `openai` | LLM 提供商 |
| `AUTO_SELECT_LLM_MODEL` | `gpt-4o-mini` | LLM 模型 |
| `OPENAI_API_KEY` | - | LLM 评分所需 |

## 注意事项

- 从仓库根目录运行时，需设置 `PYTHONPATH=./src`
- CPU 机器上的 CUDA 警告可忽略，推理会自动回退到 CPU
- 日期窗口训练（如 `--date-window 2022-01-01:2023-01-01`）可防止财报数据泄露到训练集
- 使用 `--provider dummy` 可在不调用外部 API 的情况下测试评分功能
