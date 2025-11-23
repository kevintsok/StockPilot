# A股自动选股器（示例架构）

用 Python 抓取 A 股历史数据、调用外部大模型打分，并在本地 GPU 上训练 Transformer 预测次日收盘价，最终输出可视化报告。

## 功能概览
- 抓取全市场或指定股票的日线行情与基础财务字段（示例使用 `akshare`）。
- 将每只股票按日线保存为 `numpy` 压缩文件（`data/<symbol>.npz`），包含价格、换手率、量比等特征。
- LLM 打分：抽象接口可插拔不同 Provider（OpenAI、Dummy），输出 0-10 的潜力分。
- Transformer 预测：自回归方式预测第二天收盘价（PyTorch + Adam，定期执行 evaluation）。
- 生成 HTML 榜单，展示最高分的股票并支持查看因子详情。
- 支持每日增量更新最近行情。

## 快速开始
1) 安装依赖
```bash
python -m venv .venv
.venv/Scripts/activate    # Windows
pip install -r requirements.txt
```

2) 抓取历史数据
```bash
python -m auto_select_stock.cli fetch-all
```

3) LLM 打分与报告
```bash
$env:OPENAI_API_KEY="your_key_here"    # PowerShell
python -m auto_select_stock.cli score --top 50 --provider openai
python -m auto_select_stock.cli render --top 50 --output reports/undervalued.html
```

4) Transformer 训练（GPU 优先）
```bash
python -m auto_select_stock.cli train-transformer --seq-len 60 --epochs 20 --device cuda
# 参数示例：--batch-size 128 --lr 5e-4 --eval-every 1 --save-path models/price_transformer.pt
```

5) Transformer 推理（预测次日收盘价）
```bash
python -m auto_select_stock.cli predict-transformer 600000 --seq-len 60 --checkpoint models/price_transformer.pt
```

6) 每日更新
```bash
python -m auto_select_stock.cli update-daily
```

## 目录结构
- `src/auto_select_stock/`
  - `config.py`：目录与默认配置
  - `types.py`：数据结构定义
  - `data_fetcher.py`：行情抓取与增量更新
  - `storage.py`：本地 numpy 存取
  - `llm/`：大模型接口（OpenAI、Dummy）
  - `scoring.py`：LLM 打分逻辑
  - `torch_model.py`：Transformer 模型、数据集构建、训练/推理
  - `html_report.py`：报告渲染
  - `cli.py`：命令行入口

## 依赖说明
- 数据：`akshare`
- 科学计算：`pandas`、`numpy`
- 报告：`jinja2`
- LLM：`openai`、`httpx`
- 深度学习：`torch`（建议使用有 CUDA 的版本）

## 训练细节（Transformer）
- 输入特征：`open, high, low, close, volume, amount, turnover_rate, volume_ratio, pct_change`。
- 序列切片：滑动窗口长度 `seq_len`，目标为下一个交易日的 `close`。
- 标准化：使用训练集均值/方差做 z-score；保存在 checkpoint 方便推理复用。
- 模型：Transformer Encoder（位置编码 + 多层 encoder + 线性头输出标量）。
- 损失：自回归方式的 MSE（预测下一日收盘价）；优化器 Adam。
- 验证：`--eval-every` 控制评估频率，验证集基于时间切分。

## 下一步可扩展
- 引入真实财务因子（ROE、PE、PB、现金流）并拼接到特征。
- 调参与正则化（dropout/weight decay/learning rate schedule）。
- 早停/多 checkpoint 管理；多步前瞻预测。
- 报告前端增加图表与交互筛选。
