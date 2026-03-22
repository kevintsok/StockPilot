# StockPilot 中文文档

> A股自动化选股系统 — LLM估值评分 + Transformer价格预测 + 多策略回测

预测次日收益、股票排名、一键对比10+交易策略。专为沪深/创业板A股设计。

[English README](../README.md)

---

## 功能概览

1. **数据获取** — 每日价格 + 季度财务数据入库SQLite
2. **模型训练** — Transformer预测次日收益率
3. **回测** — 10+交易策略（共享信号，一次GPU推理）
4. **LLM评分** — 大语言模型定性估值
5. **控制面板** — 本地Web界面

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 获取5年A股历史数据
python -m auto_select_stock.cli fetch-all --start 2018-01-01 --limit 100

# 3. 训练模型（按时间窗口划分防止财务数据泄露）
python -m auto_select_stock.cli train-transformer \
  --seq-len 60 --epochs 20 --batch-size 64 --device cuda \
  --date-window 2022-01-01:2023-01-01 \
  --save-path models/price_transformer.pt

# 4. 一次性回测全部10种策略（共享信号 = 1次GPU推理）
python -m auto_select_stock.cli backtest-strategies \
  --start 2023-01-01 --end 2024-12-31 \
  --checkpoint models/price_transformer.pt \
  --cost-bps 15 --slippage-bps 10

# 5. Web控制面板
python -m auto_select_stock.ops_dashboard
# 访问 http://127.0.0.1:8000
```

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        StockPilot 流程                           │
└─────────────────────────────────────────────────────────────────┘

  ┌──────────────┐     ┌─────────────────┐     ┌────────────────┐
  │  数据获取    │────▶│   数据预处理     │────▶│  SQLite (.db)  │
  │  (akshare)  │     │   npz缓存       │     │  price/fin    │
  └──────────────┘     └─────────────────┘     └───────┬────────┘
                                                       │
                       ┌────────────────────────────────┘
                       ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                    Transformer 模型训练                            │
  │  PriceTransformer: 因果编码器 + 回归/分类头                         │
  │  时间窗口划分 ──▶ 训练/验证/测试（防止数据泄露）                    │
  └────────────────────────────────────┬───────────────────────────┘
                                       │ 模型检查点
                                       ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │              多策略回测（共享信号）                                 │
  │                                                                  │
  │  _collect_signals_batched() ──▶ 所有股票/日期一次GPU推理           │
  │                    │                                             │
  │         ┌──────────┴──────────┬──────────┬──────────┐             │
  │         ▼                     ▼          ▼          ▼             │
  │   TopK-Proportional   Momentum   Risk-Parity  Sector-Neutral ... │
  │   (各策略独立运行 select_positions())                            │
  └──────────────────────────────────────────────────────────────────┘
```

**核心设计：一个模型 → 一次信号采集 → 所有策略公平对比**

---

## 模型架构

### PriceTransformer: 输入 → 输出

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           输入  (每时间步)                              │
│                                                                         │
│   ┌──────────────────┐  ┌──────────────────┐  ┌───────────────────────┐│
│   │  价格特征 (12维)   │  │ 财务特征 (7维)    │  │ 技术指标 (14维)        ││
│   │                  │  │                  │  │                       ││
│   │ open             │  │ roe              │  │ rsi_14                ││
│   │ high             │  │ net_profit_margin│  │ macd_line             ││
│   │ low              │  │ gross_margin     │  │ macd_signal           ││
│   │ close            │  │ operating_cashflow│ │ macd_hist             ││
│   │ volume           │  │   _growth        │  │ bb_position           ││
│   │ amount           │  │ debt_to_asset    │  │ bb_width              ││
│   │ turnover_rate    │  │ eps              │  │ volume_ma5            ││
│   │ volume_ratio     │  │ operating_cashflow│ │ volume_ma20           ││
│   │ pct_change       │  │   _per_share     │  │ atr_14                ││
│   │ amplitude        │  │                  │  │ stoch_k               ││
│   │ change_amount    │  │                  │  │ stoch_d               ││
│   │                  │  │                  │  │ obv_ma10              ││
│   │                  │  │                  │  │ roc_10                ││
│   │                  │  │                  │  │ momentum_10           ││
│   └────────┬─────────┘  └────────┬─────────┘  └───────────┬───────────┘│
│            └────────────────────┼───────────────────────┘            │
│                                 concat = 33 维                         │
└─────────────────────────────────────┬─────────────────────────────────┘
                                      │ shape: (batch, seq_len=1024, 33)
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          PriceTransformer                               │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  input_proj: Linear(33 → 256)                                     │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                 │                                       │
│                                 ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  正弦位置编码  (动态扩展至任意 seq_len)                              │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                 │                                       │
│                                 ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  TransformerEncoder  (10层, 8头, dim_ffn=512)                     │ │
│  │  因果掩码: 位置 i 看不到 i+1, i+2, ...                           │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                 │                                       │
│                    ┌────────────┴────────────┐                          │
│                    ▼                         ▼                          │
│  ┌───────────────────────────┐  ┌───────────────────────────────┐     │
│  │  回归头                    │  │  分类头                         │     │
│  │  Linear(256 → 1)          │  │  Linear(256 → 1)               │     │
│  │  输出: (batch, seq_len)  │  │  输出: (batch, seq_len) logits │     │
│  │  每步一个 log_return 预测 │  │  每步一个 up/down 概率         │     │
│  └─────────────┬─────────────┘  └──────────────┬──────────────────┘     │
│                │                                  │                     │
│                └──────────────┬───────────────────┘                     │
│                               ▼                                          │
│                    ┌──────────────────────┐                             │
│                    │  最后一个时间步 = t    │                             │
│                    │  即次日预测            │                             │
│                    └──────────┬───────────┘                             │
└───────────────────────────────┼─────────────────────────────────────────┘
                                │
               ┌───────────────┴────────────────┐
               ▼                                ▼
┌──────────────────────────────┐  ┌──────────────────────────────────────┐
│  回归输出                     │  │  分类输出                             │
│                              │  │                                      │
│  pred_log_return = out[0,-1] │  │  pred_direction = sigmoid(out[1,-1]) │
│                              │  │  > 0.5 → 上涨, < 0.5 → 下跌         │
│  exp(pred_log_return) - 1    │  │                                      │
│  = predicted_return         │  │  用于: 置信度排序、策略权重             │
│                              │  │                                      │
│  用于: 策略排序 & 权重计算     │  │                                      │
└──────────────────────────────┘  └──────────────────────────────────────┘
```

### 训练: 损失函数

```
loss = λ_reg · MSE(pred_log_return, real_log_return)
     + λ_cls · BCE(pred_direction, real_up/down)
     + λ_rank · RankingLoss(pred_return pairs)

默认权重:  λ_reg=0.1   λ_cls=10.0   λ_rank=1.0
          (回归损失权重低，分类损失主导，排序损失辅助)
```

### 置信度策略: 输出如何使用

```
predicted_return (回归头最后一个时间步)
         │
         ├─── abs(predicted_return) = 置信度
         │         │
         │         ▼
         │    ┌────────────────────┐
         │    │  置信度归一化权重   │
         │    │  weight_i =        │
         │    │    |ret_i| / Σ|ret││
         │    └────────────────────┘
         │
         └─── sign(predicted_return) 决定方向
                  (纯做多策略只看正值)
```

### 特征汇总

| 类别 | 维度 | 说明 |
|------|------|------|
| 价格 | 12 | 开盘/最高/最低/收盘 + 成交量/金额/换手率等 |
| 财务 | 7 | ROE、利润率、现金流、负债率、EPS — 季报数据回填 |
| 技术 | 14 | RSI、MACD、布林带、成交量均线、ATR、随机指标、OBV、ROC、动量 |
| **合计** | **33** | 每时间步一个33维向量 |

### 关键设计决策

- **因果掩码**: Transformer位置i看不到未来信息，确保不泄露未来价格
- **时间窗口划分**: 训练/验证/测试按时间划分，防止财务报告数据穿越
- **分类损失主导**: `λ_cls=10.0` >> `λ_reg=0.1`，模型优先学习方向而非精确收益率
- **排序损失**: ListMLE-style hinge loss，优化股票间的相对排序
- **动态位置编码**: 推理时可处理比训练时更长的序列

---

## 回测结果 (2024-06-01 ~ 2025-06-01)

![回测资金曲线](backtest_diverse_30_capital_curve.png)

**模型**: `price_transformer_multihorizon_full.pt` — 多期限Transformer (1d/3d/5d/7d/14d/20d头), 1677只股票, seq_len=1024, 训练3轮。**初始资金 100,000 RMB**，最小买卖单位100股，涨跌停禁止买卖。**所有策略均为纯做多（A股不允许做空）**。

| 策略 | Tag | 最终资金 | 总收益 | 夏普 | 最大回撤 | 平均换手率 |
|------|-----|--------:|-------:|-----:|---------:|----------:|
| **RiskParity-VL10-1d** | **469da** | **328,393** | **+228.4%** | **6.75** | **-19.7%** | 64.9% |
| StopLoss-3pct-5d | 47cda | 272,027 | +172.0% | 6.61 | -16.6% | 63.9% |
| StopLoss-3pct-1d | fc99f | 188,542 | +88.5% | 3.70 | -16.7% | 52.7% |
| TopK-K10-1d | 3f274 | 143,469 | +43.5% | 1.67 | -26.1% | 56.4% |
| RiskParity-VL20-5d | dfe5c | 135,366 | +35.4% | 1.08 | -31.3% | 56.3% |
| StopLoss-8pct-1d | a1e30 | 128,007 | +28.0% | 1.76 | -9.8% | 23.7% |
| StopLoss-8pct-5d | 21312 | 115,103 | +15.1% | 1.48 | -6.1% | 23.6% |
| Momentum-LB5-1d | 7ef00 | 114,655 | +14.7% | 0.44 | -37.1% | 54.4% |
| TopK-K3-3d | a2b0b | 94,026 | -6.0% | -0.14 | -44.7% | 61.0% |
| TopK-K10-14d | c8638 | 25,123 | -74.9% | -2.66 | -75.7% | 46.4% |

**关键洞察**:
- **RiskParity-VL10-1d [469da] (+228%, 夏普6.75)** 压倒性优势 — 波动率倒数加权 + 1日预测完美匹配A股高频交易特性
- **止损机制是关键**: StopLoss-3pct-5d (+172%) 远胜 StopLoss-8pct-5d (+15%)，3%止损阈值比8%更有效
- **预测期限越长越危险**: TopK-K3-20d (-88%)、TopK-K3-14d (-91%) 毁灭性亏损，1日/3日预测最可靠
- **每日推送默认使用 RiskParity-VL10-1d [469da] 策略**

> 每笔交易记录（日期/股票/价格/股数/金额）均完整保存在JSON结果中，可逐笔回溯分析。

---

## 可用策略

所有策略均为**纯做多**（A股不允许做空），利用模型6个预测期限（1d/3d/5d/7d/14d/20d）的多信号优势。

| 策略名称 | Tag | 类型 | 说明 |
|----------|-----|------|------|
| TopK-K3-{h} | 5ed5b等 | topk | 等权TopK，K=3 |
| TopK-K10-{h} | 3f274等 | topk | 等权TopK，K=10 |
| StopLoss-{n}pct-{h} | fc99f等 | trailing_stop | 追踪止损，n%=止损阈值 |
| Momentum-LB{n}-{h} | 7ef00等 | momentum_filter | 动量过滤，lookback=n天 |
| RiskParity-VL{n}-{h} | 469da等 | risk_parity | 波动率倒数加权，vol_lookback=n天 |
| Confidence-MC{n}bp-{h} | 5f3f9等 | confidence | 置信度加权，min_confidence=n bp |
| Threshold-{n}pct-{h} | 19888等 | threshold | 预测>n%才入场 |

所有策略tag均为5位MD5 hash（MD5(name:type:horizon:params)[:5]），全局唯一，可用于查询、画图和推送。

> 策略配置保存在 `strategies/configs/diverse_strategies.json`，每个策略tag永久固定。

---

## 常用命令

```bash
# 数据
python -m auto_select_stock.cli fetch-all --start 2018-01-01 [--limit N]
python -m auto_select_stock.cli update-daily [symbols...]
python -m auto_select_stock.cli fetch-financials [--limit N]

# 训练
python -m auto_select_stock.cli train-transformer \
  --seq-len 60 --epochs 20 --batch-size 64 --device cuda \
  --save-path models/price_transformer.pt \
  [--date-window 2022-01-01:2023-01-01]

# 推理
python -m auto_select_stock.cli predict-transformer 600000 \
  --checkpoint models/price_transformer.pt

# 回测
python -m auto_select_stock.cli backtest-transformer --mode topk --top-k 5 ...
python -m auto_select_stock.cli backtest-per-symbol --workers 4 ...

# 多策略（一次运行全部10种）
python -m auto_select_stock.cli backtest-strategies --list
python -m auto_select_stock.cli backtest-strategies \
  --start 2023-01-01 --end 2024-12-31 \
  --checkpoint models/price_transformer.pt \
  --cost-bps 15 --slippage-bps 10

# LLM评分
export OPENAI_API_KEY=your_key
python -m auto_select_stock.cli score --top 50 --provider openai
python -m auto_select_stock.cli render --top 50 --output reports/undervalued.html

# Web界面
python -m auto_select_stock.ops_dashboard
```

---

## 项目结构

```
src/auto_select_stock/
├── cli.py                 # 所有CLI命令
├── storage.py             # SQLite读写 (price & financial表)
├── data_fetcher.py        # 通过akshare获取每日价格
├── financials_fetcher.py  # 季度报告获取
├── scoring.py             # 基于LLM的股票评分
├── ops_dashboard.py       # Web控制面板 (端口8000)
│
└── predict/
    ├── data.py            # 特征工程, npz缓存
    ├── torch_model.py     # PriceTransformer架构
    ├── train.py           # 训练循环（时间窗口划分）
    ├── inference.py       # PricePredictor（批量，可复用）
    ├── backtest.py        # BacktestConfig, run_backtest, _collect_signals_batched
    ├── strategy.py        # build_long_short_portfolio辅助函数
    ├── checkpoints.py     # 检查点保存/加载
    └── strategies/        # v0.0.2: JSON驱动的策略系统
        ├── base.py        # Signal数据类, BaseStrategy抽象类
        ├── __init__.py    # 10种策略实现
        ├── registry.py    # StrategyRegistry (加载JSON配置)
        ├── runner.py     # run_all_strategies_shared (共享信号采集)
        └── configs/
            └── default_strategies.json  # 10种预定义策略
```

---

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `AUTO_SELECT_STOCK_DATA_DIR` | `data/` | 价格/财务数据目录 |
| `AUTO_SELECT_MODEL_DIR` | `models/` | 检查点存储 |
| `AUTO_SELECT_STOCK_PREPROCESSED_DIR` | `data/preprocessed/` | 缓存特征 |
| `AUTO_SELECT_LLM_PROVIDER` | `openai` | LLM提供商 |
| `AUTO_SELECT_LLM_MODEL` | `gpt-4o-mini` | LLM模型 |
| `OPENAI_API_KEY` | — | LLM评分所需 |

**运行前请务必设置 `PYTHONPATH=./src`**。

---

## 注意事项

- 时间窗口训练划分（`--date-window 2022-01-01:2023-01-01`）防止财务报告数据泄露到训练集
- A股T+1交易规则：当日买次日卖（当日不能卖），涉及同日买卖的策略会失败
- CPU机器上的CUDA警告可以忽略；推理会自动回退到CPU
- 使用 `--provider dummy` 可以不调用外部API进行测试

---

[English README](../README.md)
