# PriceTransformer 模型详细文档

## 模型架构

```
输入: (batch, seq_len=252, 32)
  ↓
input_proj: Linear(32 → 256)
  ↓
正弦位置编码 (Sinusoidal PE)
  ↓
TransformerEncoder: 10层, 8头, dim_ffn=512
  ↓
6× Regression Heads + 6× Classification Heads
  ↓
输出: reg_all (6, batch, seq_len), cls_all (6, batch, seq_len)
```

## 输入特征详解 (32维)

### 1. 价格特征 (Price) — 11维

| 序号 | 特征名 | 说明 | 数据来源 |
|------|--------|------|----------|
| 1 | `open` | 开盘价 | 行情数据 |
| 2 | `high` | 最高价 | 行情数据 |
| 3 | `low` | 最低价 | 行情数据 |
| 4 | `close` | 收盘价 | 行情数据 |
| 5 | `volume` | 成交量（股） | 行情数据 |
| 6 | `amount` | 成交金额（元） | 行情数据 |
| 7 | `turnover_rate` | 换手率 (%) = 成交量/流通股本×100 | 行情数据 |
| 8 | `volume_ratio` | 量比 = 当日成交量/前5日平均成交量 | 行情数据 |
| 9 | `pct_change` | 涨跌幅 (%) = (收盘价-前收盘)/前收盘×100 | 行情数据 |
| 10 | `amplitude` | 振幅 (%) = (最高价-最低价)/前收盘价×100 | 行情数据 |
| 11 | `change_amount` | 涨跌额（元） = 收盘价-前收盘价 | 行情数据 |

### 2. 财务特征 (Financial) — 7维

| 序号 | 特征名 | 说明 | 数据来源 |
|------|--------|------|----------|
| 12 | `roe` | 净资产收益率 (%) = 净利润/净资产×100 | 季报/年报 |
| 13 | `net_profit_margin` | 净利率 (%) = 净利润/营业收入×100 | 季报/年报 |
| 14 | `gross_margin` | 毛利率 (%) = (营业收入-营业成本)/营业收入×100 | 季报/年报 |
| 15 | `operating_cashflow_growth` | 经营现金流增长率 (%) | 季报/年报 |
| 16 | `debt_to_asset` | 资产负债率 (%) = 总负债/总资产×100 | 季报/年报 |
| 17 | `eps` | 每股收益（元） = 净利润/总股本 | 季报/年报 |
| 18 | `operating_cashflow_per_share` | 每股经营现金流（元） = 经营现金流/总股本 | 季报/年报 |

**注**：财务数据从季报/年报回填到每个交易日，未发布新财报前使用最近一期数据（Backward-filled）。

### 3. 技术指标 (Technical) — 14维

| 序号 | 特征名 | 说明 | 计算公式 |
|------|--------|------|----------|
| 19 | `rsi_14` | 相对强弱指数 | RSI = 100 - 100/(1+RS)，RS = 14日平均涨幅/14日平均跌幅 |
| 20 | `macd_line` | MACD主线 | EMA12 - EMA26（12日与26日指数移动平均线差值） |
| 21 | `macd_signal` | MACD信号线 | MACD的9日EMA |
| 22 | `macd_hist` | MACD柱状图 | MACD - Signal |
| 23 | `bb_position` | 布林带位置 | (close - BB_lower) / (BB_upper - BB_lower)，取值0~1 |
| 24 | `bb_width` | 布林带宽度 | (BB_upper - BB_lower) / BB_middle |
| 25 | `volume_ma5` | 成交量MA5比值 | volume / MA5(volume) |
| 26 | `volume_ma20` | 成交量MA20比值 | volume / MA20(volume) |
| 27 | `atr_14` | 平均真实波幅 | 14日ATR = TrueRange的14日移动平均 |
| 28 | `stoch_k` | 随机指标K | 100 × (close - LL14) / (HH14 - LL14)，LL14=14日最低价，HH14=14日最高价 |
| 29 | `stoch_d` | 随机指标D | stoch_k的3日移动平均 |
| 30 | `obv_ma10` | OBV均线比值 | OBV / MA10(OBV)，OBV = 累计成交量（上涨加，下跌减） |
| 31 | `roc_10` | 变动率 | (close - close_10d_ago) / close_10d_ago × 100 |
| 32 | `momentum_10` | 动量 | close - close_10d_ago |

## 技术指标详细说明

### RSI (Relative Strength Index)
- **用途**：衡量价格变动的速度和幅度，判断超买超卖
- **周期**：14日
- **范围**：0~100
- **解读**：>70 超买，<30 超卖

### MACD (Moving Average Convergence Divergence)
- **用途**：趋势追踪，动量指标
- **组成**：
  - MACD线 = EMA12 - EMA26
  - Signal线 = MACD的9日EMA
  - Histogram = MACD - Signal
- **解读**：MACD上穿Signal金叉，下穿死叉

### Bollinger Bands (布林带)
- **用途**：衡量价格波动范围，识别突破信号
- **组成**：
  - 中轨 = MA20(close)
  - 上轨 = 中轨 + 2×标准差
  - 下轨 = 中轨 - 2×标准差
- **bb_position**：价格在布林带中的位置，接近0为触底，接近1为触顶

### Stochastic Oscillator (随机指标)
- **用途**：在趋势内识别转折点
- **组成**：%K线（快线）和%D线（慢线）
- **解读**：>80 超买，<20 超卖

### ATR (Average True Range)
- **用途**：衡量市场波动性，用于止损设置
- **计算**：TR = max(H-L, |H-PC|, |L-PC|)，ATR = TR的14日均值

### OBV (On-Balance Volume)
- **用途**：成交量加权的累计指标，验证价格趋势
- **计算**：上涨日加成交量，下跌日减成交量

### ROC & Momentum
- **ROC (Rate of Change)**：价格变化率
- **Momentum**：价格动量，绝对值变化
- **用途**：识别价格趋势的加速/减速

## 数据处理流程

```
原始价格数据 (akshare)
    ↓
SQLite: price 表 (后复权价格)
    ↓
特征工程 (predict/data.py)
    ↓ 1. 基础价格特征 (11维)
    ↓ 2. 财务数据回填 (7维)
    ↓ 3. 技术指标计算 (14维，无未来数据泄露)
    ↓
合并 → 32维特征向量
    ↓
归一化 (StreamingStandardScaler)
    ↓
模型输入: (batch, 252, 32)
```

## 预测输出

模型输出6个预测期限的预测值：

| 期限 | 用途 | 适用策略 |
|------|------|----------|
| 1d | 次日收益预测 | 日内交易、短期择时 |
| 3d | 3日收益预测 | 短线交易 |
| 5d | 5日收益预测 | 中线交易 |
| 7d | 7日收益预测 | 中线交易 |
| 14d | 14日收益预测 | 中长线交易 |
| 20d | 20日收益预测 | 长线交易 |

每个期限输出：
- **回归值**：预测对数收益率 `pred_log_return`
- **分类值**：上涨概率 `pred_direction = sigmoid(cls)`

## 损失函数

```
loss = λ_reg · MSE(pred_log_return, real_log_return)
     + λ_cls · BCE(pred_direction, real_up/down)
     + λ_rank · RankingLoss(pred_return pairs)

λ_reg = 0.1   (回归损失权重低)
λ_cls = 10.0  (分类损失主导)
λ_rank = 1.0  (排序损失辅助)
```

**设计原因**：
- 分类损失主导：A股市场方向比幅度更容易预测
- 回归损失低：精确收益率预测难度大
- 排序损失：优化股票间相对排序，对策略排名至关重要
