# Behavior Cloning (BC) 模型详解

## 概述

BC (Behavior Cloning) 模型是一种模仿学习方法，通过学习专家交易策略的轨迹来实现智能交易。与传统的监督学习不同，BC专注于从专家演示中学习状态到动作的映射。

**相关文件**: `src/auto_select_stock/rl/bc_pretrain_trainer.py`

## 模型架构

```python
class BCNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),     # 6 -> 128
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),   # 128 -> 128
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), # 128 -> 64
            nn.ReLU(),
        )
        # 输出层：position size [0, 1]
        self.action_head = nn.Linear(hidden_dim // 2, 1)
        self.action_head.weight.data.fill_(0.0)  # 初始化为0
        self.action_head.bias.data.fill_(0.0)
    
    def forward(self, x):
        features = self.net(x)
        action = torch.sigmoid(self.action_head(features))  # [0, 1]
        return action
```

**网络结构**: 3层全连接网络 + LayerNorm + ReLU激活函数

| 层 | 输入 | 输出 | 参数 |
|---|---|---|---|
| FC1 | 6 | 128 | 896 |
| FC2 | 128 | 128 | 16,512 |
| FC3 | 128 | 64 | 8,256 |
| Action Head | 64 | 1 | 65 |
| **总计** | | | **~26K参数** |

## 输入状态 (6维)

```python
state = [
    price_level - 1.0,      # 股价/MA20 - 1, 归一化价格水平
    ret_1d,                 # 1日对数收益
    ret_5d,                 # 5日对数收益  
    position / 0.15,        # 当前持仓/最大持仓, 归一化
    unrealized_pnl,         # 未实现盈亏比例
    0.0,                   # 现金占位符
]
```

| 特征 | 描述 | 范围 |
|------|------|------|
| price_level - 1 | 相对MA20的价格水平 | ~[-0.5, 0.5] |
| ret_1d | 1日对数收益 | ~[-0.1, 0.1] |
| ret_5d | 5日对数收益 | ~[-0.3, 0.3] |
| position/0.15 | 归一化持仓 | [0, 1] |
| unrealized_pnl | 未实现盈亏 | ~[-0.5, 0.5] |
| cash | 现金占位 | 0 |

## 专家轨迹收集 (Expert Trajectory Collection)

### 核心思想

用规则模拟"专家"交易行为，收集大量(状态, 动作)对，然后让神经网络学习模仿。

### 专家规则

```python
def _get_expert_action(position, pred_ret, current_price, entry_price, ...):
    unrealized_pnl = (current_price - entry_price) / entry_price
    
    # 止损规则：亏5%止损
    if position > 0.01 and unrealized_pnl < -0.05:
        return 0.0
    
    # 卖出规则  
    if position > 0.01:
        if pred_ret < sell_threshold:  # 预测收益负
            return 0.0
        elif unrealized_pnl > 0.15:   # 盈利15%止盈
            return 0.0
    
    # 买入规则
    if position < 0.01:
        if pred_ret > buy_threshold:   # 预测收益正
            return max_position * 0.8   # 80%仓位买入
    
    # 持有或等待
    if position > 0.01:
        return position  # 保持当前仓位
    else:
        return 0.0      # 空仓等待
```

### 预测信号

使用简单动量预测作为专家的"判断依据"：

```python
def _compute_momentum_predictions(seq):
    preds = []
    for i in range(start_idx, len(seq) - 1):
        if i >= 5:
            mom = (seq[i, 3] - seq[i-5, 3]) / seq[i-5, 3]
        else:
            mom = 0.0
        preds.append(mom)
    return np.array(preds)
```

## 训练过程

### 数据收集

```python
# 从150只股票收集专家轨迹
collector = ExpertTrajectoryCollector(train_sequences, train_preds)
trajectories = collector.collect_trajectories(
    buy_threshold=0.01,   # 预测收益>1%买入
    sell_threshold=-0.015 # 预测收益<-1.5%卖出
)

# 收集结果
# 135,825 个 (state, action) 对
# Action 分布: mean=0.059, std=0.060, range=[0, 0.12]
```

### 训练配置

```python
model = BCNetwork(input_dim=6, hidden_dim=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

batch_size = 256
num_epochs = 50
```

### 损失函数

使用MSE损失最小化预测position与专家position的差异：

```python
pred_actions = model(batch_states)  # 模型输出 [0, 1]
loss = F.mse_loss(pred_actions, batch_actions)  # MSE
```

## Backtest交易逻辑

```python
trade_threshold = 0.05  # 只有模型预测>5%才开仓

action = model(state_t).item()  # 模型直接输出position

# 开仓：预测>阈值 且 当前空仓
if action > trade_threshold and position < 0.01:
    position = action
    trade_count += 1

# 平仓：预测<阈值 且 当前持仓
if action < trade_threshold and position > 0.01:
    position = 0.0

# 持仓期间收益
portfolio *= (1.0 + actual_return * position)
```

## 训练结果

### 训练指标

| Epoch | Train Loss | Val Loss | Val Correlation |
|-------|-----------|----------|----------------|
| 5 | 0.000964 | 0.000742 | 0.8914 |
| 10 | 0.000917 | 0.000717 | 0.8944 |
| 25 | 0.000880 | 0.000703 | 0.8964 |
| 50 | 0.000867 | 0.000695 | **0.8980** |

### 模型输出分布

| 统计量 | 值 |
|--------|------|
| Min | 0.000281 |
| Max | 0.127105 |
| Mean | 0.033530 |
| Median | 0.002878 |
| >0.05 比例 | 29.5% |

### 回测结果 (2024-01-01 ~ 2025-12-31)

| 指标 | 值 |
|------|------|
| **Total Return** | **+56.39%** |
| Sharpe Ratio | 17.5 |
| Trade Count | 2,053 |
| Avg Position | 4.32% |

## 模型输出解读

模型的sigmoid输出范围是[0, 1]，对应position范围[0, 0.12] (12%最大仓位)。

- **输出 < 0.05**: "等待"信号，空仓
- **输出 > 0.05**: "买入"信号，开仓买入
- **输出从高变低**: "卖出"信号，平仓

## 与PriceTransformer的关系

| 方面 | PriceTransformer | BCNetwork |
|------|-----------------|-----------|
| 任务 | 预测收益率 | 预测交易position |
| 输入 | 32维特征(价格+财务+技术) | 6维特征(价格+持仓) |
| 输出 | 6个期限的收益率预测 | position [0, 1] |
| 训练方式 | 监督学习(回归+分类+排序) | 模仿学习(BC) |
| 用途 | 策略排序、信号生成 | 端到端交易决策 |

BC模型可以作为独立的交易策略，也可以与PriceTransformer结合使用：
- PriceTransformer提供市场预测
- BCNetwork学习在给定市场状态下的最优仓位

## 相关文件

- `src/auto_select_stock/rl/bc_pretrain_trainer.py` - BC训练器
- `src/auto_select_stock/rl/bc_rl_finetune.py` - BC+RL微调实验
- `src/auto_select_stock/rl/rrl_sharpe_trainer.py` - RRL直接优化
- `src/auto_select_stock/rl/sac_pt_trainer.py` - SAC+PT预测

## 后续优化方向

1. **RL微调**: 用BC作为预训练，再用RL微调（实验显示RL反而降低性能）
2. **集成**: 将BC与ConfidenceStrategy集成
3. **多任务**: 同时预测方向和position
4. **更丰富的状态**: 添加市场情绪、资金流向等特征
