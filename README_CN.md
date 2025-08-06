# Quant-Strategy-for-Consecutive-Limit-Up-Stocks

> **一套已通过实盘验证、日频运行的 A 股「连板追涨」量化策略，能够在真实交易中持续输出稳定 Alpha。**

本仓库完整复刻了本人在内部系统中使用的端到端研究管线：自 **高频原始行情** 出发，逐层打磨为 **可部署的生产级交易策略**，专门捕捉中国股市的连续涨停（“连板”）机会。流程涵盖信号生成、因子工程、统计分析、因子筛选、风险调整打分与滚动回测，各模块 **文件级解耦**，便于在任意环节插拔或横向对比。  
策略还内置一条 **XGBoost 机器学习分支**，将深度因子以默认超参数组合为综合分值

该策略已在真实资金账户运行数月，收益路径与回测高度一致，证实其长期超额收益能力，也为本框架增添了实战含金量

注意，因为一个数据文件超过了github的上传大小限制。所以如果你想跑通整个项目并进行是实验，请务必使用一个网盘链接去下载这个重要的文件。文件名为touch_signals_with_factors_and_returns.pkl，是dataload1.py的输出结果。百度网盘链接如下：通过网盘分享的文件：touch_signals_with_factors_and_returns.pkl
链接: https://pan.baidu.com/s/1ItqxXXWU6aIRcGeFz4hsMw?pwd=1234 提取码: 1234

---

## 1  Pipeline at a Glance

| 阶段 | 对应脚本 | 核心产出 |
|------|-----------|----------|
| **1. 数据准备** | `dataload1.py` | `touch_signals_with_factors_and_returns.pkl` |
| **2. 因子统计** | `facstat2_exp1.py` | `factor_statistics_all_2020_2023_time.csv` |
| **3. 因子筛选** | `facfilter3_ex1.py` | `selected_factors_2020_2023_nst.csv` |
| **4. 截面标准化** | `facstandard4_ex1.py` | `signals_standardized_*_nst.pkl` |
| **5. 打分与回测** | `stockscore5_ex1.py` | `train_decile_output_*` / `test_decile_output_*` |
| *可选* ML 分支 | `xgbtrain4.py` → `xgb_ana5.py` | `xgb_branch_output/…` |

主流程（步骤 1→5）完全基于规则打分；**XGBoost 分支**在标准化后插入。

---

## 2  模块说明

### 2.1  Part 1 – 数据准备 (`dataload1.py`)
- 识别 **首次触及涨停**（10%/20% 日涨跌停规则）事件 :contentReference[oaicite:2]{index=2}，合并分钟与日线特征，计算 TWAP 执行价并生成未来收益。  
- 输出 PKL 含：`T_date / T1_date / T2_date`、高维因子矩阵（稀疏友好 dtype）、以及 `buy_price / sell_price / ret / is_win` 等字段。

### 2.2  Part 2 – 因子统计 (`facstat2_exp1.py`)
- 计算 **IC / RankIC**、ICIR、单调性、长短收益差和稳定性，无任何预过滤，为后续筛选提供全量统计。

### 2.3  Part 3 – 因子筛选 (`facfilter3_ex1.py`)
- 多维评分：信息量、稳定性、P&L、胜率。  
- 自动判定多空方向并统一符号；去除高度相关因子 (|ρ| > 0.98)。  
- 按阈值输出 Top-N 白名单，提高信号稀释度。

### 2.4  Part 4 – Rank-Z 标准化 (`facstandard4_ex1.py`)
- 对齐方向后执行 **每日截面 Rank→Z-score** 转换，保证不同因子可比。  
- 分别生成训练集 / 测试集信号文件。

### 2.5  Part 5 – 信号打分与回测 (`stockscore5_ex1.py`)
- 聚合标准化因子形成综合得分；按得分分成十等分组合；执行“涨停当日不可买”风控；以 CSI-1000 指数衡量超额收益。  
- 产出胜率柱状图、累计收益曲线（含终点标注）和训练 / 测试对比表。

### 可选：XGBoost 分支
- `xgbtrain4.py`：使用默认参数训练梯度提升回归器，对次日收益做预测。  
- `xgb_ana5.py`：沿用与规则打分相同的分组与收益分析流程，便于对照。

---

## 3  目录结构

├── dataload1.py
├── facstat2_exp1.py
├── facfilter3_ex1.py
├── facstandard4_ex1.py
├── stockscore5_ex1.py
├── xgbtrain4.py
├── xgb_ana5.py
├── facfilter3_output_time/
├── facstandard4_output_time/
├── train_decile_output_2020_2023_time/
├── test_decile_output_2024_time/
└── xgb_branch_output/


## 4. 环境与依赖

- Python ≥ 3.9  
- 基础库: pandas, numpy, scipy, tqdm, matplotlib, psutil, numba  
- ML 分支: xgboost >= 2.0, optuna  
- 性能调优: configure MAX_WORKERS and enable MEMORY_EFFICIENT flags in dataload1.py for faster execution.

---

## 5. 快速上手


```bash
# 1. 生成原始信号（SSD 上约 30–60 分钟）
python dataload1.py

# 2. 计算因子统计
python facstat2_exp1.py

# 3. 筛选优质因子
python facfilter3_ex1.py

# 4. 每日截面 Rank→Z 标准化
python facstandard4_ex1.py

# 5. 组合打分并回测
python stockscore5_ex1.py

如需运行 XGBoost 分支：
bash
python xgbtrain4.py      # train + predict
python xgb_ana5.py       # performance & decile analysis
```

## 6  示例结果

| 数据集           | 顶层组合胜率 | 顶累计超额收益     |
| ----------------- | ------------------- | ------------------------------------  |
| **训练 2020-23** | ≈ 50 %              | ≈ 320%
| **测试 2024**     | ≈ 52 %              | ≈ 100%                                |

上述结果来自公开规则路径，约为内部精细回测 P/L 的 ⅔；实盘表现与回测曲线高度契合，验证了 Alpha 的现实可持续性

## 7  拓展与定制指南

1. 新增因子：将 .fea 日线文件放入 factor_paths，dataload1.py 会自动加载
2. 调整权重 / 阈值：在 facfilter3_ex1.py 修改 SCORE_WEIGHTS、FILTER_THRESHOLDS。
3. 替换模型：只需输出 xgb_pred 字段，xgb_ana5.py 即可复用原有分析流程。

经实盘验证，“连板追涨”在合理风控和多因子组合加持下，仍具备可观、持续的超额收益潜力。
祝各位研究顺利、收益长虹！



