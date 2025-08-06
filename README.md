# Quant-Strategy-for-Consecutive-Limit-Up-Stocks

> **A live‑traded, daily-frequency 'limit‑up chasing' strategy for China A‑share equities, rigorously validated in trading and consistently delivering stable alpha.**

This repository reproduces the full end‑to‑end pipeline I deployed: transforming raw high‑frequency market data into a robust, production‑grade strategy for capturing consecutive daily limit‑up moves. The code covers signal generation, feature engineering, statistical analytics, factor selection, risk‑adjusted scoring, and rolling backtesting. Each module is fully decoupled at the file level, enabling seamless swapping or extension for research and comparative experiments.
Notably, the pipeline incorporates an XGBoost‑based machine‑learning branch that combines deep factor signals into a composite score using the default hyperparameters (i.e., no parameter tuning has been applied to preserve proprietary feature performance and ensure reproducibility).

Notably, this strategy has been **implemented in live trading**, and results confirm its feasibility and sustained excess returns over multi‑month periods—adding meaningful credibility to the framework.

---

## 1. Pipeline at a Glance

| Stage | File | Core Output |
|-------|------|-------------|
| **1. Data Preparation** | `dataload1.py` | `touch_signals_with_factors_and_returns.pkl` |
| **2. Factor Statistics** | `facstat2_exp1.py` | `factor_statistics_all_2020_2023_time.csv` |
| **3. Factor Screening** | `facfilter3_ex1.py` | `selected_factors_2020_2023_nst.csv` |
| **4. Cross-Sectional Standardisation** | `facstandard4_ex1.py` | `signals_standardized_*_nst.pkl` |
| **5. Signal Scoring & Back-Test** | `stockscore5_ex1.py` | `train_decile_output_*` / `test_decile_output_*` |
| *(Optional)* ML Branch | `xgbtrain4.py` → `xgb_ana5.py` | `xgb_branch_output/…` |

The main pipeline (steps 1→5) relies purely on rule-based scoring. The **XGBoost branch** is an optional machine-learning extension inserted post‑standardisation.

---

## 2. Module Descriptions

### 2.1 Part 1 – Data Preparation (`dataload1.py`)  
- Identifies **first-touch limit-up** triggers, merges minute-level and daily features, computes TWAP execution prices, and calculates forward returns.  
- Produces a serialised PKL including:  
  - `T_date`, `T1_date`, `T2_date`;  
  - High‑dimensional factor matrix (sparse‑friendly dtype);  
  - Execution fields `buy_price`, `sell_price`, `ret`, `is_win`.  

### 2.2 Part 2 – Factor Statistics (`facstat2_exp1.py`)  
- Computes cross-sectional metrics across time: **IC / RankIC**, ICIR, monotonicity, long–short dispersion, and stability—with no initial filtering. Provides comprehensive summary output for later filtering.

### 2.3 Part 3 – Factor Screening (`facfilter3_ex1.py`)  
- Scores factors across multiple dimensions (information content, stability, P&L, win-rate);  
- Automatically infers long/short direction and normalises signs;  
- Eliminates near-duplicate features (|ρ| > 0.98);  
- Applies thresholds and caps to generate a top-N whitelist of factors.

### 2.4 Part 4 – Rank-Z Standardisation (`facstandard4_ex1.py`)  
- Aligns factor direction, then performs daily rank-to‑Z-score transforms to standardise cross-sectionally.  
- Generates distinct train / test signal files.

### 2.5 Part 5 – Signal Scoring & Back-Test (`stockscore5_ex1.py`)  
- Aggregates standardised factors into a composite score; assigns decile portfolios; enforces “limit-up day not buyable” rule; benchmarks excess returns vs CSI‑1000.  
- Outputs include win-rate bar charts, cumulative P&L curves (with annotations), decile summary statistics, and side‑by‑side train vs test results.

### Optional: XGBoost Branch  
- `xgbtrain4.py`: builds an Optuna-tuned, K‑fold gradient boosting model to predict next-day returns.  
- `xgb_ana5.py`: uses identical downstream analytics (deciles, return curves) for apples-to-apples comparison with rule-based scores.

---

## 3. Directory Structure

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

Default data paths (e.g. `/mnt/...`) are placeholders and must be updated based on your environment.

---

## 4. Environment & Dependencies

- Python ≥ 3.9  
- Core libraries: `pandas`, `numpy`, `scipy`, `tqdm`, `matplotlib`, `psutil`, `numba`  
- ML branch: `xgboost >= 2.0`, `optuna`  
- Performance tuning: configure `MAX_WORKERS` and enable `MEMORY_EFFICIENT` flags in `dataload1.py` for faster execution.

A complete `environment.yml` is included for one-step environment reproduction.

---

## 5. Quick Start

```bash
# 1. Prepare raw data (~30–60 min on SSD)
python dataload1.py

# 2. Compute factor statistics
python facstat2_exp1.py

# 3. Screen and select factors
python facfilter3_ex1.py

# 4. Standardise cross-section via Rank→Z
python facstandard4_ex1.py

# 5. Build composite score & run back-test
python stockscore5_ex1.py

To invoke the XGBoost:
```bash
python xgbtrain4.py      # train + predict
python xgb_ana5.py       # performance & decile analysis
```

## 6  Sample Outcomes

| Dataset           | Top-decile win-rate | Top-vs-bottom excess return (cum)     |
| ----------------- | ------------------- | ------------------------------------  |
| **Train 2020-23** | ≈ 50 %              | ≈ 320%
| **Test 2024**     | ≈ 52 %              | ≈ 100%                                |

These results derive from the public rule‑based pipeline (~⅔ of internal fine‑grained backtest P/L). The live trading performance closely aligns with the backtested return trajectory—demonstrating real‑world alpha persistence and adding substantive operational credibility.

## 7  Extension & Customisation Guide

1. Add new factors: drop .fea daily‑level files into configured factor_paths; dataload1.py auto‑imports them.
2. Adjust weights or thresholds: revise SCORE_WEIGHTS / FILTER_THRESHOLDS in facfilter3_ex1.py to customise factor selection logic.
3. Replace the model component: implement your own predictor that exports xgb_pred; xgb_ana5.py will analyse it seamlessly.

This project demonstrates that a well‑engineered, feature‑driven limit‑up chasing strategy can be live‑operational and deliver consistent excess returns in the China A-share universe. Wishing you success in your exploration and strategy development!
















   
