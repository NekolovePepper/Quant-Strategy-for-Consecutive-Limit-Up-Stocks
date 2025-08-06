
"""
part 4 v3 因子标准化
任务：
1. 从 touch_signals_with_factors_and_returns.pkl 切分（之前忘切了
   不过分析因子时用的训练数据所以放现在切不影响）
2. 对 part 3 v3 输出的最终因子做横截面Rank 标准化
3. 输出标准化后的 train / test pkl。

"""
import os, sys
import pandas as pd, numpy as np
from tqdm import tqdm

# ======== 文件路径 ============
TRAIN_PKL   = './facstandard4_output/signals_merged_train_2020_2023.pkl'
TEST_PKL    = './facstandard4_output/signals_merged_test_2024.pkl'
FULL_PKL    = 'touch_signals_with_factors_and_returns.pkl'  # part1 输出
FACTOR_CSV  = './facfilter3_output/monotonic_factor_group_result_2020_2023_nst.csv'  # ← 3 v2 标签
OUT_TRAIN   = './facstandard4_output/signals_standardized_train_2020_2023_nst.pkl'
OUT_TEST    = './facstandard4_output/signals_standardized_test_2024_nst.pkl'

TRAIN_YEARS = [2020,2021,2022,2023]
TEST_YEARS  = [2024]

# ============== Rank → 正态化 标准化 =========
def rank_standardize(df, factors):
    from scipy import stats
    for fac in tqdm(factors, desc='Rank 标准化'):
        if fac not in df.columns:
            df[f'{fac}_z'] = 0
            continue
        def _rank2z(g):
            if len(g) < 2:
                return pd.Series(0, index=g.index)
            pct = g.rank(pct=True, method='average')
            return stats.norm.ppf(pct.clip(0.001,0.999))
        df[f'{fac}_z'] = df.groupby('T_date')[fac].transform(_rank2z).fillna(0)
    return df

# =========== 主流程 ===========
def main():
    #  0. 准备 train/test 宽表 
    if os.path.exists(TRAIN_PKL) and os.path.exists(TEST_PKL):
        train = pd.read_pickle(TRAIN_PKL)
        test  = pd.read_pickle(TEST_PKL)
    else:
        if not os.path.exists(FULL_PKL):
            sys.exit(f' 找不到 {FULL_PKL}，检查第一步文件qwq')
        full = pd.read_pickle(FULL_PKL)
        full['T_date'] = pd.to_datetime(full['T_date'])
        full['year']   = full['T_date'].dt.year
        train = full[full['year'].isin(TRAIN_YEARS)].copy()
        test  = full[full['year'].isin(TEST_YEARS)].copy()
        train.to_pickle(TRAIN_PKL); print(f'  保存 {TRAIN_PKL}')
        test .to_pickle(TEST_PKL ); print(f'  保存 {TEST_PKL}')

    #  1. 读取因子标签 
    if not os.path.exists(FACTOR_CSV):
        sys.exit(f' 找不到 {FACTOR_CSV}，先运行 part3_factor_screening_v2.py')
    mono = pd.read_csv(FACTOR_CSV)
    factor_labels = dict(zip(mono['factor'], mono['label']))
    factors = list(factor_labels.keys())
    print(f' {len(factors)} 因子标准化')

    #  2. 按标签调方向 
    for fac, lab in factor_labels.items():
        for df in (train, test):
            if fac not in df.columns:
                df[fac] = np.nan
            if lab == 's':
                df[fac] = -df[fac]

    #  3. Rank 标准化 
    train = rank_standardize(train, factors)
    test  = rank_standardize(test , factors)

    #  4. 保存 
    train.to_pickle(OUT_TRAIN)
    test .to_pickle(OUT_TEST )
    print(f'完成:\n  → {OUT_TRAIN}\n  → {OUT_TEST}')

if __name__ == '__main__':
    main()
