
"""
part 5 v 4.2  触板信号综合打分 & 回测
任务：
1. 涨停不可买置零
2. 指数超额收益
3. 股票分组数 GROUP_N 随便改
"""

import os, json, warnings
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings('ignore')

# ====== 全局参数 =======
GROUP_N = 10                     
APPLY_LIMITUP_ZERO = True         # ← 涨停不可买信号收益置 0

# ======= 路径 =========
train_path  = './facstandard4_output/signals_standardized_train_2020_2023_nst.pkl'
test_path = './facstandard4_output/signals_standardized_test_2024_nst.pkl'
factor_label_path = './facfilter3_output/monotonic_factor_group_result_2020_2023_nst.csv'

index_path   = '/mnt/tonglian_data2/support_data/中证1000_min_ts.fea'
index_time1, index_time2 = 931, 1459

limitup_path = '/mnt/tonglian_data2/ohlc_fea/LIMIT_UP_PRICE.fea'

train_save_dir = './train_decile_output_2020_2023_nst_0'
test_save_dir = './test_decile_output_2024_nst_0'
compare_save_dir = './compare_train_test_output_nst_0'
for d in (train_save_dir, test_save_dir, compare_save_dir):
    os.makedirs(d, exist_ok=True)

# 1. 指数收益
def calculate_index_returns(signals: pd.DataFrame) -> pd.Series:
    if not os.path.exists(index_path):
        return pd.Series(0, index=signals['T_date'].unique())

    idx_df = pd.read_feather(index_path)
    idx_df['date'] = pd.to_datetime(idx_df['date'].astype(str))
    idx_df['time'] = idx_df['time'].astype(int)
    idx_df = idx_df[(idx_df['time'] >= index_time1) & (idx_df['time'] <= index_time2)]
    daily_twap = idx_df.groupby('date')['price'].mean()

    ret_map = {}
    for t_date, grp in signals.groupby('T_date'):
        t1, t2 = grp['T1_date'].iloc[0], grp['T2_date'].iloc[0]
        if (t1 in daily_twap.index) and (t2 in daily_twap.index):
            ret_map[t_date] = (daily_twap[t2] - daily_twap[t1]) / daily_twap[t1]
        else:
            ret_map[t_date] = 0.0
    return pd.Series(ret_map)

# 2. 分组
def assign_decile_groups(signals: pd.DataFrame, z_cols: list, group_n: int = GROUP_N):
    signals = signals.copy()
    signals['score'] = signals[z_cols].sum(axis=1)

    def _group(df):
        df = df.sort_values('score', ascending=False).reset_index(drop=True)
        n = len(df)
        if n < group_n:
            df['decile'] = np.arange(1, n + 1)
        else:
            bins = np.linspace(0, n, group_n + 1).astype(int)
            bins = np.unique(bins)
            labels = list(range(1, len(bins)))
            df['decile'] = pd.cut(np.arange(n), bins=bins,
                                  labels=labels, include_lowest=True,
                                  duplicates='drop').astype(int)
        return df

    signals = signals.groupby('T_date', group_keys=False).apply(_group)
    signals['decile'] = signals['decile'].astype(int)
    return signals

# 3. 涨停价置零
def load_limitup_long():
    lup = pd.read_feather(limitup_path)
    lup['TRADE_DATE'] = pd.to_datetime(lup['TRADE_DATE'])
    lup_long = lup.melt(id_vars='TRADE_DATE', var_name='stock',
                        value_name='limitup_price')
    lup_long.rename(columns={'TRADE_DATE': 'T1_date'}, inplace=True)
    lup_long['stock'] = lup_long['stock'].astype(str)
    return lup_long[['T1_date', 'stock', 'limitup_price']]

def apply_limitup_zero(signals: pd.DataFrame, lup_long: pd.DataFrame,
                       apply_flag: bool = True):
    if not apply_flag or lup_long is None:
        return signals
    signals = signals.merge(lup_long, on=['T1_date', 'stock'], how='left')
    signals['is_unbuyable'] = signals['buy_price'] >= signals['limitup_price']
    cnt = signals['is_unbuyable'].sum()
    if cnt:
        signals.loc[signals['is_unbuyable'], ['ret', 'is_win']] = 0
    print(f'置零不可买信号 {cnt:,} ')
    return signals

# 4. 分析
def information_ratio(e: pd.Series, ann=252):
    mu = e.mean() * ann
    sd = e.std() * np.sqrt(ann)
    return np.nan if sd == 0 else mu / sd

def analyze_performance(signals: pd.DataFrame, title: str, out_dir: str):
    deciles = sorted(signals['decile'].unique())
    total_win = signals.groupby('decile')['is_win'].mean().reindex(deciles)

    daily_ret = signals.groupby(['T_date', 'decile'])['ret'] \
                       .mean().unstack('decile').sort_index()
    cum_ret = daily_ret.cumsum()

    daily_exc = signals.groupby(['T_date', 'decile'])['excess_ret'] \
                       .mean().unstack('decile').sort_index()
    cum_exc = daily_exc.cumsum()

    #  图 
    plt.figure(figsize=(8, 4))
    plt.bar(total_win.index, total_win.values)
    plt.axhline(0.5, ls='--', c='r', alpha=.5)
    plt.title(f'{title} | Winrate'); plt.xlabel('Decile'); plt.ylabel('Winrate')
    plt.tight_layout(); plt.savefig(f'{out_dir}/winrate.png', dpi=120); plt.close()

    for data, name in [(cum_ret, 'cum_return'), (cum_exc, 'cum_excess_return')]:
        plt.figure(figsize=(10, 5))
        for dec in data.columns:
            plt.plot(data.index, data[dec], label=f'D{dec}')
            # 获取该decile的最后时间点和累计收益
            last_date = data.index[-1]
            last_val = data[dec].iloc[-1]
            # 在曲线末端标注终值
            plt.annotate(
                f'{last_val:.2%}', 
                xy=(last_date, last_val), 
                xytext=(5, 0), 
                textcoords='offset points',
                va='center', fontsize=8, color='black'
            )
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
        plt.title(f'{title} | {name.replace("_"," ").title()}')
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/{name}.png', dpi=120)
        plt.close()


    #  摘要 
    summary_rows = []
    for d in deciles:
        sub = signals[signals['decile'] == d]
        ir = information_ratio(daily_exc[d].dropna()) if d in daily_exc else np.nan
        summary_rows.append({
            'decile': d,
            'sample_cnt': len(sub),
            'winrate': total_win.loc[d],
            'avg_ret': sub['ret'].mean(),
            'avg_excess_ret': sub['excess_ret'].mean(),
            'sharpe': sub['ret'].mean() / sub['ret'].std() if sub['ret'].std() else np.nan,
            'information_ratio': ir
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(f'{out_dir}/decile_summary_stats.csv', index=False)
    signals.to_csv(f'{out_dir}/scored_signals.csv', index=False)
    return summary

# 5. 主流程
def main():
    # 读取因子方向
    mono = pd.read_csv(factor_label_path)
    z_cols = [f'{f}_z' for f in mono['factor']]

    # 读取信号
    train = pd.read_pickle(train_path)
    test  = pd.read_pickle(test_path)

    # 应用涨停价置零
    lup_long = load_limitup_long()
    train = apply_limitup_zero(train, lup_long, APPLY_LIMITUP_ZERO)
    test  = apply_limitup_zero(test,  lup_long, APPLY_LIMITUP_ZERO)

    # 指数收益 & 超额
    idx_train = calculate_index_returns(train)
    train['index_ret']  = train['T_date'].map(idx_train).fillna(0)
    train['excess_ret'] = train['ret'] - train['index_ret']

    idx_all = calculate_index_returns(pd.concat([train[['T_date','T1_date','T2_date']],
                                                 test[['T_date','T1_date','T2_date']]]))
    test['index_ret']  = test['T_date'].map(idx_all).fillna(0)
    test['excess_ret'] = test['ret'] - test['index_ret']

    # 分组
    train = assign_decile_groups(train, z_cols, GROUP_N)
    test  = assign_decile_groups(test , z_cols, GROUP_N)

    # 分析
    tr_sum = analyze_performance(train, '训练集 2020-23', train_save_dir)
    te_sum = analyze_performance(test,  '测试集 2024',   test_save_dir)

    # 对比表
    compare_df = tr_sum[['decile','winrate','avg_ret','avg_excess_ret']] \
        .merge(te_sum[['decile','winrate','avg_ret','avg_excess_ret']],
               on='decile', suffixes=('_train','_test'))
    compare_df['winrate_diff'] = compare_df['winrate_test'] - compare_df['winrate_train']
    compare_df.to_csv(f'{compare_save_dir}/train_test_comparison.csv', index=False)

    print(compare_df.round(4).to_string(index=False))

    with open(f'{compare_save_dir}/summary.json', 'w') as f:
        json.dump({
            'train_signals': int(len(train)),
            'test_signals':  int(len(test)),
            'train_avg_excess': float(train['excess_ret'].mean()),
            'test_avg_excess':  float(test['excess_ret'].mean()),
            'group_n': GROUP_N,
            'apply_limitup_zero': APPLY_LIMITUP_ZERO
        }, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
