
"""
part 5 xgboost 分支
任务：
1. 读取part 4 xgboost分支结果
2. 按每日 xgb_pred 排序分 5 组，输出与 decile_analysis_core 相同的
   累计收益曲线、胜率柱状图、对比表。
"""
import os, json
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

PRED_DIR = './xgb_branch_output'
TRAIN_PKL= f'{PRED_DIR}/xgb_pred_train.pkl'
TEST_PKL = f'{PRED_DIR}/xgb_pred_test.pkl'
SAVE_DIR = f'{PRED_DIR}/analysis'
os.makedirs(SAVE_DIR, exist_ok=True)

def assign_group(df, n_grp=10):
    df = df.sort_values('xgb_pred', ascending=False).reset_index(drop=True)
    if len(df) < n_grp:
        df['group'] = np.arange(1,len(df)+1)
    else:
        df['group'] = pd.qcut(np.arange(len(df)), n_grp, labels=False)+1
    return df

def analyze(ds, name, out_dir):
    ds = ds.groupby('T_date', group_keys=False).apply(assign_group)
    # 胜率
    win = ds.groupby('group')['is_win'].mean()
    plt.figure(figsize=(6,3))
    plt.bar(win.index, win.values)
    plt.axhline(0.5,color='r',ls='--')
    plt.title(f'{name} XGB Score得到的胜率')
    plt.savefig(f'{out_dir}/{name}_winrate.png'); plt.close()

    # 累计收益
    daily_ret = ds.groupby(['T_date','group'])['ret'].mean().unstack('group')
    cum = daily_ret.cumsum()
    plt.figure(figsize=(10,5))
    for g in cum.columns:
        plt.plot(cum.index, cum[g], label=f'G{g}')
        # 终点标注
        last_date = cum.index[-1]
        last_val = cum[g].iloc[-1]
        plt.annotate(
            f'{last_val:.2%}',
            xy=(last_date, last_val),
            xytext=(5, 0),
            textcoords='offset points',
            va='center', fontsize=8, color='black'
        )
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    plt.legend(); plt.title(f'{name} Cumulative Return')
    plt.savefig(f'{out_dir}/{name}_cumret.png'); plt.close()

    # 汇总表
    stat = pd.DataFrame({
        'group': cum.columns,
        'count': [len(ds[ds.group==g]) for g in cum.columns],
        'winrate': win.values,
        'avg_ret': [ds[ds.group==g]['ret'].mean() for g in cum.columns],
        'sharpe' : [ds[ds.group==g]['ret'].mean()/ds[ds.group==g]['ret'].std()
                    if ds[ds.group==g]['ret'].std()>0 else 0 for g in cum.columns]
    })
    stat.to_csv(f'{out_dir}/{name}_summary.csv', index=False)
    return stat, win, cum

train = pd.read_pickle(TRAIN_PKL)
test  = pd.read_pickle(TEST_PKL)

train_stat, train_win, train_cum = analyze(train,'train_2020_2023',SAVE_DIR)
test_stat , test_win , test_cum  = analyze(test ,'test_2024',SAVE_DIR)

# 训练测试对比
compare = pd.DataFrame({
    'group': train_stat['group'],
    'train_winrate': train_stat['winrate'],
    'test_winrate' : test_stat ['winrate'],
    'train_avg_ret': train_stat['avg_ret'],
    'test_avg_ret' : test_stat ['avg_ret'],
    'winrate_diff' : test_stat['winrate'] - train_stat['winrate']
})
compare.to_csv(f'{SAVE_DIR}/train_test_compare.csv', index=False)
