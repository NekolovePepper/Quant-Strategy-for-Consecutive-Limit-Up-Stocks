
"""
part2 ：因子统计 v3
任务：
1. Rank_IC计算（基于分位数排名）
2. 改进单调性检测（分箱）
3. 增加信息系数(icir/rankicir)的t统计量
"""
import os, gc, json, psutil, warnings, time
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from scipy.stats import spearmanr, pearsonr, ttest_ind, ttest_1samp, norm

warnings.filterwarnings('ignore')

# ======= 参数 ======
# 测试集！！！！！！！！！！
TRAIN_YEARS = [2020, 2021, 2022, 2023]
INPUT_DATA   = 'touch_signals_with_factors_and_returns.pkl'
OUTPUT_STATS = 'factor_group_stats_2020_2023_v3.csv'
OUTPUT_SUM   = 'factor_statistics_summary_v3.json'

N_GROUP = 5                 # 分箱个数 
MIN_VALID_SAMPLE = 100        # 每个因子最少非空样本
MIN_DATES = 10               # 最少交易日数

def mem(label=''):
    rss = psutil.Process().memory_info().rss / 1024 / 1024 / 1024
    print(f'[MEM] {label}  {rss:6.2f} GB')

def calc_rank_ic(factor_vals, return_vals):

    if len(factor_vals) < 5:
        return np.nan
    
    # 转为排名
    factor_ranks = pd.Series(factor_vals).rank(pct=True, method='average')
    return_ranks = pd.Series(return_vals).rank(pct=True, method='average')
    
    # 相关系数
    ic, _ = spearmanr(factor_ranks, return_ranks)
    return ic if np.isfinite(ic) else np.nan

def detect_monotonicity(group_returns):
    # 检测单调性：('ascending', 'descending', 'none', 'u_shape', 'inverted_u')
    if len(group_returns) < 3:
        return 'none'
    
    returns = np.array(group_returns)
    
    # 严格单调性
    if np.all(np.diff(returns) > 0):
        return 'ascending'
    elif np.all(np.diff(returns) < 0):
        return 'descending'
    
    # U型或倒U型
    mid_idx = len(returns) // 2
    if len(returns) >= 5:
        left_trend = returns[mid_idx] - returns[0]
        right_trend = returns[-1] - returns[mid_idx]
        
        if left_trend < 0 and right_trend > 0:  # U型
            return 'u_shape'
        elif left_trend > 0 and right_trend < 0:  # 倒U型
            return 'inverted_u'
    
    # 弱单调性
    diff_signs = np.sign(np.diff(returns))
    if np.sum(diff_signs > 0) >= len(diff_signs) * 0.7:
        return 'weak_ascending'
    elif np.sum(diff_signs < 0) >= len(diff_signs) * 0.7:
        return 'weak_descending'
    
    return 'none'

def calc_stats(train_df, factor_cols):
    rows = []
    date_list = sorted(train_df['T_date'].unique())
    T_dates = train_df['T_date'].values

    # 提取常用列
    rets = train_df['ret'].values
    wins = train_df['is_win'].values

    for fac in tqdm(factor_cols, desc='统计因子'):
        start_t = time.time()
        series = train_df[fac].values
        mask_valid = ~np.isnan(series)
        valid = series[mask_valid]
        # 至少这么多非空样本
        if valid.size < MIN_VALID_SAMPLE:
            continue

        # 1. 全局分箱
        q = np.nanpercentile(valid, np.linspace(0, 100, N_GROUP + 1))
        q = np.unique(q)
        if (len(q) - 1 < 2) or (not np.all(np.diff(q) > 0)):
            continue

        # 分箱标签
        grp = np.full_like(series, fill_value=-1, dtype=int)
        not_nan_idx = np.where(~np.isnan(series))[0]
        grp[not_nan_idx] = np.searchsorted(q, series[not_nan_idx], side='right') - 1
        grp[grp < 0] = 0
        grp[grp >= len(q)-1] = len(q)-2

        # 分组统计
        grp_stats = []
        for g in range(len(q)-1):
            idx = np.where(grp == g)[0]
            if idx.size == 0:
                grp_stats.append({'cnt':0,'win':np.nan,'ret':np.nan})
            else:
                grp_stats.append({
                    'cnt': idx.size,
                    'win': np.nanmean(wins[idx]),
                    'ret': np.nanmean(rets[idx])
                })

        # 提取各组收益率给单调性检测
        group_returns = [stat['ret'] for stat in grp_stats if not np.isnan(stat['ret'])]
        monotonicity = detect_monotonicity(group_returns)

        # 2. Top-Bottom 组合
        top_idx = np.where(grp == len(q)-2)[0]
        bot_idx = np.where(grp == 0)[0]
        if (top_idx.size < 5) or (bot_idx.size < 5):
            continue

        win_diff = np.nanmean(wins[top_idx]) - np.nanmean(wins[bot_idx])
        ret_diff = np.nanmean(rets[top_idx]) - np.nanmean(rets[bot_idx])
        
        # T检验
        t_stat_ret, p_ret = ttest_ind(rets[top_idx], rets[bot_idx], 
                                     equal_var=False, nan_policy='omit')
        t_stat_win, p_win = ttest_ind(wins[top_idx], wins[bot_idx], 
                                     equal_var=False, nan_policy='omit')

        #  3. IC / RankIC / ICIR 
        ic_by_day = []
        rank_ic_by_day = []
        
        for d in date_list:
            idx = (T_dates == d)
            fac_d = series[idx]
            ret_d = rets[idx]
            
            if np.isnan(fac_d).all() or np.isnan(ret_d).all() or fac_d.size < 5:
                continue
                
            # 传统IC (Spearman)
            ic, _ = spearmanr(fac_d, ret_d)
            if np.isfinite(ic):
                ic_by_day.append(ic)
            
            # RankIC
            rank_ic = calc_rank_ic(fac_d, ret_d)
            if np.isfinite(rank_ic):
                rank_ic_by_day.append(rank_ic)

        ic_by_day = np.array(ic_by_day)
        rank_ic_by_day = np.array(rank_ic_by_day)

        if len(ic_by_day) < MIN_DATES or len(rank_ic_by_day) < MIN_DATES:
            continue

        # IC统计
        ic_mean = ic_by_day.mean()
        ic_std = ic_by_day.std(ddof=1) if len(ic_by_day) > 1 else 0
        icir = 0 if ic_std == 0 else ic_mean / ic_std * np.sqrt(len(ic_by_day))
        t_ic, p_ic = ttest_1samp(ic_by_day, 0.0, nan_policy='omit')

        # RankIC统计
        rank_ic_mean = rank_ic_by_day.mean()
        rank_ic_std = rank_ic_by_day.std(ddof=1) if len(rank_ic_by_day) > 1 else 0
        rank_icir = 0 if rank_ic_std == 0 else rank_ic_mean / rank_ic_std * np.sqrt(len(rank_ic_by_day))
        t_rank_ic, p_rank_ic = ttest_1samp(rank_ic_by_day, 0.0, nan_policy='omit')

        #  4. 稳定性指标 （之后再反转）
        # IC>0的比例
        ic_positive_ratio = np.mean(ic_by_day > 0) if len(ic_by_day) > 0 else 0
        rank_ic_positive_ratio = np.mean(rank_ic_by_day > 0) if len(rank_ic_by_day) > 0 else 0

        #  5. Wilson 置信区间（Top 组） 
        top_win = np.nanmean(wins[top_idx])
        n_top = top_idx.size
        if n_top > 0:
            z = norm.ppf(0.975)   # 95%
            wilson_low = (top_win + z**2/(2*n_top) - z*np.sqrt((top_win*(1-top_win)+z**2/(4*n_top))/n_top)) / (1+z**2/n_top)
            wilson_high = (top_win + z**2/(2*n_top) + z*np.sqrt((top_win*(1-top_win)+z**2/(4*n_top))/n_top)) / (1+z**2/n_top)
        else:
            wilson_low = wilson_high = np.nan

        # ------ 6. 综合评分 ------
        # 基于多个指标的综合评分
        score_components = []
        
        # RankIC权重更高（更稳健）
        if np.isfinite(rank_icir):
            score_components.append(abs(rank_icir) * 0.4)
        if np.isfinite(icir):
            score_components.append(abs(icir) * 0.3)
        if np.isfinite(ret_diff):
            score_components.append(abs(ret_diff) * 100 * 0.2)  # 放大收益率差异
        if np.isfinite(wilson_low):
            score_components.append(max(0, wilson_low - 0.5) * 0.1)  # 超过50%的部分
            
        composite_score = sum(score_components) if score_components else 0

        #  7. 保存全部全部全部的因子统计 
        rows.append({
            'factor'            : fac,
            'cnt_total'         : valid.size,
            'cnt_dates'         : len(ic_by_day),
            'bin_edges'         : q.tolist(),
            'monotonicity'      : monotonicity,
            
            # 分组统计
            'win_Q0'            : grp_stats[0]['win'],
            'win_Q4'            : grp_stats[len(q)-2]['win'],
            'ret_Q0'            : grp_stats[0]['ret'],
            'ret_Q4'            : grp_stats[len(q)-2]['ret'],
            'win_diff'          : win_diff,
            'ret_diff'          : ret_diff,
            'p_ret'             : p_ret,
            'p_win'             : p_win,
            
            # IC指标
            'ic_mean'           : ic_mean,
            'ic_std'            : ic_std,
            'icir'              : icir,
            'p_ic'              : p_ic,
            'ic_positive_ratio' : ic_positive_ratio,
            
            # RankIC指标
            'rank_ic_mean'      : rank_ic_mean,
            'rank_ic_std'       : rank_ic_std,
            'rank_icir'         : rank_icir,
            'p_rank_ic'         : p_rank_ic,
            'rank_ic_positive_ratio': rank_ic_positive_ratio,
            
            # 稳定性
            'wilson_low'        : wilson_low,
            'wilson_high'       : wilson_high,
            
            # 综合评分
            'composite_score'   : composite_score
        })

        gc.collect()
        
    return pd.DataFrame(rows)

def load_train():
    if not os.path.exists(INPUT_DATA):
        raise FileNotFoundError(INPUT_DATA)
    df = pd.read_pickle(INPUT_DATA)
    df['T_date'] = pd.to_datetime(df['T_date'])
    # 保证测试集
    df = df[df['T_date'].dt.year.isin(TRAIN_YEARS)]
    return df

def main():
    start = datetime.now()
    mem('Start')
    df = load_train()
    print(f'训练期样本规模 {len(df):,}')

    exclude = {'T_date','T1_date','T2_date','stock','stockid','year',
               'buy_price','sell_price','pnl','ret','is_win','key'}
    factor_cols = [c for c in df.columns if c not in exclude]
    print(f'待评估因子数 {len(factor_cols):,}')

    t0 = time.time()
    res = calc_stats(df, factor_cols)
    t1 = time.time()
    
    if len(res) > 0:
        # 按综合评分排序
        res = res.sort_values('composite_score', ascending=False)
        res.to_csv(OUTPUT_STATS, index=False)
        print(f'统计完成，保存个数 {len(res):,}  → {OUTPUT_STATS}')

    else:
        print("闹麻了全没过，去放宽条件")
        res.to_csv(OUTPUT_STATS, index=False)
    
    summary = {
        'total_factors'   : len(factor_cols),
        'evaluated'       : len(res),
        'elapsed'         : str(datetime.now() - start),
        'cost_seconds'    : t1 - t0,
        'per_factor_sec'  : (t1-t0)/max(1,len(res)),
        'avg_composite_score': float(res['composite_score'].mean()) if len(res) > 0 else 0
    }
    
    with open(OUTPUT_SUM, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n摘要:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    mem('End')

if __name__ == '__main__':
    main()