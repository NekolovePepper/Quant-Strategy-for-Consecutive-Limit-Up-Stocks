"""
part2: 因子统计 v3 - 纯统计版本
任务：
1. 计算所有因子的统计指标，不做任何筛选
2. 包含IC、RankIC、ICIR、RankICIR等常用指标
3. 分组收益率分析和单调性检测
4. 输出完整的统计结果供后续筛选使用
"""
import os, gc, json, warnings, time
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from scipy.stats import spearmanr, pearsonr, ttest_ind, ttest_1samp, norm, linregress
from scipy.stats import kendalltau
import psutil

warnings.filterwarnings('ignore')

# ======= 参数 ======
TRAIN_YEARS = [2020, 2021, 2022]
INPUT_DATA = 'touch_signals_with_factors_and_returns.pkl'
OUTPUT_STATS = 'factor_statistics_all_2020_2023_time.csv'
OUTPUT_SUMMARY = 'factor_statistics_summary_time.json'

N_GROUP = 5  # 分组数量
N_GROUP_DECILE = 10  # 十分组用于更细致的分析

def mem(label=''):
    rss = psutil.Process().memory_info().rss / 1024 / 1024 / 1024
    print(f'[MEM] {label}  {rss:6.2f} GB')

def calc_ic_metrics(factor_vals, return_vals, method='pearson'):
    """计算IC相关指标"""
    if len(factor_vals) < 5:
        return np.nan
    
    # 去除NaN值
    mask = ~(np.isnan(factor_vals) | np.isnan(return_vals))
    if mask.sum() < 5:
        return np.nan
    
    factor_clean = factor_vals[mask]
    return_clean = return_vals[mask]
    
    if method == 'pearson':
        corr, _ = pearsonr(factor_clean, return_clean)
    elif method == 'spearman':
        corr, _ = spearmanr(factor_clean, return_clean)
    elif method == 'kendall':
        corr, _ = kendalltau(factor_clean, return_clean)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return corr if np.isfinite(corr) else np.nan

def detect_monotonicity_advanced(group_returns):
    """增强的单调性检测，返回类型和强度"""
    if len(group_returns) < 3:
        return 'none', 0.0
    
    returns = np.array(group_returns)
    n = len(returns)
    
    # 线性回归检测趋势
    x = np.arange(n)
    slope, _, r_value, p_value, _ = linregress(x, returns)
    r_squared = r_value ** 2
    
    # 严格单调性
    if np.all(np.diff(returns) > 0):
        return 'strict_ascending', 1.0
    elif np.all(np.diff(returns) < 0):
        return 'strict_descending', 1.0
    
    # 基于斜率和R²判断
    if p_value < 0.05:  # 统计显著
        if slope > 0 and r_squared > 0.7:
            return 'ascending', r_squared
        elif slope < 0 and r_squared > 0.7:
            return 'descending', r_squared
    
    # U型或倒U型检测
    if n >= 5:
        mid_idx = n // 2
        left_slope = (returns[mid_idx] - returns[0]) / mid_idx if mid_idx > 0 else 0
        right_slope = (returns[-1] - returns[mid_idx]) / (n - mid_idx - 1) if n - mid_idx - 1 > 0 else 0
        
        if left_slope < -0.001 and right_slope > 0.001:
            strength = min(abs(left_slope), abs(right_slope)) * 10  # 归一化强度
            return 'u_shape', min(strength, 1.0)
        elif left_slope > 0.001 and right_slope < -0.001:
            strength = min(abs(left_slope), abs(right_slope)) * 10
            return 'inverted_u', min(strength, 1.0)
    
    # 弱单调性
    diff_signs = np.sign(np.diff(returns))
    pos_ratio = np.sum(diff_signs > 0) / len(diff_signs)
    
    if pos_ratio >= 0.8:
        return 'weak_ascending', pos_ratio
    elif pos_ratio <= 0.2:
        return 'weak_descending', 1 - pos_ratio
    
    return 'none', 0.0

def calc_group_metrics(train_df, factor_name, n_groups=5):
    """计算分组相关指标"""
    series = train_df[factor_name].values
    rets = train_df['ret'].values
    wins = train_df['is_win'].values
    
    # 剔除NaN
    mask_valid = ~np.isnan(series)
    if mask_valid.sum() < n_groups * 10:  # 每组至少10个样本
        return None
    
    # 分组
    valid_series = series[mask_valid]
    quantiles = np.percentile(valid_series, np.linspace(0, 100, n_groups + 1))
    quantiles = np.unique(quantiles)
    
    if len(quantiles) <= 2:
        return None
    
    # 计算分组标签
    groups = np.searchsorted(quantiles[1:-1], series)
    groups[~mask_valid] = -1
    
    # 分组统计
    group_stats = []
    for g in range(n_groups):
        idx = (groups == g)
        if idx.sum() == 0:
            group_stats.append({
                'group': g,
                'count': 0,
                'ret_mean': np.nan,
                'ret_std': np.nan,
                'win_rate': np.nan,
                'sharpe': np.nan
            })
        else:
            g_rets = rets[idx]
            g_wins = wins[idx]
            ret_mean = np.nanmean(g_rets)
            ret_std = np.nanstd(g_rets)
            
            group_stats.append({
                'group': g,
                'count': idx.sum(),
                'ret_mean': ret_mean,
                'ret_std': ret_std,
                'win_rate': np.nanmean(g_wins),
                'sharpe': ret_mean / ret_std if ret_std > 0 else np.nan
            })
    
    # 计算多空收益
    top_idx = (groups == n_groups - 1)
    bottom_idx = (groups == 0)
    
    if top_idx.sum() > 0 and bottom_idx.sum() > 0:
        ls_ret = np.nanmean(rets[top_idx]) - np.nanmean(rets[bottom_idx])
        ls_win = np.nanmean(wins[top_idx]) - np.nanmean(wins[bottom_idx])
        
        # t检验
        t_stat_ret, p_ret = ttest_ind(rets[top_idx], rets[bottom_idx], 
                                     equal_var=False, nan_policy='omit')
        t_stat_win, p_win = ttest_ind(wins[top_idx], wins[bottom_idx], 
                                     equal_var=False, nan_policy='omit')
    else:
        ls_ret = ls_win = np.nan
        t_stat_ret = p_ret = t_stat_win = p_win = np.nan
    
    # 单调性检测
    group_returns = [s['ret_mean'] for s in group_stats if not np.isnan(s['ret_mean'])]
    mono_type, mono_strength = detect_monotonicity_advanced(group_returns)
    
    return {
        'group_stats': group_stats,
        'long_short_ret': ls_ret,
        'long_short_win': ls_win,
        't_stat_ret': t_stat_ret,
        'p_value_ret': p_ret,
        't_stat_win': t_stat_win,
        'p_value_win': p_win,
        'monotonicity': mono_type,
        'mono_strength': mono_strength
    }

def calc_stability_metrics(ic_series, rankic_series):
    """计算稳定性相关指标"""
    metrics = {}
    
    # 转换为numpy数组
    ic_array = np.array(ic_series) if len(ic_series) > 0 else np.array([])
    rankic_array = np.array(rankic_series) if len(rankic_series) > 0 else np.array([])
    
    # IC稳定性
    if len(ic_array) > 0:
        metrics['ic_positive_ratio'] = np.mean(ic_array > 0)
        metrics['ic_negative_ratio'] = np.mean(ic_array < 0)
        metrics['ic_abs_mean'] = np.mean(np.abs(ic_array))
        metrics['ic_volatility'] = np.std(ic_array)
        
        # 月度IC稳定性（假设20个交易日为一个月）
        if len(ic_array) >= 20:
            monthly_ics = [np.mean(ic_array[i:i+20]) for i in range(0, len(ic_array)-19, 20)]
            metrics['monthly_ic_std'] = np.std(monthly_ics) if len(monthly_ics) > 1 else np.nan
        else:
            metrics['monthly_ic_std'] = np.nan
    else:
        metrics['ic_positive_ratio'] = np.nan
        metrics['ic_negative_ratio'] = np.nan
        metrics['ic_abs_mean'] = np.nan
        metrics['ic_volatility'] = np.nan
        metrics['monthly_ic_std'] = np.nan
    
    # RankIC稳定性
    if len(rankic_array) > 0:
        metrics['rankic_positive_ratio'] = np.mean(rankic_array > 0)
        metrics['rankic_negative_ratio'] = np.mean(rankic_array < 0)
        metrics['rankic_abs_mean'] = np.mean(np.abs(rankic_array))
        metrics['rankic_volatility'] = np.std(rankic_array)
    else:
        metrics['rankic_positive_ratio'] = np.nan
        metrics['rankic_negative_ratio'] = np.nan
        metrics['rankic_abs_mean'] = np.nan
        metrics['rankic_volatility'] = np.nan
    
    return metrics

def calc_all_statistics(train_df, factor_cols):
    """计算所有因子的完整统计信息"""
    rows = []
    date_list = sorted(train_df['T_date'].unique())
    
    for fac in tqdm(factor_cols, desc='统计因子'):
        if fac not in train_df.columns:
            continue
            
        series = train_df[fac].values
        valid_count = (~np.isnan(series)).sum()
        
        # 即使样本很少也统计，只是标记数据质量
        row = {
            'factor': fac,
            'valid_count': valid_count,
            'valid_ratio': valid_count / len(series),
            'data_quality': 'good' if valid_count > 1000 else 'poor' if valid_count > 100 else 'very_poor'
        }
        
        # 1. IC和RankIC计算
        ic_by_day = []
        rankic_by_day = []
        
        for d in date_list:
            day_mask = (train_df['T_date'] == d)
            day_factor = series[day_mask]
            day_ret = train_df['ret'].values[day_mask]
            
            if len(day_factor) < 5 or np.isnan(day_factor).all():
                continue
            
            # IC (Pearson)
            ic = calc_ic_metrics(day_factor, day_ret, 'pearson')
            if not np.isnan(ic):
                ic_by_day.append(ic)
            
            # RankIC (Spearman)
            rankic = calc_ic_metrics(day_factor, day_ret, 'spearman')
            if not np.isnan(rankic):
                rankic_by_day.append(rankic)
        
        # IC统计
        if len(ic_by_day) > 0:
            ic_array = np.array(ic_by_day)
            row['ic_mean'] = ic_array.mean()
            row['ic_std'] = ic_array.std()
            row['ic_ir'] = row['ic_mean'] / row['ic_std'] if row['ic_std'] > 0 else 0
            row['ic_t_stat'], row['ic_p_value'] = ttest_1samp(ic_array, 0)
            row['ic_count'] = len(ic_by_day)
        else:
            row.update({k: np.nan for k in ['ic_mean', 'ic_std', 'ic_ir', 'ic_t_stat', 'ic_p_value', 'ic_count']})
        
        # RankIC统计
        if len(rankic_by_day) > 0:
            rankic_array = np.array(rankic_by_day)
            row['rankic_mean'] = rankic_array.mean()
            row['rankic_std'] = rankic_array.std()
            row['rankic_ir'] = row['rankic_mean'] / row['rankic_std'] if row['rankic_std'] > 0 else 0
            row['rankic_t_stat'], row['rankic_p_value'] = ttest_1samp(rankic_array, 0)
            row['rankic_count'] = len(rankic_by_day)
        else:
            row.update({k: np.nan for k in ['rankic_mean', 'rankic_std', 'rankic_ir', 'rankic_t_stat', 'rankic_p_value', 'rankic_count']})
        
        # 2. 稳定性指标
        stability = calc_stability_metrics(ic_by_day, rankic_by_day)
        row.update(stability)
        
        # 3. 分组分析（5分组）
        group_result = calc_group_metrics(train_df, fac, N_GROUP)
        if group_result:
            row['long_short_ret'] = group_result['long_short_ret']
            row['long_short_win'] = group_result['long_short_win']
            row['ls_t_stat'] = group_result['t_stat_ret']
            row['ls_p_value'] = group_result['p_value_ret']
            row['monotonicity'] = group_result['monotonicity']
            row['mono_strength'] = group_result['mono_strength']
            
            # 各组收益率和胜率
            for i, gs in enumerate(group_result['group_stats']):
                row[f'g{i}_ret'] = gs['ret_mean']
                row[f'g{i}_win'] = gs['win_rate']
                row[f'g{i}_count'] = gs['count']
        
        # 4. 十分组分析（用于更细致的单调性）
        decile_result = calc_group_metrics(train_df, fac, N_GROUP_DECILE)
        if decile_result:
            row['decile_mono'] = decile_result['monotonicity']
            row['decile_mono_strength'] = decile_result['mono_strength']
            # Top和Bottom十分位
            row['top_decile_ret'] = decile_result['group_stats'][-1]['ret_mean']
            row['bottom_decile_ret'] = decile_result['group_stats'][0]['ret_mean']
            row['top_decile_win'] = decile_result['group_stats'][-1]['win_rate']
            row['bottom_decile_win'] = decile_result['group_stats'][0]['win_rate']
        
        # 5. 整体统计
        factor_rets = train_df[~np.isnan(series)]['ret'].values
        if len(factor_rets) > 0:
            row['overall_mean_ret'] = np.mean(factor_rets)
            row['overall_std_ret'] = np.std(factor_rets)
            row['overall_sharpe'] = row['overall_mean_ret'] / row['overall_std_ret'] if row['overall_std_ret'] > 0 else 0
        
        rows.append(row)
        
        # 定期清理内存
        if len(rows) % 100 == 0:
            gc.collect()
    
    return pd.DataFrame(rows)

def main():
    start = datetime.now()
    mem('Start')
    
    # 加载数据
    if not os.path.exists(INPUT_DATA):
        raise FileNotFoundError(INPUT_DATA)
    
    df = pd.read_pickle(INPUT_DATA)
    df['T_date'] = pd.to_datetime(df['T_date'])
    df = df[df['T_date'].dt.year.isin(TRAIN_YEARS)]
    
    print(f'训练期样本数: {len(df):,}')
    
    # 识别因子列
    exclude_cols = {'T_date', 'T1_date', 'T2_date', 'stock', 'stockid', 'year',
                   'buy_price', 'sell_price', 'pnl', 'ret', 'is_win', 'key'}
    factor_cols = [c for c in df.columns if c not in exclude_cols]
    print(f'待统计因子数: {len(factor_cols):,}')
    
    # 计算统计
    t0 = time.time()
    stats_df = calc_all_statistics(df, factor_cols)
    t1 = time.time()
    
    # 保存结果
    stats_df.to_csv(OUTPUT_STATS, index=False)
    print(f'统计完成，保存 {len(stats_df):,} 条记录 → {OUTPUT_STATS}')
    
    # 生成摘要
    summary = {
        'total_factors': len(factor_cols),
        'factors_analyzed': len(stats_df),
        'good_quality_factors': len(stats_df[stats_df['data_quality'] == 'good']),
        'poor_quality_factors': len(stats_df[stats_df['data_quality'].isin(['poor', 'very_poor'])]),
        'elapsed_time': str(datetime.now() - start),
        'compute_seconds': t1 - t0,
        'avg_ic_mean': float(stats_df['ic_mean'].mean()) if 'ic_mean' in stats_df else 0,
        'avg_rankic_mean': float(stats_df['rankic_mean'].mean()) if 'rankic_mean' in stats_df else 0,
        'factors_with_monotonicity': len(stats_df[~stats_df['monotonicity'].isin(['none', np.nan])]) if 'monotonicity' in stats_df else 0
    }
    
    with open(OUTPUT_SUMMARY, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n统计摘要:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    
    mem('End')

if __name__ == '__main__':
    main()