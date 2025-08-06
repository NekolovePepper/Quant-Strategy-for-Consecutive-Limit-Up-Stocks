"""
part 3 v4: 因子筛选 - 基于统计结果的综合筛选（简化版）
任务：
1. 读取统计结果，进行综合评分
2. 因子方向确定和统一
3. 高相关性因子去重（相关系数>0.98视为相同）
4. 基于可配置权重的综合筛选
5. 支持ST股票剔除等预处理
"""
import os, json, warnings, time
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
warnings.filterwarnings('ignore')

# ======= 参数配置 =======
# ST股票处理
EXCLUDE_ST = True
ST_STATUS_FILE = '/mnt/tonglian_data2/support_data/st_status.fea'

# 输入输出文件
STATS_FILE = 'factor_statistics_all_2020_2023_time.csv'
TRAIN_DATA_FILE = 'touch_signals_with_factors_and_returns.pkl'
OUTPUT_DIR = './facfilter3_output_time'
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_DIRECTION = f'{OUTPUT_DIR}/factor_directions_2020_2023{"_nst" if EXCLUDE_ST else ""}.csv'
OUTPUT_DUPLICATES = f'{OUTPUT_DIR}/factor_duplicates_2020_2023{"_nst" if EXCLUDE_ST else ""}.csv'
OUTPUT_SCORES = f'{OUTPUT_DIR}/factor_scores_2020_2023{"_nst" if EXCLUDE_ST else ""}.csv'
OUTPUT_FINAL = f'{OUTPUT_DIR}/selected_factors_2020_2023{"_nst" if EXCLUDE_ST else ""}.csv'
OUTPUT_SUMMARY = f'{OUTPUT_DIR}/filter_summary{"_nst" if EXCLUDE_ST else ""}.json'

# 筛选参数
# 数据质量阈值
MIN_IC_COUNT = 10  # 最少IC计算天数
MIN_VALID_RATIO = 0.01  # 最低有效数据比例

# 去重阈值
DUPLICATE_CORRELATION_THRESHOLD = 0.98  # 相关系数超过此值视为重复因子

# 综合评分权重（可以将不需要的指标权重设为0）
SCORE_WEIGHTS = {
    # IC相关权重
    'ic_ir': 0.2,  # IC信息比率
    'rankic_ir': 0.25,  # RankIC信息比率（更稳健）
    'ic_abs_mean': 0.00,  # IC绝对值均值
    'rankic_abs_mean': 0.05,  # RankIC绝对值均值
    
    # 收益相关权重
    'long_short_ret': 0.15,  # 多空收益
    'long_short_win': 0.1,  # 多空胜率差
    
    # 单调性权重
    'monotonicity_score': 0.15,  # 单调性得分
    
    # 稳定性权重
    'stability_score': 0.1,  # 稳定性得分
    
    # 统计显著性（设为0表示不考虑）
    'significance_score': 0.0,
}

# 各类筛选阈值（设为None表示不限制）
FILTER_THRESHOLDS = {
    'min_rankic_ir': None,  # 最小RankIC IR
    'min_rankic_abs_mean': 0.01,  # 最小多空收益
    'min_mono_strength': None,  # 最小单调性强度
    'min_comprehensive_score': None,  # 最小综合得分
}

# 最终选择参数
MAX_FACTORS = 100  # 最终选择的最大因子数

# ======== 工具函数 ========
def to_py(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_py(v) for v in obj]
    return obj

def load_st_status():
    """加载ST状态数据"""
    if not os.path.exists(ST_STATUS_FILE):
        print(f'ST文件不存在: {ST_STATUS_FILE}')
        return None
    
    st_df = pd.read_feather(ST_STATUS_FILE)
    if 'TRADE_DATE' in st_df.columns:
        st_df.rename(columns={'TRADE_DATE': 'T_date'}, inplace=True)
    
    # 转换为长表格式
    st_long = st_df.melt(id_vars='T_date', var_name='stock', value_name='is_st')
    st_long['T_date'] = pd.to_datetime(st_long['T_date'])
    st_long = st_long[st_long['is_st'] == 1][['T_date', 'stock']]
    
    return st_long

def filter_st_factors(train_df, st_df, factor_stats):
    """剔除ST股票影响后重新评估因子质量"""
    if st_df is None or train_df is None:
        return factor_stats
    
    # 标记ST股票信号
    train_df = train_df.merge(st_df, on=['T_date', 'stock'], how='left', indicator=True)
    train_df['is_st'] = (train_df['_merge'] == 'both')
    train_df = train_df.drop('_merge', axis=1)
    
    # 计算剔除ST后的有效样本比例
    st_impact = []
    for _, row in factor_stats.iterrows():
        factor = row['factor']
        if factor not in train_df.columns:
            st_impact.append(1.0)
            continue
        
        # 计算ST影响
        non_st_mask = ~train_df['is_st']
        factor_valid = ~train_df[factor].isna()
        
        total_valid = factor_valid.sum()
        non_st_valid = (factor_valid & non_st_mask).sum()
        
        impact_ratio = non_st_valid / total_valid if total_valid > 0 else 0
        st_impact.append(impact_ratio)
    
    factor_stats['non_st_ratio'] = st_impact
    return factor_stats

# ======== 方向确定 ========
def determine_factor_direction(row):
    """基于多个指标投票确定因子最优方向"""
    votes = {'buy': 0, 'sell': 0}
    weights = {'buy': 0, 'sell': 0}
    
    # RankIC投票（权重最高）
    if pd.notna(row.get('rankic_mean')):
        direction = 'buy' if row['rankic_mean'] > 0 else 'sell'
        votes[direction] += 1
        weights[direction] += 3
    
    # IC投票
    if pd.notna(row.get('ic_mean')):
        direction = 'buy' if row['ic_mean'] > 0 else 'sell'
        votes[direction] += 1
        weights[direction] += 2
    
    # 多空收益投票
    if pd.notna(row.get('long_short_ret')):
        direction = 'buy' if row['long_short_ret'] > 0 else 'sell'
        votes[direction] += 1
        weights[direction] += 2
    
    # 多空胜率投票
    if pd.notna(row.get('long_short_win')):
        direction = 'buy' if row['long_short_win'] > 0 else 'sell'
        votes[direction] += 1
        weights[direction] += 1
    
    # 单调性投票
    mono = row.get('monotonicity', 'none')
    if mono in ['strict_ascending', 'ascending', 'weak_ascending']:
        votes['buy'] += 1
        weights['buy'] += 2
    elif mono in ['strict_descending', 'descending', 'weak_descending']:
        votes['sell'] += 1
        weights['sell'] += 2
    
    # 基于加权投票决定方向
    if weights['buy'] > weights['sell']:
        return 'buy'
    elif weights['sell'] > weights['buy']:
        return 'sell'
    else:
        # 平局时看票数
        return 'buy' if votes['buy'] >= votes['sell'] else 'sell'

def unify_factor_direction(stats_df):
    """统一因子方向，使所有指标都指向同一方向"""
    df = stats_df.copy()
    
    # 确定每个因子的方向
    directions = []
    for _, row in df.iterrows():
        direction = determine_factor_direction(row)
        directions.append(direction)
    
    df['direction'] = directions
    
    # 需要翻转符号的列
    sign_cols = ['ic_mean', 'ic_ir', 'rankic_mean', 'rankic_ir',
                 'long_short_ret', 'long_short_win', 'ls_t_stat',
                 'top_decile_ret', 'bottom_decile_ret',
                 'top_decile_win', 'bottom_decile_win']
    
    # 需要翻转比例的列
    ratio_cols = ['ic_positive_ratio', 'rankic_positive_ratio']
    
    # 需要交换的列对
    swap_pairs = [
        ('top_decile_ret', 'bottom_decile_ret'),
        ('top_decile_win', 'bottom_decile_win'),
        ('g4_ret', 'g0_ret'),
        ('g4_win', 'g0_win'),
    ]
    
    # 单调性映射
    mono_map = {
        'strict_ascending': 'strict_descending',
        'ascending': 'descending',
        'weak_ascending': 'weak_descending',
        'strict_descending': 'strict_ascending',
        'descending': 'ascending',
        'weak_descending': 'weak_ascending',
    }
    
    # 应用方向转换
    for idx, row in df.iterrows():
        if row['direction'] == 'sell':
            # 翻转符号
            for col in sign_cols:
                if col in df.columns and pd.notna(df.at[idx, col]):
                    df.at[idx, col] = -df.at[idx, col]
            
            # 翻转比例
            for col in ratio_cols:
                if col in df.columns and pd.notna(df.at[idx, col]):
                    df.at[idx, col] = 1 - df.at[idx, col]
            
            # 交换列
            for col1, col2 in swap_pairs:
                if col1 in df.columns and col2 in df.columns:
                    df.at[idx, col1], df.at[idx, col2] = df.at[idx, col2], df.at[idx, col1]
            
            # 翻转单调性
            for mono_col in ['monotonicity', 'decile_mono']:
                if mono_col in df.columns:
                    mono = df.at[idx, mono_col]
                    if mono in mono_map:
                        df.at[idx, mono_col] = mono_map[mono]
    
    return df

# ======== 相关性去重 ========
def find_duplicate_factors(train_df, factors, threshold=0.98):
    """找出高度相关的重复因子"""
    valid_factors = []
    factor_data = {}
    
    # 准备数据
    for fac in factors:
        if fac not in train_df.columns:
            continue
        
        data = train_df[fac].values
        if np.isnan(data).all() or np.nanstd(data) < 1e-8:
            continue
        
        # 标准化
        data_mean = np.nanmean(data)
        data_std = np.nanstd(data) + 1e-8
        data_norm = (data - data_mean) / data_std
        
        factor_data[fac] = data_norm
        valid_factors.append(fac)
    
    n = len(valid_factors)
    if n <= 1:
        return {}
    
    # 计算相关性并找出重复
    duplicates = {}  # {保留的因子: [被去重的因子列表]}
    removed = set()
    
    for i in range(n):
        if valid_factors[i] in removed:
            continue
            
        duplicates[valid_factors[i]] = []
        
        for j in range(i + 1, n):
            if valid_factors[j] in removed:
                continue
            
            # 计算相关性
            data_i = factor_data[valid_factors[i]]
            data_j = factor_data[valid_factors[j]]
            
            # 找出两个因子都有效的数据点
            valid_mask = ~(np.isnan(data_i) | np.isnan(data_j))
            
            if valid_mask.sum() < 100:
                continue
            
            corr = np.corrcoef(data_i[valid_mask], data_j[valid_mask])[0, 1]
            
            # 如果相关性超过阈值，标记为重复
            if abs(corr) >= threshold:
                duplicates[valid_factors[i]].append(valid_factors[j])
                removed.add(valid_factors[j])
    
    # 清理空列表
    duplicates = {k: v for k, v in duplicates.items() if v}
    
    return duplicates

# ======== 综合评分 ========
def calculate_monotonicity_score(row):
    """计算单调性得分"""
    mono_scores = {
        'strict_ascending': 1.0,
        'strict_descending': 1.0,
        'ascending': 0.8,
        'descending': 0.8,
        'weak_ascending': 0.5,
        'weak_descending': 0.5,
        'u_shape': 0.3,
        'inverted_u': 0.3,
        'none': 0.0
    }
    
    # 综合5分组和10分组的单调性
    mono5 = row.get('monotonicity', 'none')
    mono10 = row.get('decile_mono', 'none')
    strength5 = row.get('mono_strength', 0)
    strength10 = row.get('decile_mono_strength', 0)
    
    score5 = mono_scores.get(mono5, 0) * strength5
    score10 = mono_scores.get(mono10, 0) * strength10
    
    # 加权平均，10分组权重略高
    return 0.4 * score5 + 0.6 * score10

def calculate_stability_score(row):
    """计算稳定性得分"""
    scores = []
    
    # IC正比例
    if pd.notna(row.get('ic_positive_ratio')):
        # 转换到[0,1]，0.5映射到0，1映射到1
        scores.append(max(0, (row['ic_positive_ratio'] - 0.5) * 2))
    
    # RankIC正比例
    if pd.notna(row.get('rankic_positive_ratio')):
        scores.append(max(0, (row['rankic_positive_ratio'] - 0.5) * 2))
    
    # IC波动率（越小越好）
    if pd.notna(row.get('ic_volatility')) and pd.notna(row.get('ic_abs_mean')):
        if row['ic_abs_mean'] > 0:
            # 计算变异系数，越小越稳定
            cv = row['ic_volatility'] / row['ic_abs_mean']
            scores.append(max(0, 1 - cv))
    
    return np.mean(scores) if scores else 0

def calculate_significance_score(row):
    """计算统计显著性得分"""
    scores = []
    
    # p值转换为得分（越小越好）
    p_value_cols = ['ic_p_value', 'rankic_p_value', 'ls_p_value']
    for col in p_value_cols:
        if pd.notna(row.get(col)):
            p_val = row[col]
            if p_val < 0.01:
                scores.append(1.0)
            elif p_val < 0.05:
                scores.append(0.8)
            elif p_val < 0.1:
                scores.append(0.5)
            else:
                scores.append(0.0)
    
    return np.mean(scores) if scores else 0

def calculate_comprehensive_score(stats_df, weights):
    """计算综合得分"""
    df = stats_df.copy()
    
    # 准备需要标准化的指标
    score_components = {}
    
    # 直接使用的指标
    direct_metrics = ['ic_ir', 'rankic_ir', 'ic_abs_mean', 'rankic_abs_mean',
                     'long_short_ret', 'long_short_win']
    
    for metric in direct_metrics:
        if metric in df.columns and weights.get(metric, 0) > 0:
            score_components[metric] = df[metric].fillna(0).values
    
    # 计算衍生指标
    if weights.get('monotonicity_score', 0) > 0:
        score_components['monotonicity_score'] = df.apply(calculate_monotonicity_score, axis=1).values
    
    if weights.get('stability_score', 0) > 0:
        score_components['stability_score'] = df.apply(calculate_stability_score, axis=1).values
    
    if weights.get('significance_score', 0) > 0:
        score_components['significance_score'] = df.apply(calculate_significance_score, axis=1).values
    
    # 标准化所有指标（使用RobustScaler对异常值更稳健）
    scaler = RobustScaler()
    normalized_scores = {}
    
    for metric, values in score_components.items():
        if np.nanstd(values) > 1e-8:
            values_2d = values.reshape(-1, 1)
            normalized = scaler.fit_transform(values_2d).flatten()
            # 转换到[0,1]区间
            normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-8)
        else:
            normalized = np.zeros_like(values)
        normalized_scores[metric] = normalized
    
    # 计算加权得分
    total_score = np.zeros(len(df))
    total_weight = 0
    
    for metric, weight in weights.items():
        if metric in normalized_scores and weight > 0:
            total_score += normalized_scores[metric] * weight
            total_weight += weight
            # 保存各分项得分
            df[f'{metric}_norm'] = normalized_scores[metric]
    
    # 归一化到[0,1]
    if total_weight > 0:
        total_score /= total_weight
    
    df['comprehensive_score'] = total_score
    
    return df

# ======== 筛选函数 ========
def apply_filters(scored_df, thresholds):
    """应用筛选条件"""
    df = scored_df.copy()
    
    # 应用各项阈值
    for metric, threshold in thresholds.items():
        if threshold is not None:
            col_name = metric.replace('min_', '')
            if col_name in df.columns:
                df = df[df[col_name] >= threshold]
    
    return df

# ======== 主函数 ========
def main():
    start_time = time.time()
    
    # 1. 加载统计结果
    if not os.path.exists(STATS_FILE):
        raise FileNotFoundError(f"统计文件不存在: {STATS_FILE}")
    
    stats_df = pd.read_csv(STATS_FILE)
    print(f"加载 {len(stats_df)} 个因子的统计结果")
    
    # 2. 数据质量筛选
    if MIN_IC_COUNT > 0 or MIN_VALID_RATIO > 0:
        quality_mask = (stats_df['ic_count'] >= MIN_IC_COUNT) & \
                       (stats_df['valid_ratio'] >= MIN_VALID_RATIO)
        stats_df = stats_df[quality_mask]
        print(f"数据质量筛选后剩余: {len(stats_df)}")
    else:
        print(f"跳过数据质量筛选，保留所有 {len(stats_df)} 个因子")
    
    # 3. ST股票处理
    train_df = None
    if EXCLUDE_ST:
        st_df = load_st_status()
        if st_df is not None and os.path.exists(TRAIN_DATA_FILE):
            train_df = pd.read_pickle(TRAIN_DATA_FILE)
            train_df['T_date'] = pd.to_datetime(train_df['T_date'])
            train_df = train_df[train_df['T_date'].dt.year.isin([2020, 2021, 2022, 2023])]
            stats_df = filter_st_factors(train_df, st_df, stats_df)
            print(f"ST影响评估完成")
    
    # 4. 确定因子方向
    stats_df = unify_factor_direction(stats_df)
    direction_df = stats_df[['factor', 'direction']].copy()
    direction_df.to_csv(OUTPUT_DIRECTION, index=False)
    print(f"因子方向确定完成: buy={len(stats_df[stats_df['direction']=='buy'])}, "
          f"sell={len(stats_df[stats_df['direction']=='sell'])}")
    
    # 5. 去重处理
    if train_df is None and os.path.exists(TRAIN_DATA_FILE):
        train_df = pd.read_pickle(TRAIN_DATA_FILE)
        train_df['T_date'] = pd.to_datetime(train_df['T_date'])
        train_df = train_df[train_df['T_date'].dt.year.isin([2020, 2021, 2022, 2023])]
    
    duplicates = {}
    removed_factors = set()
    
    if train_df is not None:
        duplicates = find_duplicate_factors(train_df, stats_df['factor'].tolist(), 
                                          DUPLICATE_CORRELATION_THRESHOLD)
        
        # 保存重复因子记录
        dup_records = []
        for kept, removed_list in duplicates.items():
            for removed in removed_list:
                dup_records.append({
                    'kept_factor': kept,
                    'removed_factor': removed,
                    'correlation': DUPLICATE_CORRELATION_THRESHOLD
                })
                removed_factors.add(removed)
        
        if dup_records:
            dup_df = pd.DataFrame(dup_records)
            dup_df.to_csv(OUTPUT_DUPLICATES, index=False)
        
        print(f"去重处理: 发现 {len(duplicates)} 组重复因子，移除 {len(removed_factors)} 个")
    
    # 移除重复因子
    stats_df = stats_df[~stats_df['factor'].isin(removed_factors)]
    print(f"去重后剩余: {len(stats_df)}")
    
    # 6. 综合评分
    scored_df = calculate_comprehensive_score(stats_df, SCORE_WEIGHTS)
    scored_df.to_csv(OUTPUT_SCORES, index=False)
    print(f"综合评分计算完成")
    
    # 7. 筛选因子
    # 应用筛选阈值
    filtered_df = apply_filters(scored_df, FILTER_THRESHOLDS)
    print(f"阈值筛选后: {len(filtered_df)}")
    
    # 按综合得分排序并限制数量
    final_df = filtered_df.nlargest(min(MAX_FACTORS, len(filtered_df)), 'comprehensive_score')
    
    # 保存最终结果
    output_cols = ['factor', 'direction', 'comprehensive_score',
                   'rankic_ir', 'ic_ir', 'long_short_ret', 'monotonicity',
                   'mono_strength', 'rankic_positive_ratio']
    
    # 确保所有列都存在
    output_cols = [col for col in output_cols if col in final_df.columns]
    
    final_df[output_cols].to_csv(OUTPUT_FINAL, index=False)
    print(f"最终选择 {len(final_df)} 个因子 → {OUTPUT_FINAL}")
    
    # 8. 生成摘要
    summary = {
            'initial_factors': int(len(pd.read_csv(STATS_FILE))),
            'after_quality_filter': int(len(stats_df)) if MIN_IC_COUNT == 0 and MIN_VALID_RATIO == 0 else int(quality_mask.sum()),
            'after_duplicate_removal': int(len(stats_df)),
            'after_threshold_filter': int(len(filtered_df)),
            'final_selected': int(len(final_df)),
            'duplicates_found': int(len(duplicates)),
            'factors_removed_as_duplicates': int(len(removed_factors)),
            'direction_distribution': {str(k): int(v) for k, v in final_df['direction'].value_counts().to_dict().items()},
            'avg_score': float(final_df['comprehensive_score'].mean()),
            'top10_factors': [str(x) for x in final_df.nlargest(10, 'comprehensive_score')['factor'].tolist()],
            'parameters': {
                'exclude_st': bool(EXCLUDE_ST),
                'duplicate_threshold': float(DUPLICATE_CORRELATION_THRESHOLD),
                'score_weights': {k: float(v) for k, v in SCORE_WEIGHTS.items()},
                'filter_thresholds': {k: (float(v) if v is not None else None) for k, v in FILTER_THRESHOLDS.items()},
                'max_factors': int(MAX_FACTORS)
            },
            'elapsed_seconds': float(time.time() - start_time)
        }
    
    with open(OUTPUT_SUMMARY, 'w') as f:
        json.dump(to_py(summary), f, indent=2, ensure_ascii=False)
    
    print(f"\n筛选摘要:")
    print(json.dumps(to_py(summary), indent=2, ensure_ascii=False))
    

if __name__ == '__main__':
    main()