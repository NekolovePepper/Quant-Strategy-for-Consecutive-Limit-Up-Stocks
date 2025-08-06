
"""
part1：数据准备
任务：
1. 生成触板信号
2. 读取、合并全部因子数据
3. 计算TWAP买卖价格
4. 计算收益率指标
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import concurrent.futures
import gc
import warnings
import psutil
from datetime import datetime
import numba
from numba import jit
warnings.filterwarnings('ignore')

# ========== 配置参数 ==========
FORCE_REBUILD = False  # 设为True全部重新生成，False则直接读已存在的文件
MEMORY_EFFICIENT = False  # 及时处理，不边写边运行了
MAX_WORKERS = min(32, os.cpu_count())

# 时间参数
train_years = [2020, 2021, 2022, 2023]
test_years = [2024]
full_date_range = (f'{train_years[0]}-01-01', f'{test_years[-1]}-12-31')

# 路径参数
ohlc_path = '/mnt/tonglian_data2/ohlc_fea/'
factor_paths = [
    '/mnt/factor/min2day2/',
    '/mnt/factor/min2day/',
    '/mnt/factor/min_fac2/',
    '/mnt/factor/min_fac/',
]
turnover_path = os.path.join(ohlc_path, 'TURNOVER_RATE.fea')
min_data_path = '/mnt/tonglian_data2/min_data/'
twap_df_path = '/home/user5/project1/twap_after_morning.pkl'
buy_time1, buy_time2 = 931, 940

# 输出文件名
OUTPUT_SIGNALS_FILE = 'touch_signals_with_factors_and_returns.pkl'
OUTPUT_SUMMARY_FILE = 'data_preparation_summary.json'

# ========== 工具函数 ==========

# 运行太久，监测内存使用、处理下数值数据

def print_memory_usage(label=""):
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"[内存] {label} RSS: {memory_info.rss / 1024 / 1024 / 1024:.2f} GB")

def optimize_dtypes(df):
    for col in df.columns:
        # 跳过已知的非数值列
        if col in ['T_date', 'T1_date', 'T2_date', 'stock', 'TRADE_DATE']:
            continue
        if np.issubdtype(df[col].dtype, np.datetime64):
            continue

        col_type = df[col].dtype
        if col_type != 'object':
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type).startswith('int'):
                if c_min > np.iinfo(np.int8 ).min and c_max < np.iinfo(np.int8 ).max:
                    df[col] = df[col].astype(np.int8 )
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.float32)
    return df

# 对分钟数据代码统一字段
def get_stockid(stock):
    return 'SH' + stock if stock.startswith('6') else 'SZ' + stock

# ========== 1 生成触板信号 ==========

def generate_touch_signals():

    print("1 生成触板信号")

    signals_file = 'touch_signals_basic.pkl'
    if not FORCE_REBUILD and os.path.exists(signals_file):
        print(f"直接加载触板信号文件: {signals_file}")
        signals = pd.read_pickle(signals_file)
        print(f"触板信号数量: {len(signals):,}")
        return signals
    
    print("生成触板信号ing")
    print_memory_usage("开始")
    
    # 读取价格数据
    highest_df = pd.read_feather(os.path.join(ohlc_path, 'HIGHEST_PRICE.fea'))
    limitup_df = pd.read_feather(os.path.join(ohlc_path, 'LIMIT_UP_PRICE.fea'))
    
    # 时间筛选
    for df in [highest_df, limitup_df]:
        df['TRADE_DATE'] = pd.to_datetime(df['TRADE_DATE'])
        df.query(f"'{full_date_range[0]}' <= TRADE_DATE <= '{full_date_range[1]}'", inplace=True)
    
    # 对齐数据
    highest_df = highest_df.set_index('TRADE_DATE')
    limitup_df = limitup_df.set_index('TRADE_DATE')
    
    if MEMORY_EFFICIENT:
        highest_df = highest_df.astype(np.float32)
        limitup_df = limitup_df.astype(np.float32)
    
    # aligned 对齐
    high_aligned, limit_aligned = highest_df.align(limitup_df, join='inner', axis=1)
    high_aligned, limit_aligned = high_aligned.align(limit_aligned, join='inner', axis=0)
    trade_dates = high_aligned.index
    
    print_memory_usage("数据对齐好了")
    
    # 计算触板信号
    arr_high = high_aligned.values
    arr_limit = limit_aligned.values
    touch_mat = np.isclose(arr_high, arr_limit, equal_nan=False)
    touch_flag = touch_mat & ~np.roll(touch_mat, 1, axis=0)
    touch_flag[0, :] = touch_mat[0, :]
    touch_flag = touch_flag[:-2]  # 确保T+2存在
    
    # 提取信号
    rows, cols = np.where(touch_flag)
    stocks = high_aligned.columns[cols]
    
    signals = pd.DataFrame({
        'T_date': trade_dates[rows],
        'stock': stocks.astype(str),
        'T1_date': trade_dates[rows + 1],
        'T2_date': trade_dates[rows + 2]
    })
    
    signals['year'] = pd.to_datetime(signals['T_date']).dt.year
    
    # 保存基础信号
    signals.to_pickle(signals_file)
    print(f"基础信号已保存: {signals_file}")
    
    # 清内存
    del highest_df, limitup_df, high_aligned, limit_aligned, touch_mat, touch_flag
    gc.collect()
    
    print(f"生成触板信号数量: {len(signals):,}")
    print_memory_usage("触板信号生成完成")
    return signals

# ========== 2 合并因子数据 ==========
def read_single_factor_file_optimized(args):
    # 单个因子文件读取
    fpath, need_dates, target_columns = args
    try:
        fname = os.path.basename(fpath)
        # 日期长度
        if len(fname) < 8:
            return None
            
        dt_str = fname[:8]
        try:
            dt = pd.to_datetime(dt_str)
        except:
            return None
            
        if dt not in need_dates:
            return None

        # 开读    
        fac_df = pd.read_feather(fpath)
        if fac_df.empty:
            return None
            
        # 标准化列名
        if 'index' in fac_df.columns: 
            fac_df.rename(columns={'index': 'stock'}, inplace=True)
        if 'StockID' in fac_df.columns: 
            fac_df.rename(columns={'StockID': 'stock'}, inplace=True)
            
        # 只留要的列
        if target_columns:
            available_cols = [col for col in target_columns if col in fac_df.columns]
            if available_cols:
                keep_cols = ['stock'] + available_cols
                fac_df = fac_df[keep_cols]
        
        # 添加时间和优化数据类型
        fac_df['T_date'] = dt
        fac_df['stock'] = fac_df['stock'].astype('category')
        
        if MEMORY_EFFICIENT:
            fac_df = optimize_dtypes(fac_df)
            
        return fac_df
        
    except Exception as e:
        if 'feather' not in str(e).lower():
            print(f' 读取{fpath}失败: {e}')
        return None

def merge_factor_data(signals):
    # 合并因子数据
    print("第二步：合并因子数据")

    
    factor_data_file = 'signals_with_factors.pkl'
    if not FORCE_REBUILD and os.path.exists(factor_data_file):
        merged = pd.read_pickle(factor_data_file)
        print(f"数据形状: {merged.shape}")
        return merged
    
    print("合并因子数据ing")
    print_memory_usage("开始合并")
    
    # 1. 换手率数据
    print("处理换手率数据...")
    turnover_df = pd.read_feather(turnover_path)
    turnover_df['TRADE_DATE'] = pd.to_datetime(turnover_df['TRADE_DATE'])
    
    if MEMORY_EFFICIENT:
        turnover_df = optimize_dtypes(turnover_df)
    
    turnover_long = turnover_df.melt(id_vars='TRADE_DATE', var_name='stock', value_name='turnover')
    turnover_long['T_date'] = turnover_long['TRADE_DATE']
    turnover_long['stock'] = turnover_long['stock'].astype('category')
    turnover_long = turnover_long[['T_date', 'stock', 'turnover']]
    
    del turnover_df
    gc.collect()
    
    # 2. 因子文件处理
    need_dates = set(signals['T_date'].dt.normalize())
    factor_files = []
    for p in factor_paths:
        if os.path.exists(p):
            factor_files.extend([
                os.path.join(p, f) for f in os.listdir(p) 
                if f.endswith('.fea') and len(f) >= 12
            ])
    
    print(f'因子文件数量: {len(factor_files):,}')
    
    # 3. 批量并行读取
    all_factors = []
    batch_size = 200
    
    for i in range(0, len(factor_files), batch_size):
        batch_files = factor_files[i:i + batch_size]
        print(f"处理第{i//batch_size + 1}批文件 ({len(batch_files)}个)...")
        
        tasks = [(fpath, need_dates, None) for fpath in batch_files]
        
        batch_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(read_single_factor_file_optimized, task) for task in tasks]
            
            for fut in tqdm(concurrent.futures.as_completed(futures), 
                          total=len(futures), desc=f'批次{i//batch_size + 1}'):
                result = fut.result()
                if result is not None and not result.empty:
                    batch_results.append(result)
        
        if batch_results:
            try:
                batch_combined = pd.concat(batch_results, ignore_index=True, sort=False)
                batch_combined = batch_combined.drop_duplicates(
                    subset=['T_date', 'stock'], keep='last'
                )
                all_factors.append(batch_combined)
            except Exception as e:
                print(f"批次{i//batch_size + 1}合并失败: {e}")
        
        del batch_results
        gc.collect()
    
    # 4. 合并所有因子数据
    if all_factors:
        factor_all = pd.concat(all_factors, ignore_index=True, sort=False)
        factor_all = factor_all.drop_duplicates(subset=['T_date', 'stock'], keep='last')
        print(f"因子数据形状: {factor_all.shape}")
        del all_factors
    else:
        print("未读取到因子数据orz")
        factor_all = pd.DataFrame(columns=['T_date', 'stock'])
    
    # 5. 最终合并
    merged = signals.merge(turnover_long, on=['T_date', 'stock'], how='left')
    
    if not factor_all.empty:
        merged = merged.merge(factor_all, on=['T_date', 'stock'], how='left', suffixes=('', ''))
    
    # 保存中间结果
    merged.to_pickle(factor_data_file)
    print(f"因子数据已保存: {factor_data_file}")
    
    print(f"最终数据形状: {merged.shape}")
    print_memory_usage("因子数据已合并")
    
    del turnover_long, factor_all
    gc.collect()
    
    return merged

# ========== 3 计算TWAP价格 ==========
def calculate_twap_prices(merged):
    #计算TWAP买卖价格#
    print("第三步：计算TWAP价格")

    price_data_file = 'signals_with_prices.pkl'
    if not FORCE_REBUILD and os.path.exists(price_data_file):
        print(f"已存在价格数据文件: {price_data_file}")
        merged = pd.read_pickle(price_data_file)
        print(f"数据形状: {merged.shape}")
        return merged
    
    # 预先创建stockid映射
    merged['stockid'] = merged['stock'].apply(get_stockid)
    buy_price = np.full(len(merged), np.nan, dtype=np.float32)
    
    grouped = merged.groupby('T1_date')
    processed_dates = 0
    
    for t1_date, sub in tqdm(grouped, desc='买入TWAP'):
        fname = pd.Timestamp(t1_date).strftime('%Y%m%d') + '.fea'
        fpath = os.path.join(min_data_path, fname)
        
        if not os.path.exists(fpath): 
            continue
            
        try:
            mindata = pd.read_feather(fpath)
            if 'StockID' not in mindata.columns: 
                continue
                
            df_buy = mindata[
                (mindata['time'] >= buy_time1) & (mindata['time'] <= buy_time2)
            ]
            
            if df_buy.empty:
                continue
            
            # twap
            stockids = sub['stockid'].unique()
            twap_map = (df_buy[df_buy['StockID'].isin(stockids)]
                       .groupby('StockID')['price']
                       .mean()
                       .to_dict())
            
            mask = sub['stockid'].isin(twap_map.keys())
            valid_indices = sub[mask].index
            valid_stockids = sub.loc[valid_indices, 'stockid']
            buy_price[valid_indices] = [twap_map[sid] for sid in valid_stockids]
            
            processed_dates += 1
            
        except Exception:
            continue
    
    merged['buy_price'] = buy_price
    print(f"买入TWAP计算完成")
    
    # 计算卖出TWAP
    try:
        twap_df = pd.read_pickle(twap_df_path)
        if not np.issubdtype(twap_df.index.dtype, np.datetime64):
            twap_df.index = pd.to_datetime(twap_df.index)
        
        sell_price = np.full(len(merged), np.nan, dtype=np.float32)
        
        for date, group in tqdm(merged.groupby('T2_date'), desc='卖出TWAP'):
            if date in twap_df.index:
                date_prices = twap_df.loc[date]
                for idx, stock in zip(group.index, group['stock']):
                    if stock in date_prices:
                        sell_price[idx] = date_prices[stock]
        
        merged['sell_price'] = sell_price
        print("卖出TWAP计算完成")
        
    except Exception as e:
        print(f"卖出TWAP计算失败: {e}")
        merged['sell_price'] = np.nan
    
    # 保存价格数据
    merged.to_pickle(price_data_file)
    print(f"价格数据已保存: {price_data_file}")
    print_memory_usage("价格计算完成")
    
    return merged

# ========== 4 计算收益指标 ==========
def calculate_returns(merged):
    #计算收益率指标#
    print("第四步：计算收益指标")
    
    # 过滤有效数据
    before_count = len(merged)
    # 买入和卖出价都要有信号
    valid_mask = (merged['buy_price'].notna() & merged['sell_price'].notna())
    merged = merged[valid_mask].copy()
    print(f"有效数据: {len(merged):,} / 原始数据: {before_count:,}")
    
    # 计算收益指标
    merged['pnl'] = merged['sell_price'] - merged['buy_price']
    merged['ret'] = merged['pnl'] / merged['buy_price']
    merged['is_win'] = (merged['pnl'] > 0).astype(np.int8)
    
    # 数据类型优化
    if MEMORY_EFFICIENT:
        merged = optimize_dtypes(merged)
    
    print(f"平均收益率: {merged['ret'].mean():.4f}")
    print(f"总胜率: {merged['is_win'].mean():.4f}")
    
    return merged

# ========== 主函数 ==========
def main():

    start_time = datetime.now()
    print_memory_usage("程序开始")
    
    # 检查输出文件是否已存在
    if not FORCE_REBUILD and os.path.exists(OUTPUT_SIGNALS_FILE):
        print(f"\n找到输出文件: {OUTPUT_SIGNALS_FILE}")
        
        # 加载并显示基本统计
        merged = pd.read_pickle(OUTPUT_SIGNALS_FILE)
        print(f"数据形状: {merged.shape}")
        print(f"训练数据: {len(merged[merged['year'].isin(train_years)]):,}")
        print(f"测试数据: {len(merged[merged['year'].isin(test_years)]):,}")
        
        factor_cols = [c for c in merged.columns if c not in 
                      {'T_date','T1_date','T2_date','stock','stockid','year',
                       'buy_price','sell_price','pnl','ret','is_win'}]
        print(f"因子数量: {len(factor_cols)}")
        return merged
    
    # 执行数据准备流程
    try:
        # 1: 生成触板信号
        signals = generate_touch_signals()
        
        # 2: 合并因子数据  
        merged = merge_factor_data(signals)
        
        # 3: 计算TWAP价格
        merged = calculate_twap_prices(merged)
        
        # 4: 计算收益指标
        merged = calculate_returns(merged)
        
        
        merged.to_pickle(OUTPUT_SIGNALS_FILE)
        print(f"最终数据已保存: {OUTPUT_SIGNALS_FILE}")
        
        # 生成摘要信息
        summary = {
            'total_signals': len(merged),
            'train_signals': len(merged[merged['year'].isin(train_years)]),
            'test_signals': len(merged[merged['year'].isin(test_years)]),
            'factor_count': len([c for c in merged.columns if c not in 
                               {'T_date','T1_date','T2_date','stock','stockid','year',
                                'buy_price','sell_price','pnl','ret','is_win'}]),
            'avg_return': float(merged['ret'].mean()),
            'total_winrate': float(merged['is_win'].mean()),
            'processing_time': str(datetime.now() - start_time)
        }
        
        import json
        with open(OUTPUT_SUMMARY_FILE, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"摘要信息已保存: {OUTPUT_SUMMARY_FILE}")
        
        # 显示最终统计

        print("数据准备阶段完成！")
        print(f"总信号数量: {summary['total_signals']:,}")
        print(f"训练期信号: {summary['train_signals']:,}")
        print(f"测试期信号: {summary['test_signals']:,}")
        print(f"因子数量: {summary['factor_count']}")
        print(f"平均收益率: {summary['avg_return']:.4f}")
        print(f"总胜率: {summary['total_winrate']:.4f}")
        print(f"处理时间: {summary['processing_time']}")
        
        print_memory_usage("程序结束")
        return merged
        
    except Exception as e:
        print(f"数据准备阶段发生错误: {e}")
        raise

if __name__ == '__main__':
    merged = main()