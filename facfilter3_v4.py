
"""
part 3 v4 ：因子筛选
任务
1. EXCLUDE_ST = True 时，加载st_status.fea，
   并剔除信号中的ST股票（与T_date、stock匹配）
2. 最先投票法先定最优方向，所有统计、聚类、评分、筛选流程均以最优方向打分
3. 聚类（但还没去重，有可能一个类里是一模一样的）
4. 先给聚类打分、再筛选打分后不那么相关的，再打分
"""
import os, json, warnings, time, copy
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
warnings.filterwarnings('ignore')

# ======= 参数区 =======
EXCLUDE_ST = True   # =True时剔除ST股票
ST_STATUS_FILE = '/mnt/tonglian_data2/support_data/st_status.fea'

STATS_FILE   = 'factor_group_stats_2020_2023_v3.csv'
OUTPUT_CLUSTER = './facfilter3_output/factor_cluster_result_2020_2023_nst.csv'
OUTPUT_FILTER  = './facfilter3_output/factor_filter_result_2020_2023_nst.csv'
OUTPUT_FINAL   = './facfilter3_output/monotonic_factor_group_result_2020_2023_nst.csv'
OUTPUT_SUMMARY = './facfilter3_output/factor_screening_summary_nst.json'

# 随便写的试试
CLUSTER_THRESH = 0.7
BASE_FILTERS = {
    'rank_icir_abs'  : 0.01,
    'cnt_dates'      : 20,
    'wilson_low'     : 0.45,
}
PREFERENCE_WEIGHTS = {
    'rank_icir_weight'    : 0.4,
    'monotonicity_weight' : 0.25,
    'stability_weight'    : 0.2,
    'significance_weight' : 0.15,
}

# ======== 1. 方向投票，处理统计指标 ========
def vote_direction(row)->str:
    # 综合 RankIC/收益差异/胜率差异/单调性 投票确定方向
    votes=[]
    if pd.notna(row['rank_ic_mean']):
        votes.append('b' if row['rank_ic_mean']>0 else 's')
    if pd.notna(row['ret_diff']):
        votes.append('b' if row['ret_diff']>0 else 's')
    if pd.notna(row['win_diff']):
        votes.append('b' if row['win_diff']>0 else 's')
    mono=row.get('monotonicity','none')
    if mono in ['ascending','weak_ascending']: votes.append('b')
    elif mono in ['descending','weak_descending']: votes.append('s')
    # 万一全没
    if not votes:             
        return 'b'
    if votes.count('b')>votes.count('s'):  return 'b'
    if votes.count('s')>votes.count('b'):  return 's'
    # 万一平票：看 rank_ic_mean （感觉rankic最牛）
    return 'b' if row.get('rank_ic_mean',0)>=0 else 's'

_SIGN_COLS = [
    'ic_mean','icir','rank_ic_mean','rank_icir',
    'ret_diff','win_diff',
    'win_Q0','win_Q4','ret_Q0','ret_Q4',
]
_RATIO_COLS = ['ic_positive_ratio','rank_ic_positive_ratio']
_MONO_MAP   = {
    'ascending'       :'descending',
    'descending'      :'ascending',
    'weak_ascending'  :'weak_descending',
    'weak_descending' :'weak_ascending',
}
def apply_optimal_direction(stats_df:pd.DataFrame)->pd.DataFrame:
    # 对每一行确定 label，并将指标统一为正向
    df=stats_df.copy()
    labels=[]
    for idx,row in df.iterrows():
        lab=vote_direction(row)
        labels.append(lab)
        if lab=='s':
            for c in _SIGN_COLS:
                if c in df.columns:
                    df.at[idx,c]= -df.at[idx,c]
            for c in _RATIO_COLS:
                if c in df.columns and pd.notna(df.at[idx,c]):
                    # 对统计的比例列作1-原值处理（如rankic positive ratio）
                    df.at[idx,c]= 1-df.at[idx,c]
            # 单调性翻转
            mono=row.get('monotonicity','none')
            if mono in _MONO_MAP:
                df.at[idx,'monotonicity']= _MONO_MAP[mono]
    df['label']=labels
    return df

# ======== 2. 相关性矩阵 ======
def calc_corr_matrix(train_df, factors:list):
    valid=[]
    dat={}
    for fac in factors:
        if fac not in train_df.columns: continue
        x=train_df[fac].astype(float).values
        if np.isnan(x).all() or np.nanstd(x)<1e-8: continue
        dat[fac]=(x-np.nanmean(x))/np.nanstd(x)
        valid.append(fac)
    n=len(valid)
    if n<=1: return None,valid
    corr=np.eye(n)
    for i in range(n):
        for j in range(i+1,n):
            vmask=~(np.isnan(dat[valid[i]])|np.isnan(dat[valid[j]]))
            if vmask.sum()<100: cv=0
            else:
                cv=np.corrcoef(dat[valid[i]][vmask],dat[valid[j]][vmask])[0,1]
                if not np.isfinite(cv): cv=0
            corr[i,j]=corr[j,i]=cv
    return corr,valid

def cluster_factors(corr, names, thr=0.7):
    if corr is None or len(names)<=1: return {1:names}
    dist=1-np.abs(corr)
    try:
        Z=linkage(squareform(dist,checks=False),'average')
        labels=fcluster(Z,t=1-thr,criterion='distance')
        clusters={}
        for f,l in zip(names,labels):
            clusters.setdefault(l,[]).append(f)
        return clusters
    except Exception as e:
        print(f'聚类失败:{e}，搞成单独分组')
        return {i+1:[n] for i,n in enumerate(names)}

# =========== 3. 综合评分 ==========
# 打分函数设计，之后两次打分直接用
def enhanced_score(r):
    score=0
    if pd.notna(r['rank_icir']):
        score+=abs(r['rank_icir'])*PREFERENCE_WEIGHTS['rank_icir_weight']
    mono=r.get('monotonicity','none')
    mono_bonus={'ascending':1,'descending':1,
                'weak_ascending':0.6,'weak_descending':0.6,
                'u_shape':0.3,'inverted_u':0.3}.get(mono,0)
    score+=mono_bonus*PREFERENCE_WEIGHTS['monotonicity_weight']
    if pd.notna(r['rank_ic_positive_ratio']):
        score+=max(0,r['rank_ic_positive_ratio']-0.5)*2*PREFERENCE_WEIGHTS['stability_weight']
    sig=0
    if pd.notna(r['p_rank_ic']) and r['p_rank_ic']<0.1: sig+=0.5
    if pd.notna(r['p_ret']) and r['p_ret']<0.1: sig+=0.5
    if pd.notna(r['wilson_low']) and r['wilson_low']>0.5: sig+=0.5
    score+=(sig/1.5)*PREFERENCE_WEIGHTS['significance_weight']
    return score

# 聚类内打分然后筛选
def select_best(cluster_factors, stats_df):
    sub=stats_df[stats_df['factor'].isin(cluster_factors)].copy()
    if sub.empty: return []
    sub['enhanced_score']=sub.apply(enhanced_score,axis=1)
    sub=sub.sort_values('enhanced_score',ascending=False)
    n=len(sub)
    keep=n if n<=3 else max(2,int(n*0.5))
    return sub.head(keep)['factor'].tolist()

# 聚类外过滤后再打分取70%
def progressive_filter(df):
    basic=(df['cnt_dates']>=BASE_FILTERS['cnt_dates']) & \
          (df['rank_icir'].abs()>=BASE_FILTERS['rank_icir_abs']) & \
          (df['wilson_low']>=BASE_FILTERS['wilson_low'])
    filt=df[basic].copy()
    if filt.empty: return filt
    filt['enhanced_score']=filt.apply(enhanced_score,axis=1)
    keep=max(10,int(len(filt)*0.7))
    return filt.nlargest(keep,'enhanced_score')

# =========== 4. 剔除ST股票信号的工具函数 ========
def filter_st_signals(df_signals):
    # 去除ST股票的所有信号
    if not os.path.exists(ST_STATUS_FILE):
        print(f'未找到ST文件: {ST_STATUS_FILE}')
        return df_signals
    st_df = pd.read_feather(ST_STATUS_FILE)
    # 规整长表（T_date, stock, isst）
    if 'TRADE_DATE' in st_df.columns: st_df.rename(columns={'TRADE_DATE':'T_date'},inplace=True)
    st_long = st_df.melt(id_vars='T_date',var_name='stock',value_name='isst')
    st_long['T_date'] = pd.to_datetime(st_long['T_date'])
    # 仅保留ST=1记录
    st_long = st_long[st_long['isst'] == 1]
    # 用merge筛掉信号
    orig_n = len(df_signals)
    df = df_signals.copy()
    df['T_date'] = pd.to_datetime(df['T_date'])
    df = df.merge(st_long[['T_date','stock']], on=['T_date','stock'], how='left', indicator=True)
    df = df[df['_merge'] == 'left_only'].drop('_merge',axis=1)
    print(f' 剔除掉的ST信号 {orig_n-len(df):,} ')
    return df

# ========= 5. 主流程 ==========
def main():
    t0=time.time()
    if not os.path.exists(STATS_FILE):
        raise FileNotFoundError(STATS_FILE)
    stats_raw=pd.read_csv(STATS_FILE)
    print(f'原始统计 {len(stats_raw):,} 条')

    # 5.1 剔除ST股票影响
    DATA_PKL='touch_signals_with_factors_and_returns.pkl'
    train=None
    if os.path.exists(DATA_PKL):
        train=pd.read_pickle(DATA_PKL)
        train['T_date']=pd.to_datetime(train['T_date'])
        train=train[train['T_date'].dt.year.isin([2020,2021,2022,2023])]
        if EXCLUDE_ST:
            train = filter_st_signals(train)
    else:
        if EXCLUDE_ST:
            print('未找到信号主表，快去找')

    # 5.2 方向投票并翻转
    stats_opt=apply_optimal_direction(stats_raw)
    print('方向投票、指标统一好了')

    # 5.3 相关性矩阵 & 聚类
    corr,valid=calc_corr_matrix(train,stats_opt['factor'].tolist()) if train is not None else (None,stats_opt['factor'].tolist())
    clusters=cluster_factors(corr,valid,CLUSTER_THRESH)
    pd.DataFrame([{'factor':f,'cluster_id':cid,'cluster_size':len(fs)}
                  for cid,fs in clusters.items() for f in fs]
                ).to_csv(OUTPUT_CLUSTER,index=False)
    print(f'聚类完成，共 {len(clusters)} 组 → {OUTPUT_CLUSTER}')

    # 5.4 组内优选
    chosen=[]
    for cid,fac_list in clusters.items():
        best=select_best(fac_list,stats_opt)
        chosen.extend(best)
    stats_chosen=stats_opt[stats_opt['factor'].isin(chosen)]
    print(f'组内选后剩 {len(stats_chosen)} 因子')

    # 5.5 渐进式筛选
    stats_filtered=progressive_filter(stats_chosen)
    stats_filtered.to_csv(OUTPUT_FILTER,index=False)
    print(f'渐进筛选后 {len(stats_filtered)} 因子 → {OUTPUT_FILTER}')

    # 5.6 最终结果
    final=stats_filtered.sort_values('enhanced_score',ascending=False)
    final_cols=['factor','label','enhanced_score','rank_icir','monotonicity',
                'wilson_low','rank_ic_positive_ratio','ret_diff','win_diff']
    final[final_cols].to_csv(OUTPUT_FINAL,index=False)
    print(f'最后输出 {len(final)} 因子 → {OUTPUT_FINAL}')

    # 5.7 摘要
    summary={
        'total_factors':len(stats_raw),
        'after_opt_dir':len(stats_opt),
        'clusters':len(clusters),
        'after_cluster_select':len(stats_chosen),
        'after_progressive_filter':len(stats_filtered),
        'final':len(final),
        'direction_distribution':final['label'].value_counts().to_dict(),
        'elapsed_sec':round(time.time()-t0,2),
        'exclude_st':EXCLUDE_ST,
        'parameters':{
            'cluster_thresh':CLUSTER_THRESH,
            'base_filters':BASE_FILTERS,
            'weights':PREFERENCE_WEIGHTS
        }
    }
    with open(OUTPUT_SUMMARY,'w') as f:
        json.dump(summary,f,indent=2,ensure_ascii=False)
    print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=='__main__':
    main()

