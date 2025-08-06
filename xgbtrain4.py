
"""
part 4 xgboost分支
任务：
1. part3_v2 输出的因子
2. 2020-2023 训练集做 K 折 + Optuna 调参，调参目标最大化ic
3. 对训练集和测试集打分，（xgboost的未来收益率预测收益）
"""
import os, pickle, warnings, optuna
import pandas as pd, numpy as np, xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost.callback import EarlyStopping

warnings.filterwarnings('ignore')

# =========  路径 =============
TRAIN_PKL = './facstandard4_output/signals_merged_train_2020_2023.pkl'
TEST_PKL  = './facstandard4_output/signals_merged_test_2024.pkl'
FACTOR_CSV= './facfilter3_output/monotonic_factor_group_result_2020_2023_nst.csv'
OUT_DIR   = './xgb_branch_output'
os.makedirs(OUT_DIR, exist_ok=True)

# =========  读取数据 ============
mono = pd.read_csv(FACTOR_CSV)
factor_labels = dict(zip(mono['factor'], mono['label']))
factors = list(factor_labels.keys())
print(f' {len(factors)} 个筛选因子作特征')

train = pd.read_pickle(TRAIN_PKL)
test  = pd.read_pickle(TEST_PKL)

# 方向调整
for fac, lbl in factor_labels.items():
    for df in (train, test):
        if fac not in df.columns:
            df[fac] = np.nan
        if lbl == 's':            
            df[fac] = -df[fac]

X_train = train[factors].fillna(0).astype(np.float32)
y_train = train['ret'].astype(np.float32)
X_test  = test[factors].fillna(0).astype(np.float32)
y_test  = test['ret'].astype(np.float32)

scaler = StandardScaler().fit(X_train)      
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

# ============= K 折 + Optuna ===========
N_FOLDS = 5
tscv = TimeSeriesSplit(n_splits=N_FOLDS)
models, fold_scores = [], []

def objective(trial, X_tr, y_tr, X_val, y_val):
    params = {
        # —— Optuna 超参搜索 —— 
        'n_estimators'     : trial.suggest_int ('n_estimators',   200, 800),
        'max_depth'        : trial.suggest_int ('max_depth',        3, 12),
        'learning_rate'    : trial.suggest_float('learning_rate', 1e-3, 2e-2, log=True),
        'subsample'        : trial.suggest_float('subsample',      0.5, 0.9),
        'colsample_bytree' : trial.suggest_float('colsample_bytree',0.3, 0.9),
        'gamma'            : trial.suggest_float('gamma',          0,   0.1),
        'min_child_weight' : trial.suggest_float('min_child_weight',0.5, 5),
        'reg_alpha'        : trial.suggest_float('reg_alpha',      0,   1),
        'reg_lambda'       : trial.suggest_float('reg_lambda',     0.1, 1),
        # —— 固定参数 ——
        'objective'        : 'reg:squarederror',
        'tree_method'      : 'hist',
        'random_state'     : 42,
        # —— 早停直接放这里 ——
        'early_stopping_rounds': 50
    }

    model = xgb.XGBRegressor(**params)

    # 触发早停必须有验证集
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False                  
    )

    pred = model.predict(X_val)
    ic   = np.corrcoef(pred, y_val)[0, 1]

    # 最小化目标（-ic）（最大化ic）
    return -ic if np.isfinite(ic) else 0.0



print('\n>>> 开始 K 折训练')
for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
    print(f'Fold {fold+1}/{N_FOLDS}')
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    if fold == 0:        # 仅第一折做贝叶斯调参，如研报
        study = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(lambda t: objective(t, X_tr, y_tr, X_val, y_val),
                       n_trials=40, n_jobs=1, show_progress_bar=True)
        best_params = study.best_params
        best_params.update({
            'objective'             : 'reg:squarederror',
            'tree_method'           : 'hist',
            'random_state'          : 42,
            'early_stopping_rounds' : 50       
        })
        print('  你是好参数:', best_params)
    #######################
    # 开练！
    model = xgb.XGBRegressor(**best_params)
    early_stop = xgb.callback.EarlyStopping(rounds=50, save_best=True)

    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                verbose=False)

    pred_val = model.predict(X_val)
    ic = np.corrcoef(pred_val, y_val)[0,1]
    # 分
    fold_scores.append({'fold':fold, 'ic':ic})
    print(f'  验证 IC = {ic:.4f}')
    models.append(model)

# ==========  预测 ==========
def avg_predict(m_list, X):
    return np.mean([m.predict(X) for m in m_list], axis=0)

train_pred = avg_predict(models, X_train)
test_pred  = avg_predict(models, X_test)

train_out = train[['T_date','stock','ret','is_win']].copy()
test_out  = test [['T_date','stock','ret','is_win']].copy()
train_out['xgb_pred'] = train_pred
test_out ['xgb_pred'] = test_pred

train_out.to_pickle(f'{OUT_DIR}/xgb_pred_train.pkl')
test_out .to_pickle(f'{OUT_DIR}/xgb_pred_test.pkl')
print('预测文件已保存')

imp = pd.DataFrame({
    'feature': factors,
    'importance': np.mean([m.feature_importances_ for m in models], axis=0)
}).sort_values('importance', ascending=False)
with open(f'{OUT_DIR}/xgb_models.pkl','wb') as f:
    pickle.dump({'models':models, 'params':best_params,
                 'scaler':scaler, 'feature_importance':imp,
                 'fold_scores':fold_scores}, f)
print('模型已保存')
