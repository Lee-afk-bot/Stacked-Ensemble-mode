import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings

# 忽略一些繁琐的警告
warnings.filterwarnings('ignore')

# ==========================================
# 1. 物理特征工程 (必须包含，否则无法达到 0.99)
# ==========================================
def feature_engineering(df):
    data = df.copy()
    # 论文核心：引入 √t 和 T*t 交互项
    data['sqrt_Time'] = np.sqrt(data['Time'])
    data['Temp_x_Time'] = data['Temperature'] * data['Time']
    return data

# ==========================================
# 2. 定义每个模型的超参数搜索空间
#    这里设置的范围覆盖了绝大多数高性能模型的参数区间
# ==========================================
def get_param_grids():
    # 1. XGBoost 搜索空间
    xgb_params = {
        'n_estimators': [500, 1000, 2000],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 2, 5]
    }

    # 2. LightGBM 搜索空间
    lgb_params = {
        'n_estimators': [500, 1000, 2000],
        'num_leaves': [31, 50, 100],
        'learning_rate': [0.01, 0.05, 0.1],
        'feature_fraction': [0.6, 0.8, 1.0],
        'bagging_fraction': [0.6, 0.8, 1.0],
        'bagging_freq': [1, 5]
    }

    # 3. Random Forest 搜索空间
    rf_params = {
        'n_estimators': [200, 500, 1000],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    # 4. CatBoost 搜索空间
    cb_params = {
        'iterations': [500, 1000],
        'depth': [4, 6, 8, 10],
        'learning_rate': [0.03, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5, 7]
    }

    # 5. Decision Tree (作为基模型较弱，简单调优即可)
    dt_params = {
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10]
    }

    # 6. MLP (神经网络)
    mlp_params = {
        'hidden_layer_sizes': [(50,50), (100,50), (100,100,50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01]
    }

    return xgb_params, lgb_params, rf_params, cb_params, dt_params, mlp_params

# ==========================================
# 3. 自动微调函数
# ==========================================
def tune_model(model, param_grid, X_train, y_train, model_name="Model"):
    print(f"正在微调 {model_name} ...")
    # 使用 RandomizedSearchCV 进行搜索 (比 GridSearch 快 10 倍)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=20,  # 每个模型尝试 20 种组合，可根据算力增加到 50
        scoring='r2',
        cv=3,       # 3折交叉验证
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    search.fit(X_train, y_train)
    print(f"  -> {model_name} 最佳 R2: {search.best_score_:.4f}")
    print(f"  -> 最佳参数: {search.best_params_}")
    return search.best_estimator_

# ==========================================
# 4. 主执行流程
# ==========================================
def main_optimization_workflow(csv_file_path):
    # 1. 加载与预处理
    print("加载数据中...")
    try:
        df = pd.read_csv(csv_file_path)
    except:
        print("错误：请提供正确的CSV文件路径")
        return

    # 重命名以匹配逻辑
    rename_map = {'Holding Time': 'Time', 'MoO3 Content': 'MoO3', 'Aspect Ratio': 'AR'}
    for k, v in rename_map.items():
        if k in df.columns:
            df = df.rename(columns={k: v})

    # 特征工程
    X = feature_engineering(df[['Temperature', 'Time', 'MoO3']])
    y = df[['Length', 'Width', 'AR']]  # 多目标

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 注意：为了简化微调，我们通常针对某一个主要目标（如 AR 或 Length）进行调优，
    # 或者对每个目标取平均。这里我们以 'Length' 为例来寻找最佳骨干参数，
    # 因为物理规律最显著。如果算力允许，应对 y 的每一列分别调优。
    y_train_tuning = y_train['Length']

    # 2. 获取搜索空间
    xgb_p, lgb_p, rf_p, cb_p, dt_p, mlp_p = get_param_grids()

    # 3. 逐个微调基模型 (Level-0)
    print("\n=== 开始 Level-0 基模型微调 ===")

    best_xgb = tune_model(xgb.XGBRegressor(n_jobs=-1, random_state=42), xgb_p, X_train, y_train_tuning, "XGBoost")
    best_lgb = tune_model(lgb.LGBMRegressor(n_jobs=-1, random_state=42, verbose=-1), lgb_p, X_train, y_train_tuning, "LightGBM")
    best_rf = tune_model(RandomForestRegressor(n_jobs=-1, random_state=42), rf_p, X_train, y_train_tuning, "RandomForest")
    best_cat = tune_model(cb.CatBoostRegressor(verbose=0, allow_writing_files=False, random_state=42), cb_p, X_train, y_train_tuning, "CatBoost")
    best_dt = tune_model(DecisionTreeRegressor(random_state=42), dt_p, X_train, y_train_tuning, "DecisionTree")
    best_mlp = tune_model(MLPRegressor(max_iter=1000, random_state=42), mlp_p, X_train, y_train_tuning, "MLP")

    # 4. 构建 Stacking 模型
    # 使用微调后的最佳估计器
    base_learners = [
        ('xgb', best_xgb),
        ('lgb', best_lgb),
        ('rf', best_rf),
        ('cat', best_cat),
        ('dt', best_dt),
        ('mlp', best_mlp)
    ]

    # Level-1 元学习器 (通常不需要太复杂的调优，给一个稳健的 XGB 即可)
    meta_learner = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        n_jobs=-1,
        random_state=42
    )

    print("\n=== 组装最终 Stacking 模型并进行多目标训练 ===")
    final_stacking = StackingRegressor(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )

    # 包装为多输出回归
    multi_target_model = MultiOutputRegressor(final_stacking)

    # 在所有目标上训练
    multi_target_model.fit(X_train, y_train)

    # 5. 最终验证
    y_pred = multi_target_model.predict(X_test)
    r2_final = r2_score(y_test, y_pred)

    print("\n" + "="*40)
    print(f"经过微调后的最终测试集 R2 Score: {r2_final:.4f}")
    if r2_final > 0.99:
        print("恭喜！模型已达到论文级精度！")
    else:
        print("提示：如果 R2 仍未达到 0.99，请检查是否使用了扩充后的30万条数据进行训练。")
    print("="*40)

    return multi_target_model

if __name__ == "__main__":
   "D:\硕士\晶须数据\新建 Microsoft Excel 工作表.xlsx"

    pass
