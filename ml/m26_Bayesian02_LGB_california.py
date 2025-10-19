### <<34>>

# 베이지안옵티마이제이션으로 최적의 파라미터찾기
# 부스팅류는 y데이터 종류(이진, 다중 등)에 따른 손실함수 따로 지정안해줘도 알아서 처리함.

import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import time
import random
import lightgbm as lgbm

seed = 333
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8, 
    # stratify=y,   # 분류데이터에서만 사용가능
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

bayesian_params = {
    'num_leaves': (20, 150),              # 하나의 트리에서 최대 리프 개수 (복잡도 제어)
    'learning_rate': (0.01, 0.3),         # 학습률
    'max_depth': (3, 15),                 # 트리 최대 깊이 (-1은 제한 없음)
    'min_data_in_leaf': (10, 100),        # 하나의 리프에 필요한 최소 데이터 수
    'feature_fraction': (0.5, 1.0),       # 트리 생성 시 사용할 feature 비율
    'bagging_fraction': (0.5, 1.0),       # 트리 생성 시 사용할 데이터 샘플 비율
    'bagging_freq': (0, 10),              # bagging 수행 빈도(0이면 사용 안함)
    'lambda_l1': (0, 5),                  # L1 정규화
    'lambda_l2': (0, 5),                  # L2 정규화
    'min_gain_to_split': (0, 1),          # 노드 분할 최소 이득
    'max_bin': (100, 500),                # feature bin 개수
    'n_estimators': (100, 500),           # 부스팅 라운드 수(트리 개수)
}

# 2. 모델
# 블랙박스함수 정의
def lgb(num_leaves, learning_rate, max_depth, min_data_in_leaf, feature_fraction,
           bagging_fraction, bagging_freq, lambda_l1, lambda_l2, min_gain_to_split, max_bin, n_estimators):

    params = {
        'num_leaves': int(round(num_leaves)),
        'learning_rate': learning_rate,
        'max_depth': int(round(max_depth)),
        'min_data_in_leaf': int(round(min_data_in_leaf)),
        'feature_fraction': feature_fraction,
        'bagging_fraction': bagging_fraction,
        'bagging_freq': int(round(bagging_freq)),
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2,
        'min_gain_to_split': min_gain_to_split,
        'max_bin': int(round(max_bin)),
        'n_estimators': int(round(n_estimators)),
    }
    model = LGBMRegressor(**params, n_jobs=-1, verbosity=-1,)   # verbosity=-1 : verbose 옵션 끔

    model.fit(x_train, y_train, 
              eval_set = [(x_test, y_test)],
              callbacks=[lgbm.early_stopping(10)],   # lightbm은 여기서 early_stopping 옵션 지정
              )
    y_pred = model.predict(x_test)
    print('y_pred:', y_pred)
    # 베이지안에서는 훈련에 사용할 손실함수를 직접 작성한다.(여기선 r2 또는 accuracy 사용)
    r2 = r2_score(y_test, y_pred)
    print('r2:', r2)
    return r2
    
optimizer = BayesianOptimization(
    f=lgb,   # 최대값을 뽑아야하기때문에 그냥씀. 최소값을 뽑아야하면 -xgb_cv해야함 (ex : -0.001과 -0.1을 비교하면 -0.001을 크기때문에 -0.001에 해당하는 파라미터를 반환함)
    pbounds=bayesian_params,
    random_state=333,
    verbose=2,
)

# 3. 훈련
n_iter = 100
start = time.time()
optimizer.maximize(init_points=5, n_iter=n_iter)
end = time.time()

print(optimizer.max)
# {'target': 0.8463423686428435, 'params': {'bagging_fraction': 1.0, 'bagging_freq': 10.0, 'feature_fraction': 0.5, 'lambda_l1': 0.0, 'lambda_l2': 0.0, 'learning_rate': 0.10203409207754803, 'max_bin': 440.9481443176052, 'max_depth': 15.0, 'min_data_in_leaf': 96.71070672822451, 'min_gain_to_split': 0.0, 'n_estimators': 149.1006563089701, 'num_leaves': 78.52099956639493}}
print(n_iter, '번 걸린 시간 :', round(end - start), '초')
# 100 번 걸린 시간 : 40 초