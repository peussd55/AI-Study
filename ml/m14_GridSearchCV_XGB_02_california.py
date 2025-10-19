### <<32>>

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import  LinearSVC, SVC
import time
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.utils import  all_estimators
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# GridSearch : 모든 경우의 수를 탐색하는 방법

# 1. 데이터
x = datasets = fetch_california_housing()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8, 
    # stratify=y,   # 분류데이터에서만 사용가능
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333, )

parameters = [
    {'n_estimators':[100,500], 'max_depth':[6,10,12], 'learning_rate':[0.1, 0.01, 0.001]},  # 18
    {'max_depth':[6,8,10,12], 'learning_rate':[0.1, 0.01, 0.001]},  # 12
    {'min_child_weight':[2,3,5,10], 'learning_rate':[0.1, 0.01, 0.001]},    # 12
]

# 2. 모델
xgb = XGBRegressor()
model = GridSearchCV(xgb, param_grid=parameters, cv=kfold, 
                     verbose = 1,
                     refit=True,    # best 파라미터로 전체 훈련 데이터(x_train, y_train)를 다시 fit 1번 (디폴트)
                     n_jobs=-1,     # cpu를 풀 가동(모든 쓰레드를 가동)
                     )  # 12+18+12+1

# 3. 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()

print('최적의 매개변수 :', model.best_estimator_)
# 최적의 매개변수 : XGBRegressor(base_score=None, booster=None, callbacks=None,
#              colsample_bylevel=None, colsample_bynode=None,
#              colsample_bytree=None, device=None, early_stopping_rounds=None, 
#              enable_categorical=False, eval_metric=None, feature_types=None, 
#              gamma=None, grow_policy=None, importance_type=None,
#              interaction_constraints=None, learning_rate=0.1, max_bin=None,  
#              max_cat_threshold=None, max_cat_to_onehot=None,
#              max_delta_step=None, max_depth=6, max_leaves=None,
#              min_child_weight=None, missing=nan, monotone_constraints=None,  
#              multi_strategy=None, n_estimators=500, n_jobs=None,
#              num_parallel_tree=None, random_state=None, ...)

print('최적의 파라미터 :', model.best_params_)
# 최적의 파라미터 : {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 500}

# 4. 평가, 예측
print('best_score :', model.best_score_)
# best_score : 0.8437452115581335

print('model.score :', model.score(x_test, y_test))
# model.score : 0.8504349409098124

y_pred = model.predict(x_test)
print('r2_score :', r2_score(y_test, y_pred))
# r2_score : 0.8504349409098124

print('걸린시간 :', round(end - start), '초')
# 걸린시간 : 124 초
