### <<33>>

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
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
from sklearn.preprocessing import LabelEncoder
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# GridSearch : 모든 경우의 수를 탐색하는 방법

# 1. 데이터
datasets = fetch_covtype()
x = datasets.data
y= datasets.target
y = y - 1 # XGB는 0부터 시작하는 클래스 레이블 받음

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8, 
    stratify=y,   # 분류데이터에서만 사용가능
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333, )

parameters = [
    {'n_estimators':[100,500], 'max_depth':[6,10,12], 'learning_rate':[0.1, 0.01, 0.001]},  # 18
    {'max_depth':[6,8,10,12], 'learning_rate':[0.1, 0.01, 0.001]},  # 12
    {'min_child_weight':[2,3,5,10], 'learning_rate':[0.1, 0.01, 0.001]},    # 12
]

# 2. 모델
xgb = XGBClassifier()
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

print('최적의 파라미터 :', model.best_params_)

# 4. 평가, 예측
print('best_score :', model.best_score_)

print('model.score :', model.score(x_test, y_test))

y_pred = model.predict(x_test)
print('r2_score :', accuracy_score(y_test, y_pred))

print('걸린시간 :', round(end - start), '초')
"""

"""

# 5. 가중치 저장
import joblib
path = './_save/m15_cv_results/'
joblib.dump(model.best_estimator_, path + 'm15_best_model_10_fetch_covtype.joblib')
# 모델은 GridSearch로 돌렸지만 저장은 XGBoost로 저장된다.