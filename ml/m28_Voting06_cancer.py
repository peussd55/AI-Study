### <<34>>

import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')
import time
import random

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.ensemble import VotingClassifier


seed = 333
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8, 
    stratify=y,   # 분류데이터에서만 사용가능
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
xgb = XGBClassifier()
lgbm = LGBMClassifier()
cat = CatBoostClassifier()

model = VotingClassifier(
    estimators = [('XGB', xgb), ('LGBM', lgbm), ('CAT', cat)],
    # voting = 'hard', # 디폴트
    voting = 'soft',
    weights=[2,1,1],    # 각 모델에 가중치적용
)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 :', results)
# soft voting : 0.9824561403508771
# hard voting : 0.9824561403508771
