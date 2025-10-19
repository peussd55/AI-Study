### <<34>>

# 모델 3개를 Voting해보자
# 보팅 : 여러모델을 돌려서 투표

import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.datasets import fetch_california_housing
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
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor


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

# 2. 모델
xgb = XGBRegressor()
lgbm = LGBMRegressor()
cat = CatBoostRegressor()

model = VotingRegressor(
    estimators = [('XGB', xgb), ('LGBM', lgbm), ('CAT', cat)],
    # voting = 'hard',
    # voting = 'soft',
    # 회귀에서는 하드보팅이 없기때문에 보팅 선택하는 파라미터없다.
    weights=[2,1,1],    # 각 모델에 가중치적용
)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 :', results)
# 0.8439799611951704
