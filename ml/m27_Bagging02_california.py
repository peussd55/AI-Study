### <<34>>

# 배깅 : 한 모델을 중복허용 랜덤샘플링하며 학습

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
import time
import random
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor

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
# model = DecisionTreeRegressor()

model = BaggingRegressor(DecisionTreeRegressor(),
                         n_estimators=100,
                         n_jobs=-1,
                         random_state=333,
                         bootstrap=True,   # 데이터 중복사용 여부 // 디폴트 : True // False하면 훈련데이터가 적어서 성능 떨어짐.
                         )

# model = RandomForestRegressor(random_state=333)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 :', results)
# 디시전트리만 쓸때 : 0.59521884870791
# 디시전트리 배깅했을 때 (bootstrap=True) : 0.7995724581590308
# 디시전트리 배깅했을 때 (bootstrap=False) : 0.623862646335765
# 랜덤포레스트 쓸때 : 0.7989622154915357
# 랜덤포레스트는 디시전트리를 배깅한 모델이다.