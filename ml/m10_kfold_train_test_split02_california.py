### <<32>>

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
import warnings     # 경고무시
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor

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

# 2. 모델
model = HistGradientBoostingRegressor()

# 3. 훈련
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('r2_score :', scores, '\n평균 r2_score :', round(np.mean(scores), 4))
# [kfold_train_test_split]
# r2_score : [0.83978557 0.83866319 0.83032923 0.83098388 0.82708287] 
# 평균 r2_score : 0.8334

# [kfold]
# r2_score : [0.82844582 0.84063748 0.82240123 0.84105292 0.84572463] 
# 평균 r2_score : 0.8357

y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)

r2 = r2_score(y_test, y_pred)
print('cross_val_predict r2_score :', round(r2, 4))     # 0.8061