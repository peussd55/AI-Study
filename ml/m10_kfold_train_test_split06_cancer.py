### <<32>>

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
import warnings     # 경고무시
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor

# 1. 데이터
datasets = load_breast_cancer()
x, y = load_breast_cancer(return_X_y=True)
print(x.shape, y.shape) # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8, 
    stratify=y,   # 분류데이터에서만 사용가능
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333, )

# 2. 모델
model = RandomForestClassifier()

# 3. 훈련
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('acc :', scores, '\n평균 acc :', round(np.mean(scores), 4))
# [kfold_train_test_split]
# acc : [0.96703297 0.97802198 0.94505495 0.93406593 0.97802198] 
# 평균 acc : 0.9604

# [kfold]
# acc : [0.96491228 0.95614035 0.95614035 0.98245614 0.9380531 ] 
# 평균 acc : 0.9595

y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)

acc = accuracy_score(y_test, y_pred)
print('cross_val_predict acc :', round(acc, 4))     # 0.9298