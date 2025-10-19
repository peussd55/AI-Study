### <<32>>

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
import warnings     # 경고무시
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

# 1. 데이터
datasets = load_digits()
x = datasets.data
y= datasets.target

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
# acc : [0.98611111 0.96180556 0.96515679 0.95818815 0.96515679] 
# 평균 acc : 0.9673

# [kfold]
# acc : [0.97777778 0.98333333 0.97771588 0.97214485 0.97493036] 
# 평균 acc : 0.9772

y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)

acc = accuracy_score(y_test, y_pred)
print('cross_val_predict acc :', round(acc, 4))     # 0.9556