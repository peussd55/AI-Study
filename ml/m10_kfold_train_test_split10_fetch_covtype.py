### <<32>>

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
import warnings     # 경고무시
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 1. 데이터
datasets = fetch_covtype()
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
# acc : [0.95091543 0.95027    0.95106603 0.94967836 0.94998978] 
# 평균 acc : 0.9504

# [kfold]
# acc : [0.95471718 0.95568101 0.95552572 0.95539664 0.95605928] 
# 평균 acc : 0.9555

y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)

acc = accuracy_score(y_test, y_pred)
print('cross_val_predict acc :', round(acc, 4))     # 0.9085