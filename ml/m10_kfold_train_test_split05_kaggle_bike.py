### <<32>>

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
import warnings     # 경고무시
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor

# 1. 데이터
path = './_data/kaggle/bike/'
# 맨 앞, 공통 컬럼 datetime을 인덱스 컬럼으로 지정
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)   
print(x)
y = train_csv['count']
print(y)
print(y.shape)

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
# r2_score : [0.35550031 0.39120363 0.34382874 0.32001694 0.3565478 ] 
# 평균 r2_score : 0.3534

# [kfold]
# r2_score : [0.35352498 0.33077443 0.37605704 0.37007788 0.35513083] 
# 평균 r2_score : 0.3571

y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)

r2 = r2_score(y_test, y_pred)
print('cross_val_predict r2_score :', round(r2, 4))     # 0.2315