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
path = './_data/kaggle/santander/'           
# 맨 앞, 공통 컬럼 datetime을 인덱스 컬럼으로 지정
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')
print(train_csv.shape, test_csv.shape, submission_csv.shape)  # (200000, 202) (200000, 201) (200000, 2)
print(train_csv['target'].value_counts())
# 이진분류 (불균형)
# 0    179902
# 1     20098
# train data x로 y 분리
x = train_csv.drop(['target'], axis=1)
print(x)
y = train_csv['target']

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


# [kfold]
# acc : [0.97777778 0.98333333 0.97771588 0.97214485 0.97493036] 
# 평균 acc : 0.9772

y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)

acc = accuracy_score(y_test, y_pred)
print('cross_val_predict acc :', round(acc, 4))     # 