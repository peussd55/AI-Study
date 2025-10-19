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
path = './_data/dacon/diabetes/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

# 데이터 전처리
x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']
x = x.replace(0, np.nan).fillna(x.median())  # 0값을 NaN으로 변환 후 중앙값으로 대체

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
# acc : [0.79047619 0.75961538 0.74038462 0.74038462 0.75961538] 
# 평균 acc : 0.7581

# [kfold]
# acc : [0.77862595 0.78625954 0.75384615 0.76153846 0.69230769] 
# 평균 acc : 0.7545

y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)

acc = accuracy_score(y_test, y_pred)
print('cross_val_predict acc :', round(acc, 4))     # 0.771