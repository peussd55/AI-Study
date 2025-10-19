### <<34>>

import numpy as np
import pandas as pd
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
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.ensemble import VotingClassifier


seed = 333
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
path = './_data/kaggle/otto/'
# 맨 앞, 공통 컬럼 datetime을 인덱스 컬럼으로 지정
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')
# # target컬럼 레이블 인코딩(원핫 인코딩 사전작업)###
# # 정수형을 직접 원핫인코딩할경우 keras, pandas, sklearn 방식 모두 가능하지만 문자형태로 되어있을 경우에는 pandas방식만 문자열에서 직접 원핫인코딩이 가능하다.
# le = LabelEncoder() # 인스턴스화
# train_csv['target'] = le.fit_transform(train_csv['target'])   # fit 함수 + transform 함친 합친 함수 : 변환해서 적용
# train data x로 y 분리
x = train_csv.drop(['target'], axis=1)
print(x)
print('x type:',type(x))
y = train_csv['target']
print('y type:',type(y))
print(y)

le = LabelEncoder()         # xgb가 있음에도 여기서 라벨인코딩안해도 에러가 나지않으나 안전하게 인코딩해준다.
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8, 
    stratify=y,   # 분류데이터에서만 사용가능
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333,)

# 2. 모델
xgb = XGBClassifier()
lgbm = LGBMClassifier()
cat = CatBoostClassifier()

model = VotingClassifier(
    estimators = [('XGB', xgb), ('LGBM', lgbm), ('CAT', cat)],
    # voting = 'hard', # 디폴트
    voting = 'soft',
)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 :', results)

# 0.8115707821590175
# 0.8115707821590175
