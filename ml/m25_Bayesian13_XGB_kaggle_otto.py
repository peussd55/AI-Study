### <<34>>

# 베이지안옵티마이제이션으로 최적의 파라미터찾기
# 부스팅류는 y데이터 종류(이진, 다중 등)에 따른 손실함수 따로 지정안해줘도 알아서 처리함.

import numpy as np
import pandas as pd
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
from sklearn.preprocessing import LabelEncoder

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

le = LabelEncoder()     # xgboost 쓰려면 y 정수형으로 라벨링(0~8)
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
bayesian_params = {     # 베이지안 파라미터는 실수형범위취급
    'n_estimators':(100,500),
    'learning_rate' : (0.001, 0.5),
    'max_depth' : (3,10),
    # 'num_leaves' : (24, 40),            # XGBRFRegressor 에 없는 파라미터
    # 'min_child_samples' : (10, 200),    # XGBRFRegressor 에 없는 파라미터
    'min_child_weight' : (1, 50),
    'gamma': (0, 5),
    'subsample' : (0.5, 2),
    'colsample_bytree' : (0.5, 1),
    'colsample_bylevel' : (0.5, 1),
    # 'max_bin' : (9, 500),               # 버전에 따라 될수도있고 안될수도있음
    'reg_lambda' : (0, 100),              # 디폴트 1 // L2 정규화 // 랏지
    'reg_alpha' : (0, 10),                # 디폴트 0 // L1 정규화 // 라쏘

}

# 2. 모델
# 블랙박스함수 정의
def xgb_cv(n_estimators, learning_rate, max_depth, min_child_weight, gamma,
           subsample, colsample_bytree, colsample_bylevel, reg_lambda, reg_alpha):

    params = {
        
        'n_estimators': int(n_estimators),                  # 정수형으로 받아야함. XGB의 epochs에 해당함.
        'learning_rate': learning_rate,
        'max_depth': int(round(max_depth)),                 # 정수형으로 받아야함. 그냥 int만쓰면 내림처리된다.
        'min_child_weight': int(round(min_child_weight)),   # 정수형으로 받아야함. 그냥 int만쓰면 내림처리된다.
        'gamma': gamma,
        'subsample': max(min(subsample, 1), 0),             # subsample 값은 0 ~ 1 사이여야함
        'colsample_bytree': colsample_bytree,
        'colsample_bylevel': colsample_bylevel,
        'reg_lambda': max(reg_lambda, 0),                   # reg_lambda은 0이상어야함
        'reg_alpha': reg_alpha,
    }
    model = XGBClassifier(**params, n_jobs=-1, early_stopping_rounds=10,)
    
    model.fit(x_train, y_train, 
              eval_set = [(x_test, y_test)],
              verbose=0,
              )
    y_pred = model.predict(x_test)
    print('y_pred:', y_pred)
    # 베이지안에서는 훈련에 사용할 손실함수를 직접 작성한다.(여기선 r2 또는 accuracy 사용)
    acc = accuracy_score(y_test, y_pred)
    print('acc:', acc)
    return acc
    
optimizer = BayesianOptimization(
    f=xgb_cv,   # 최대값을 뽑아야하기때문에 그냥씀. 최소값을 뽑아야하면 -xgb_cv해야함 (ex : -(0.001)과 -(0.1)을 비교하면 -(0.001)가 크기때문에 0.001에 해당하는 파라미터를 반환함)
    pbounds=bayesian_params,
    random_state=333,
    verbose=2
)

# 3. 훈련
n_iter = 100
start = time.time()
optimizer.maximize(init_points=5, n_iter=n_iter)    # maximize : 무조건 target이 큰 파라미터만 반환
end = time.time()

print(optimizer.max)
# {'target': 0.8226405946994182, 'params': {'n_estimators': 410.2390714364008, 'learning_rate': 0.5, 'max_depth': 10.0, 'min_child_weight': 3.530768652697704, 'gamma': 0.0, 'subsample': 2.0, 'colsample_bytree': 1.0, 'colsample_bylevel': 1.0, 'reg_lambda': 44.505244723948714, 'reg_alpha': 0.0}}
print(n_iter, '번 걸린 시간 :', round(end - start), '초')
# 100 번 걸린 시간 : 759 초