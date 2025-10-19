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
from lightgbm import LGBMClassifier, LGBMRegressor
import lightgbm as lgbm

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

# le = LabelEncoder()     # lightgbm은 학습때 자동인코딩해서 여기서 인코딩안해도 오류안남
# y = le.fit_transform(y)

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
    'num_leaves': (20, 150),              # 하나의 트리에서 최대 리프 개수 (복잡도 제어)
    'learning_rate': (0.01, 0.3),         # 학습률
    'max_depth': (3, 15),                 # 트리 최대 깊이 (-1은 제한 없음)
    'min_data_in_leaf': (10, 100),        # 하나의 리프에 필요한 최소 데이터 수
    'feature_fraction': (0.5, 1.0),       # 트리 생성 시 사용할 feature 비율
    'bagging_fraction': (0.5, 1.0),       # 트리 생성 시 사용할 데이터 샘플 비율
    'bagging_freq': (0, 10),              # bagging 수행 빈도(0이면 사용 안함)
    'lambda_l1': (0, 5),                  # L1 정규화
    'lambda_l2': (0, 5),                  # L2 정규화
    'min_gain_to_split': (0, 1),          # 노드 분할 최소 이득
    'max_bin': (100, 500),                # feature bin 개수
    'n_estimators': (100, 500),           # 부스팅 라운드 수(트리 개수)

}

# 2. 모델
# 블랙박스함수 정의
def lgb(num_leaves, learning_rate, max_depth, min_data_in_leaf, feature_fraction,
           bagging_fraction, bagging_freq, lambda_l1, lambda_l2, min_gain_to_split, max_bin, n_estimators):

    params = {
        'num_leaves': int(round(num_leaves)),
        'learning_rate': learning_rate,
        'max_depth': int(round(max_depth)),
        'min_data_in_leaf': int(round(min_data_in_leaf)),
        'feature_fraction': feature_fraction,
        'bagging_fraction': bagging_fraction,
        'bagging_freq': int(round(bagging_freq)),
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2,
        'min_gain_to_split': min_gain_to_split,
        'max_bin': int(round(max_bin)),
        'n_estimators': int(round(n_estimators)),
    }
    model = LGBMClassifier(**params, n_jobs=-1, verbosity=-1,)   # verbosity=-1 : verbose 옵션 끔

    model.fit(x_train, y_train, 
              eval_set = [(x_test, y_test)],
              callbacks=[lgbm.early_stopping(10)],   # lightbm은 여기서 early_stopping 옵션 지정
              )
    y_pred = model.predict(x_test)
    print('y_pred:', y_pred)
    # 베이지안에서는 훈련에 사용할 손실함수를 직접 작성한다.(여기선 r2 또는 accuracy 사용)
    acc = accuracy_score(y_test, y_pred)
    print('acc:', acc)
    return acc
    
optimizer = BayesianOptimization(
    f=lgb,   # 최대값을 뽑아야하기때문에 그냥씀. 최소값을 뽑아야하면 -xgb_cv해야함 (ex : -(0.001)과 -(0.1)을 비교하면 -(0.001)가 크기때문에 0.001에 해당하는 파라미터를 반환함)
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
# {'target': 0.8270038784744667, 'params': {'bagging_fraction': 0.6298818382047089, 'bagging_freq': 4.4654420446886816, 'feature_fraction': 1.0, 'lambda_l1': 0.23356367778199771, 'lambda_l2': 1.54069146920613, 'learning_rate': 0.05634370513066924, 'max_bin': 470.01615725540455, 'max_depth': 10.588947447845978, 'min_data_in_leaf': 81.18372101875049, 'min_gain_to_split': 0.007360119224571336, 'n_estimators': 422.8779855756035, 'num_leaves': 142.28758306214962}}
print(n_iter, '번 걸린 시간 :', round(end - start), '초')
# 100 번 걸린 시간 : 690 초
# -> xgb보다 약간빠르고 성능도 좀 더 좋다.