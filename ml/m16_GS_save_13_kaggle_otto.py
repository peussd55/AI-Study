### <<33>>

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import  LinearSVC, SVC
import time
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.utils import  all_estimators
from xgboost import XGBClassifier, XGBRegressor
import xgboost
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder

# GridSearch : 모든 경우의 수를 탐색하는 방법

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
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333, )

parameters = [
    {'n_estimators':[100,500], 'max_depth':[6,10,12], 'learning_rate':[0.1, 0.01, 0.001]},  # 18
    {'max_depth':[6,8,10,12], 'learning_rate':[0.1, 0.01, 0.001]},  # 12
    {'min_child_weight':[2,3,5,10], 'learning_rate':[0.1, 0.01, 0.001]},    # 12
]

# 2. 모델
xgb = XGBClassifier(tree_method='hist',  device='cuda:0')       # xgboost gpu 사용 옵션
model = GridSearchCV(xgb, param_grid=parameters, cv=kfold, 
                     verbose = 1,
                     refit=True,    # best 파라미터로 전체 훈련 데이터(x_train, y_train)를 다시 fit 1번 (디폴트)
                     n_jobs=-1,     # cpu를 풀 가동(모든 쓰레드를 가동)
                     )  # 12+18+12+1

# 3. 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()

print('최적의 매개변수 :', model.best_estimator_)

print('최적의 파라미터 :', model.best_params_)

# 4. 평가, 예측
print('best_score :', model.best_score_)

print('model.score :', model.score(x_test, y_test))

y_pred = model.predict(x_test)
print('accuracy_score :', accuracy_score(y_test, y_pred))

print('걸린시간 :', round(end - start), '초')
"""

"""

# cpu, gpu>cuda 사용량 100%나오는 이유 검색
# GPU가 신경망의 행렬 연산 등 대규모 병렬 처리를 담당하지만,

# CPU는 데이터 전처리, 배치 생성, 데이터 로딩, 모델의 일부 제어 로직

r"""
c:\Users\HI-806\anaconda3\envs\tf274gpu\lib\site-packages\xgboost\core.py:158: UserWarning: [10:41:22] WARNING: C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-08cbc0333d8d4aae1-1\xgboost\xgboost-ci-windows\src\common\error_msg.cc:58: Falling back to prediction using DMatrix due to mismatched devices. This might lead to higher memory usage and slower performance. XGBoost is running on: cuda:0, while the input data 
is on: cpu.
Potential solutions:
- Use a data structure that matches the device ordinal in the booster.
- Set the device for booster before call to inplace_predict.

This warning will only be shown once.
-> 
XGBoost가 GPU(CUDA)로 학습된 모델인데, 예측(predict)할 때 입력 데이터(X)가 CPU 메모리(일반 numpy/pandas 배열)에 있어서, 내부적으로 CPU↔GPU 데이터 이동이 발생하며 성능 저하가 있을 수 있다는 경고
"""

# 5. 가중치 저장
import joblib
path = './_save/m15_cv_results/'
joblib.dump(model.best_estimator_, path + 'm15_best_model_13_kaggle_otto.joblib')
# 모델은 GridSearch로 돌렸지만 저장은 XGBoost로 저장된다.