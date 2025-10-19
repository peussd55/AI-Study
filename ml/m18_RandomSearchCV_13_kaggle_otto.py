### <<33>>

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import  LinearSVC, SVC
import time
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.utils import  all_estimators
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
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
xgb = XGBClassifier()
# model = GridSearchCV(xgb, param_grid=parameters, cv=kfold, 
#                      verbose = 1,
#                      refit=True,    # best 파라미터로 전체 훈련 데이터(x_train, y_train)를 다시 fit 1번 (디폴트). 아래에 best_가 들어간 모델 옵션은 refit=True일때만 쓸 수있음
#                      n_jobs=-1,     # cpu를 풀 가동(모든 쓰레드를 가동)
                     
#                      )  # 12+18+12+1

model = RandomizedSearchCV(xgb, param_distributions=parameters, cv=kfold, 
                     verbose = 1,
                     refit=True,
                     n_jobs=-1,
                     random_state=1111,
                     n_iter=10,
                     )  

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
print('r2_score :', accuracy_score(y_test, y_pred))

print('걸린시간 :', round(end - start), '초')

"""
최적의 파라미터 : {'n_estimators': 500, 'max_depth': 10, 'learning_rate': 0.1}
best_score : 0.820936569002825
model.score : 0.8228021978021978
r2_score : 0.8228021978021978
걸린시간 : 279 초
"""

import pandas as pd
print(pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True))   # rank_test_score를 기준으로 오름차순 정렬
print(pd.DataFrame(model.cv_results_).columns)

path = './_save/m18_cv_results/'
pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True).to_csv(path + 'm18_ rs_cv_results_13_kaggle_otto.csv')

# 5. 가중치 저장
path = './_save/m18_cv_results/'
joblib.dump(model.best_estimator_, path + 'm18_best_model_12_kaggle_otto.joblib')

