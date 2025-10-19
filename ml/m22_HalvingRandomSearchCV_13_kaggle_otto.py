### <<33>>

# RandomizedSearchCV : GridSearch랑 반대로 랜덤으로 몇개만 연산

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, StratifiedKFold
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
from xgboost import XGBClassifier, XGBRFRegressor
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from sklearn.experimental import enable_halving_search_cv  # 이 줄은 꼭 필요!
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import LabelEncoder
"""
[GridSearchCV]
    모든 파라미터 조합에 대해 교차검증을 수행해,
    가장 성능이 좋은 하이퍼파라미터를 찾는 방법입니다.

[HalvingGridSearchCV]
    모든 파라미터 조합을 처음에는 **적은 자원(예: 일부 데이터, 적은 반복수 등)**으로 빠르게 평가합니다.
    성능이 좋은 일부 후보만 남기고, 남은 후보에 더 많은 자원을 투입해 다시 평가합니다.
    이 과정을 후보가 1개 남을 때까지 반복합니다.
    **자원(resource)**은 보통 "훈련 데이터 샘플 수"이지만, 트리 개수 등으로 바꿀 수도 있습니다
"""

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

# 처음 학습때 factor만큼 나뉘어서 학습하기때문에 데이터가 작으면작을수록 성능이 떨어진다.
# model = HalvingGridSearchCV(xgb, param_grid=parameters, cv=kfold, 
#                      verbose = 1,
#                      refit=True,
#                      n_jobs=-1,
#                     #  n_iters,
                    
#                      random_state=1111,
#                      factor = 3,          # default : 3
#                     #  min_resources=30,  # deafult 내부적으로 정해짐
#                      )  

model = HalvingRandomSearchCV(xgb, param_distributions=parameters, cv=kfold, 
                     verbose = 1,
                     refit=True,
                     n_jobs=-1,
                    #  n_iters,
                    
                     random_state=1111,
                     factor = 3,            # default : 3
                    #  min_resources=30,  # deafult 내부적으로 정해짐
                    )
"""
[param_grid와 데이터의 갯수에 의해서 계산되고 정해짐]
-> factor가 늘어날수록 : 후보(파라미터)가 많이 사라지고 샘플이 늘어난다.

n_iterations: 4
n_required_iterations: 4
n_possible_iterations: 6
min_resources_: 90
max_resources_: 49502
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 42
n_resources: 90
Fitting 5 folds for each of 42 candidates, totalling 210 fits
----------
iter: 1
n_candidates: 14
n_resources: 270
Fitting 5 folds for each of 14 candidates, totalling 70 fits
----------
iter: 2
n_candidates: 5
n_resources: 810
Fitting 5 folds for each of 5 candidates, totalling 25 fits
----------
iter: 3
n_candidates: 2
n_resources: 2430
Fitting 5 folds for each of 2 candidates, totalling 10 fits
"""
# 3. 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()

print('최적의 매개변수 :', model.best_estimator_)
# 최적의 매개변수 : XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=None, colsample_bynode=None,
#               colsample_bytree=None, device=None, early_stopping_rounds=None, 
#               enable_categorical=False, eval_metric=None, feature_types=None, 
#               gamma=None, grow_policy=None, importance_type=None,
#               interaction_constraints=None, learning_rate=0.1, max_bin=None,  
#               max_cat_threshold=None, max_cat_to_onehot=None,
#               max_delta_step=None, max_depth=6, max_leaves=None,
#               min_child_weight=None, missing=nan, monotone_constraints=None,  
#               multi_strategy=None, n_estimators=500, n_jobs=None,
#               num_parallel_tree=None, objective='multi:softprob', ...) 

print('최적의 파라미터 :', model.best_params_) 

# 4. 평가, 예측
print('best_score :', model.best_score_)                # (x_train)에서의 교차검증 평균 성능

print('model.score :', model.score(x_test, y_test))     # 실제 테스트 데이터에서의 모델 성능

y_pred = model.predict(x_test)
print('accuracy_score :', accuracy_score(y_test, y_pred))

y_pred_best = model.best_estimator_.predict(x_test)     # refit=True 이면 predict나 best_estimator_.predict나 차이없다 : 이미 refit=True 옵션으로 최적의 가중치를 찾아놨기때문.
print('accuracy_score :', accuracy_score(y_test, y_pred_best))

print('걸린시간 :', round(end - start), '초')
"""
최적의 파라미터 : {'max_depth': 6, 'learning_rate': 0.1}
best_score : 0.7342522591319842
model.score : 0.7969457013574661
accuracy_score : 0.7969457013574661
accuracy_score : 0.7969457013574661

-> GrdisSearch보다 성능이 전체적으로 떨어짐
"""

import pandas as pd
# print(pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True))   # rank_test_score를 기준으로 오름차순 정렬
print(pd.DataFrame(model.cv_results_).columns)

path = './_save/m22_cv_results/'
pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True).to_csv(path + 'm22_hrs_cv_results_13_kaggle_otto.csv')

# 5. 가중치 저장
path = './_save/m22_cv_results/'
joblib.dump(model.best_estimator_, path + 'm22_best_model_13_kaggle_otto.joblib')