### <<33>>

# RandomizedSearchCV : GridSearch랑 반대로 랜덤으로 몇개만 연산

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import  LinearSVC, SVC
import time
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.utils import  all_estimators
from xgboost import XGBClassifier
import joblib

from sklearn.experimental import enable_halving_search_cv  # 이 줄은 꼭 필요!
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV
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
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8,
    stratify=y
)

n_split = 5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333,)

parameters = [
    {'n_estimators':[100,500], 'max_depth':[6,10,12], 'learning_rate':[0.1, 0.01, 0.001]},  # 18
    {'max_depth':[6,8,10,12], 'learning_rate':[0.1, 0.01, 0.001]},  # 12
    {'min_child_weight':[2,3,5,10], 'learning_rate':[0.1, 0.01, 0.001]},    # 12
]   # 18+12+12=42

# 2. 모델
xgb = XGBClassifier()
# model = GridSearchCV(xgb, param_grid=parameters, cv=kfold, 
#                      verbose = 1,
#                      refit=True,    # best 파라미터로 전체 훈련 데이터(x_train, y_train)를 다시 fit 1번 (디폴트). 아래에 best_가 들어간 모델 옵션은 refit=True일때만 쓸 수있음
#                      n_jobs=-1,     # cpu를 풀 가동(모든 쓰레드를 가동)
                     
#                      )  # 12+18+12+1

# model = HalvingGridSearchCV(xgb, param_grid=parameters, cv=kfold, 
#                      verbose = 1,
#                      refit=True,
#                      n_jobs=-1,
#                     #  n_iters,
                    
#                      random_state=1111,
#                      factor = 3,        # 배율 : min_resources * factor
#                      min_resources=30,  # deafult 내부적으로 정해짐
#                      )  

model = HalvingRandomSearchCV(xgb, param_distributions=parameters, cv=kfold, 
                     verbose = 1,
                     refit=True,
                     n_jobs=-1,
                    #  n_iter=10,
                    
                     random_state=1111,
                     factor = 3,        # 배율 : min_resources * factor
                     min_resources=30,  # deafult 내부적으로 정해짐
                     )  
"""
[param_grid와 데이터의 갯수에 의해서 계산되고 정해짐]

n_iterations: 2
n_required_iterations: 2
n_possible_iterations: 2
min_resources_: 30
max_resources_: 120
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 4
n_resources: 30
Fitting 5 folds for each of 4 candidates, totalling 20 fits
----------
iter: 1
n_candidates: 2
n_resources: 90
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
# 최적의 파라미터 :  {'learning_rate': 0.01, 'max_depth': 12, 'n_estimators': 500}

# 4. 평가, 예측
print('best_score :', model.best_score_)                # (x_train)에서의 교차검증 평균 성능
# best_score : 0.95

print('model.score :', model.score(x_test, y_test))     # 실제 테스트 데이터에서의 모델 성능
# model.score : 0.9

y_pred = model.predict(x_test)
print('accuracy_score :', accuracy_score(y_test, y_pred))
# accuracy_score : 0.9

y_pred_best = model.best_estimator_.predict(x_test)     # refit=True 이면 predict나 best_estimator_.predict나 차이없다 : 이미 refit=True 옵션으로 최적의 가중치를 찾아놨기때문.
print('best_acc_score :', accuracy_score(y_test, y_pred_best))
# best_acc_socre : 0.9

print('걸린시간 :', round(end - start), '초')
# 걸린시간 : 3 초

import pandas as pd
# print(pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True))   # rank_test_score를 기준으로 오름차순 정렬
print(pd.DataFrame(model.cv_results_).columns)
# Index(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
#        'param_learning_rate', 'param_max_depth', 'param_n_estimators',
#        'param_min_child_weight', 'params', 'split0_test_score',
#        'split1_test_score', 'split2_test_score', 'split3_test_score',
#        'split4_test_score', 'mean_test_score', 'std_test_score',
#        'rank_test_score'],

path = './_save/m22_cv_results/'

pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True).to_csv(path + 'm22_hrs_cv_results.csv')

# 5. 가중치 저장
path = './_save/m22_cv_results/'
joblib.dump(model.best_estimator_, path + 'm22_best_model.joblib')

