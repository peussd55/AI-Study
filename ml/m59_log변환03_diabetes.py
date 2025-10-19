### <<39>>

# 49-3 카피

from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
import random
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, r2_score

seed = 123
random.seed(seed)
np.random.seed(seed)
metric_name = 'rmse'
verbose = 0

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (150, 4) (150,)

# 로그변환
case = 3
if (case == 1):     # y로그변환
    y = np.log1p(y)
elif (case == 2):   # x로그변환
    x = np.log1p(x)
elif (case == 3):
    y = np.log1p(y)
    x = np.log1p(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    # stratify=y,
)

# 2. 모델
es = xgb.callback.EarlyStopping(
    rounds = 50,
    metric_name = metric_name,  # 회귀 : rmse, rme, rmsle
                                # 다중분류 : mloglos, merror
                                # 이진분류 : logloss, error

    data_name = 'validation_0', # fit에서 eval_set 몇번째 인덱스로 검증할건지 옵션
    # save_best = True,         # 2.x 버전에서 deprecated
)

model = XGBRegressor(
                    n_estimators = 500,
                    max_depth = 6,
                    gamma = 0,
                    min_child_weight = 0,
                    subsample = 0.4,
                    reg_alpha = 0,
                    reg_lambda = 1,
                    random_state=seed,                      
                    
                    eval_metric = metric_name,  # 회귀 : rmse, rme, rmsle
                                                # 다중분류 : mloglos, merror
                                                # 이진분류 : logloss, error
                                                # 2.1.1버전 이후로 사용하는 위치가 fit에서 model로 위치이동
                    
                    callbacks = [es],
                    )
print(f"ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ 그냥 모델훈련 시작 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
model.fit(x_train, y_train,
          eval_set = [(x_test, y_test)],
          verbose = verbose,
          )
print(f"ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ 그냥 모델훈련 끝 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
print("\n")

print("=========", model.__class__.__name__, "========")
print('r2_score :', model.score(x_test, y_test))     

print(model.feature_importances_)   

threshold = np.sort(model.feature_importances_) # default : 오름차순
print(threshold)    
print("\n")

from sklearn.feature_selection import SelectFromModel

for i in threshold:
    # [(중요) 독립적인 훈련을 위해 반복문내에서 es 객체 초기화]
    # 여기서 early stop 객체 정의 안하고 반복문 바깥에서 정의하면 훈련정보(갱신된 logloss, rounds 등)가 누적되기때문에 독립적인 훈련 불가능
    es2 = xgb.callback.EarlyStopping(
    rounds = 50,
    metric_name = metric_name,  # 회귀 : rmse, rme, rmsle
                                # 다중분류 : mloglos, merror
                                # 이진분류 : logloss, error

    data_name = 'validation_0', # fit에서 eval_set 몇번째 인덱스로 검증할건지 옵션
    # save_best = True,         # 2.x 버전에서 deprecated
    )
    
    selection = SelectFromModel(model, threshold=i, prefit=False)
    # threshold가 i값 이상인것을 모두 훈련시킨다.
    # prefit = False : 모델이 아직 학습되지 않았을때, fit 호출에서 훈련한다. (기본값)
    # prefit = True : 이미 학습된 모델을 전달할때, fit 호출하지 않음
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    # print(select_x_train.shape)
    
    select_model = XGBRegressor(
        n_estimators = 500,
        max_depth = 6,
        gamma = 0,
        min_child_weight = 0,
        subsample = 0.4,
        reg_alpha = 0,
        reg_lambda = 1,
        random_state=seed,                      
        
        eval_metric = metric_name,  # 회귀 : rmse, rme, rmsle
                                    # 다중분류 : mloglos, merror (m은 multi의 약자) 
                                    # 이진분류 : logloss, error
                                    # 2.1.1버전 이후로 사용하는 위치가 fit에서 model로 위치이동
        
        callbacks = [es2],
    )
    
    # x_train, x_test -> select_x_train, select_x_test
    # print(f"ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ threshold : {i} 훈련 시작 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
    select_model.fit(select_x_train, y_train,
        eval_set=[(select_x_test, y_test)],
        verbose = verbose,
    )
    # print(f"ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ threshold : {i} 훈련 끝 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")

    select_y_pred = select_model.predict(select_x_test)
    
    score = r2_score(y_test, select_y_pred)
    print('Trech: %.3f, n=%d, r2_score: %.4f%%' %(i, select_x_train.shape[1], score*100))
    # print("\n\n")
print(f"verbose : {verbose}")

"""
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ 그냥 모델훈련 시작 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ 그냥 모델훈련 끝 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ


========= XGBRegressor ========
r2_score : 0.46896612405522065
[0.04560075 0.04596306 0.13227104 0.11751472 0.08156345 0.09924012
 0.07758132 0.09443878 0.21943946 0.08638725]
[0.04560075 0.04596306 0.07758132 0.08156345 0.08638725 0.09443878
 0.09924012 0.11751472 0.13227104 0.21943946]


Trech: 0.046, n=10, r2_score: 46.8966%
Trech: 0.046, n=9, r2_score: 41.7855%
Trech: 0.078, n=8, r2_score: 50.3336%
Trech: 0.082, n=7, r2_score: 44.9602%
Trech: 0.086, n=6, r2_score: 48.1201%
Trech: 0.094, n=5, r2_score: 48.9285%
Trech: 0.099, n=4, r2_score: 46.2208%
Trech: 0.118, n=3, r2_score: 52.1255%
Trech: 0.132, n=2, r2_score: 51.2895%
Trech: 0.219, n=1, r2_score: 31.8392%
verbose : 0

[y로그변환]
Trech: 0.044, n=10, r2_score: 41.2333%
Trech: 0.061, n=9, r2_score: 45.4821%
Trech: 0.078, n=8, r2_score: 44.2046%
Trech: 0.093, n=7, r2_score: 39.5420%
Trech: 0.097, n=6, r2_score: 28.2736%
Trech: 0.102, n=5, r2_score: 40.9016%
Trech: 0.104, n=4, r2_score: 38.6321%
Trech: 0.106, n=3, r2_score: 50.6941%
Trech: 0.113, n=2, r2_score: 47.1551%
Trech: 0.203, n=1, r2_score: 26.2574%
verbose : 0

[x로그변환]
Trech: 0.044, n=10, r2_score: 41.2333%
Trech: 0.061, n=9, r2_score: 45.4821%
Trech: 0.078, n=8, r2_score: 44.2046%
Trech: 0.093, n=7, r2_score: 39.5420%
Trech: 0.097, n=6, r2_score: 28.2736%
Trech: 0.102, n=5, r2_score: 40.9016%
Trech: 0.104, n=4, r2_score: 38.6321%
Trech: 0.106, n=3, r2_score: 50.6941%
Trech: 0.113, n=2, r2_score: 47.1551%
Trech: 0.203, n=1, r2_score: 26.2574%
verbose : 0

[x+y로그변환]
Trech: 0.044, n=10, r2_score: 41.2333%
Trech: 0.061, n=9, r2_score: 45.4821%
Trech: 0.078, n=8, r2_score: 44.2046%
Trech: 0.093, n=7, r2_score: 39.5420%
Trech: 0.097, n=6, r2_score: 28.2736%
Trech: 0.102, n=5, r2_score: 40.9016%
Trech: 0.104, n=4, r2_score: 38.6321%
Trech: 0.106, n=3, r2_score: 50.6941%
Trech: 0.113, n=2, r2_score: 47.1551%
Trech: 0.203, n=1, r2_score: 26.2574%
verbose : 0
"""
    