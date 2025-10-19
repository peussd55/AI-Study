### <<39>>

# 49-4 카피

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

# pandas 컬럼명 불일치경고 무시 (x가 pd.dataframe일때 사용)
import warnings
warnings.filterwarnings('ignore', message='X has feature names, but SelectFromModel was fitted without feature names')

seed = 123
random.seed(seed)
np.random.seed(seed)
metric_name = 'rmse'
verbose = 0

# 1. 데이터
path = './_data/dacon/따릉이/'          # 시스템 경로에서 시작.

train_csv =  pd.read_csv(path + 'train.csv', index_col=0)     # 0번컬럼을 인덱스컬럼으로 지정
test_csv = pd.read_csv(path + 'test.csv', index_col=0)  
submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)

# 결측치 처리
train_csv = train_csv.fillna(train_csv.mean())
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['count'], axis=1)   # count 컬럼 제거
y = train_csv['count']                  # 타겟 변수

# 로그변환
case = 3
if (case == 1):     # y로그변환
    y = np.log1p(y)
elif (case == 2):   # x로그변환
    x = np.log1p(x)
elif (case == 3):   # 둘 다변환
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
r2_score : 0.7291577502727749
[0.18228628 0.16232552 0.29978216 0.04990303 0.05100336 0.0768147
 0.06528616 0.05491458 0.05768418]
[0.04990303 0.05100336 0.05491458 0.05768418 0.06528616 0.0768147
 0.16232552 0.18228628 0.29978216]

[그냥]
Trech: 0.050, n=9, r2_score: 72.9158%
Trech: 0.051, n=8, r2_score: 73.0448%
Trech: 0.055, n=7, r2_score: 73.9686%
Trech: 0.058, n=6, r2_score: 73.8420%
Trech: 0.065, n=5, r2_score: 74.8711%
Trech: 0.077, n=4, r2_score: 70.3576%
Trech: 0.162, n=3, r2_score: 74.2225%
Trech: 0.182, n=2, r2_score: 63.4593%
Trech: 0.300, n=1, r2_score: 0.4227%
verbose : 0

[y로그변환]
Trech: 0.036, n=9, r2_score: 76.8355%
Trech: 0.039, n=8, r2_score: 75.0716%
Trech: 0.041, n=7, r2_score: 78.4274%
Trech: 0.041, n=6, r2_score: 76.5252%
Trech: 0.045, n=5, r2_score: 73.0189%
Trech: 0.052, n=4, r2_score: 67.4690%
Trech: 0.053, n=3, r2_score: 64.6012%
Trech: 0.185, n=2, r2_score: 70.3551%
Trech: 0.508, n=1, r2_score: 2.8383%

[x로그변환]
Trech: 0.050, n=9, r2_score: 72.9158%
Trech: 0.051, n=8, r2_score: 73.0448%
Trech: 0.055, n=7, r2_score: 73.9686%
Trech: 0.058, n=6, r2_score: 73.8420%
Trech: 0.065, n=5, r2_score: 74.8711%
Trech: 0.077, n=4, r2_score: 70.3576%
Trech: 0.162, n=3, r2_score: 74.2225%
Trech: 0.182, n=2, r2_score: 63.4593%
Trech: 0.300, n=1, r2_score: 0.4227%

[x,y 로그변환]
Trech: 0.036, n=9, r2_score: 76.8355%
Trech: 0.039, n=8, r2_score: 75.0716%
Trech: 0.041, n=7, r2_score: 78.4274%
Trech: 0.041, n=6, r2_score: 76.5252%
Trech: 0.045, n=5, r2_score: 73.0189%
Trech: 0.052, n=4, r2_score: 67.4690%
Trech: 0.053, n=3, r2_score: 64.6012%
Trech: 0.185, n=2, r2_score: 70.3551%
Trech: 0.508, n=1, r2_score: 2.8383%

"""
    