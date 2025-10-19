### <<37>>

from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
import random
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder

# pandas 컬럼명 불일치경고 무시 (x가 pd.dataframe일때 사용)
import warnings
warnings.filterwarnings('ignore', message='X has feature names, but SelectFromModel was fitted without feature names')

seed = 123
random.seed(seed)
np.random.seed(seed)
metric_name = 'logloss'
verbose = 0

# 1. 데이터
path = './_data/kaggle/bank/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

# 인코딩
le_geo = LabelEncoder()
le_gen = LabelEncoder()
train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])
train_csv['Gender'] = le_gen.fit_transform(train_csv['Gender'])
test_csv['Geography'] = le_geo.transform(test_csv['Geography'])
test_csv['Gender'] = le_gen.transform(test_csv['Gender'])

# 불필요 컬럼 제거
train_csv = train_csv.drop(["CustomerId", "Surname"], axis=1)
test_csv = test_csv.drop(["CustomerId", "Surname"], axis=1)

x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    stratify=y,
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

model = XGBClassifier(
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
print('accuracy_score :', model.score(x_test, y_test))     

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
    
    select_model = XGBClassifier(
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
    score = accuracy_score(y_test, select_y_pred)
    print('Trech: %.3f, n=%d, acc : %.4f%%' %(i, select_x_train.shape[1], score*100))
    # print("\n\n")
print(f"verbose : {verbose}")

"""
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ 그냥 모델훈련 시작 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ 그냥 모델훈련 끝 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ


========= XGBClassifier ========
accuracy_score : 0.8624231223679826
[0.02420698 0.06562234 0.06320192 0.1282076  0.02197078 0.04007166
 0.43934998 0.02435699 0.1670902  0.0259215 ]
[0.02197078 0.02420698 0.02435699 0.0259215  0.04007166 0.06320192
 0.06562234 0.1282076  0.1670902  0.43934998]


Trech: 0.022, n=10, acc : 86.2423%
Trech: 0.024, n=9, acc : 86.2878%
Trech: 0.024, n=8, acc : 86.2393%
Trech: 0.026, n=7, acc : 86.3726%
Trech: 0.040, n=6, acc : 86.1514%
Trech: 0.063, n=5, acc : 85.6061%
Trech: 0.066, n=4, acc : 85.4243%
Trech: 0.128, n=3, acc : 84.9880%
Trech: 0.167, n=2, acc : 80.3314%
Trech: 0.439, n=1, acc : 80.3314%
verbose : 0
"""
    