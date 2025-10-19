### <<37>>

from sklearn.datasets import load_digits
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
metric_name = 'mlogloss'
verbose = 0

# 1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)

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
accuracy_score : 0.9833333333333333
[0.00000000e+00 9.97454673e-03 1.26001276e-02 8.95390380e-03
 7.35879922e-03 3.45711224e-02 8.32405873e-03 5.18104248e-03
 5.54371800e-05 5.24413167e-03 1.46660786e-02 2.46699899e-03
 1.95412971e-02 9.78769083e-03 8.65448825e-03 4.29555104e-04
 2.07703797e-05 5.94928116e-03 1.13846855e-02 3.91702056e-02
 1.70520227e-02 5.65190502e-02 4.98131616e-03 3.09662792e-05
 4.52939561e-03 2.47764797e-03 3.84011827e-02 2.15260237e-02
 2.52425782e-02 2.42144540e-02 1.15897376e-02 3.00653373e-05
 0.00000000e+00 4.70032766e-02 1.41975200e-02 8.54414981e-03
 5.83286621e-02 1.61266681e-02 2.71992944e-02 0.00000000e+00
 8.93361005e-07 6.04125531e-03 5.04409485e-02 5.84130809e-02
 1.64529923e-02 1.59212034e-02 2.06985753e-02 3.61206097e-04
 7.54009452e-05 3.24740447e-03 1.51243666e-02 7.99043477e-03
 9.26676299e-03 1.49407694e-02 2.44768187e-02 3.52153182e-03
 1.06019334e-05 1.36659399e-03 1.80300567e-02 3.67415138e-03
 6.09417148e-02 3.19967717e-02 2.97099072e-02 2.49683745e-02]
[0.00000000e+00 0.00000000e+00 0.00000000e+00 8.93361005e-07
 1.06019334e-05 2.07703797e-05 3.00653373e-05 3.09662792e-05
 5.54371800e-05 7.54009452e-05 3.61206097e-04 4.29555104e-04
 1.36659399e-03 2.46699899e-03 2.47764797e-03 3.24740447e-03
 3.52153182e-03 3.67415138e-03 4.52939561e-03 4.98131616e-03
 5.18104248e-03 5.24413167e-03 5.94928116e-03 6.04125531e-03
 7.35879922e-03 7.99043477e-03 8.32405873e-03 8.54414981e-03
 8.65448825e-03 8.95390380e-03 9.26676299e-03 9.78769083e-03
 9.97454673e-03 1.13846855e-02 1.15897376e-02 1.26001276e-02
 1.41975200e-02 1.46660786e-02 1.49407694e-02 1.51243666e-02
 1.59212034e-02 1.61266681e-02 1.64529923e-02 1.70520227e-02
 1.80300567e-02 1.95412971e-02 2.06985753e-02 2.15260237e-02
 2.42144540e-02 2.44768187e-02 2.49683745e-02 2.52425782e-02
 2.71992944e-02 2.97099072e-02 3.19967717e-02 3.45711224e-02
 3.84011827e-02 3.91702056e-02 4.70032766e-02 5.04409485e-02
 5.65190502e-02 5.83286621e-02 5.84130809e-02 6.09417148e-02]


Trech: 0.000, n=64, acc : 98.3333%
Trech: 0.000, n=64, acc : 98.3333%
Trech: 0.000, n=64, acc : 98.3333%
Trech: 0.000, n=61, acc : 98.3333%
Trech: 0.000, n=60, acc : 98.3333%
Trech: 0.000, n=59, acc : 98.3333%
Trech: 0.000, n=58, acc : 98.3333%
Trech: 0.000, n=57, acc : 98.3333%
Trech: 0.000, n=56, acc : 98.3333%
Trech: 0.000, n=55, acc : 98.3333%
Trech: 0.000, n=54, acc : 98.3333%
Trech: 0.000, n=53, acc : 98.3333%
Trech: 0.001, n=52, acc : 98.3333%
Trech: 0.002, n=51, acc : 98.0556%
Trech: 0.002, n=50, acc : 98.3333%
Trech: 0.003, n=49, acc : 98.3333%
Trech: 0.004, n=48, acc : 98.3333%
Trech: 0.004, n=47, acc : 98.3333%
Trech: 0.005, n=46, acc : 98.0556%
Trech: 0.005, n=45, acc : 97.7778%
...
...
"""
    