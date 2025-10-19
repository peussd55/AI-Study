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
accuracy_score : 0.8160148674854557
[0.0043807  0.00245042 0.00342028 0.00436065 0.00347633 0.00484129
 0.00665738 0.00613669 0.01223982 0.00459142 0.06333242 0.00417389
 0.00509889 0.01640549 0.02122745 0.00664215 0.00960386 0.00494861
 0.00820389 0.0057204  0.00609215 0.00485951 0.00822412 0.00744031
 0.0103373  0.01940538 0.00615366 0.00485261 0.00664199 0.01757971
 0.00510621 0.00669862 0.00576865 0.0216581  0.01127759 0.01931313
 0.00636506 0.00595942 0.01931522 0.01536755 0.00825903 0.01648488
 0.01253795 0.00549813 0.00887991 0.00529729 0.01666651 0.00801248
 0.00576856 0.01144089 0.00718603 0.0063264  0.01463159 0.00658103
 0.008253   0.0112902  0.01096795 0.00955549 0.01506291 0.04394242
 0.00594618 0.01126703 0.00609037 0.0085284  0.00562189 0.00665021
 0.00986367 0.01837996 0.0231967  0.00646252 0.00737629 0.01068057
 0.00609127 0.00586025 0.01813152 0.00821429 0.00894297 0.01516348
 0.00757443 0.00754051 0.00695019 0.00409312 0.00792977 0.01721026
 0.00821736 0.01287139 0.00613536 0.00951714 0.00582245 0.0554974
 0.01325317 0.0083036  0.00764529]
[0.00245042 0.00342028 0.00347633 0.00409312 0.00417389 0.00436065
 0.0043807  0.00459142 0.00484129 0.00485261 0.00485951 0.00494861
 0.00509889 0.00510621 0.00529729 0.00549813 0.00562189 0.0057204
 0.00576856 0.00576865 0.00582245 0.00586025 0.00594618 0.00595942
 0.00609037 0.00609127 0.00609215 0.00613536 0.00613669 0.00615366
 0.0063264  0.00636506 0.00646252 0.00658103 0.00664199 0.00664215
 0.00665021 0.00665738 0.00669862 0.00695019 0.00718603 0.00737629
 0.00744031 0.00754051 0.00757443 0.00764529 0.00792977 0.00801248
 0.00820389 0.00821429 0.00821736 0.00822412 0.008253   0.00825903
 0.0083036  0.0085284  0.00887991 0.00894297 0.00951714 0.00955549
 0.00960386 0.00986367 0.0103373  0.01068057 0.01096795 0.01126703
 0.01127759 0.0112902  0.01144089 0.01223982 0.01253795 0.01287139
 0.01325317 0.01463159 0.01506291 0.01516348 0.01536755 0.01640549
 0.01648488 0.01666651 0.01721026 0.01757971 0.01813152 0.01837996
 0.01931313 0.01931522 0.01940538 0.02122745 0.0216581  0.0231967
 0.04394242 0.0554974  0.06333242]


Trech: 0.002, n=93, acc : 81.6015%
Trech: 0.003, n=92, acc : 81.5207%
Trech: 0.003, n=91, acc : 81.3672%
Trech: 0.004, n=90, acc : 81.0359%
...
...
"""
    