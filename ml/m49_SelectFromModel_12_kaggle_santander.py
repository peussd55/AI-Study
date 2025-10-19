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
metric_name = 'logloss'
verbose = 0

# 1. 데이터
path = './_data/kaggle/santander/'           
# 맨 앞, 공통 컬럼 datetime을 인덱스 컬럼으로 지정
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')
print(train_csv.shape, test_csv.shape, submission_csv.shape)  # (200000, 202) (200000, 201) (200000, 2)
print(train_csv['target'].value_counts())
# 이진분류 (불균형)
# 0    179902
# 1     20098
# train data x로 y 분리
x = train_csv.drop(['target'], axis=1)
print(x)
y = train_csv['target']

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
accuracy_score : 0.912225
[0.00373289 0.00429133 0.00491533 0.00225471 0.00231474 0.00367595
 0.00680252 0.00320725 0.00345471 0.00587689 0.00266543 0.00377157
 0.00907471 0.00827626 0.00314075 0.00347976 0.00302166 0.00378008
 0.00537152 0.00314922 0.00402897 0.00725369 0.00752968 0.00420123
 0.00468836 0.00380216 0.00905321 0.00405762 0.00437138 0.00348767
 0.00330676 0.00380736 0.00591532 0.00628657 0.00792018 0.00525636
 0.00537947 0.00330366 0.00320297 0.00376419 0.00653325 0.00305821
 0.00330558 0.00517992 0.00684289 0.0041624  0.00408284 0.00303446
 0.00513306 0.00497054 0.00371013 0.00525974 0.00448327 0.00817768
 0.00347689 0.00429071 0.00537123 0.00407967 0.00390307 0.00330644
 0.0039376  0.00366042 0.003803   0.00396245 0.00428996 0.00280928
 0.00463757 0.00623312 0.00358846 0.00354423 0.00455502 0.00505815
 0.00392434 0.00362083 0.00388591 0.00636612 0.00754683 0.00429686
 0.00664449 0.00378305 0.00834143 0.01175228 0.00517247 0.00418398
 0.00370119 0.00428203 0.00623297 0.0049064  0.00429419 0.00564689
 0.00486158 0.00571809 0.0061748  0.00535013 0.00746104 0.00602319
 0.00378931 0.00402936 0.00330833 0.00761658 0.00322173 0.00399236
 0.00390074 0.00376071 0.00467977 0.00429337 0.00542425 0.00563158
 0.00651437 0.00715655 0.00909404 0.00496533 0.0045073  0.00294377
 0.00504708 0.00653443 0.00452522 0.00363336 0.00509944 0.00532655
 0.0041019  0.00584357 0.00553612 0.00547883 0.00318752 0.00495276
 0.00281692 0.0068756  0.00492147 0.00379576 0.00509153 0.00542459
 0.00467498 0.00753941 0.00418924 0.00453583 0.00393769 0.00434625
 0.00400617 0.01015292 0.00382696 0.00540267 0.00412317 0.00366339
 0.00379394 0.00508952 0.00858807 0.00582908 0.00733965 0.0060285
 0.0050472  0.00470515 0.00372907 0.00379213 0.0062265  0.00602982
 0.004056   0.00513433 0.00386819 0.00343113 0.00387967 0.00424906
 0.00503605 0.00472962 0.00700103 0.00798705 0.00787516 0.00576339
 0.00398356 0.00599206 0.00636645 0.00459201 0.00489647 0.00546199
 0.00875731 0.00420992 0.00390614 0.00671852 0.00416125 0.00722704
 0.00526538 0.00339337 0.00419594 0.00357469 0.00639178 0.00393698
 0.00519457 0.0045767  0.00602571 0.00360075 0.00706818 0.00744251
 0.00617825 0.00408302 0.00437926 0.00547967 0.0046666  0.00523279
 0.00843421 0.00449047]
[0.00225471 0.00231474 0.00266543 0.00280928 0.00281692 0.00294377
 0.00302166 0.00303446 0.00305821 0.00314075 0.00314922 0.00318752
 0.00320297 0.00320725 0.00322173 0.00330366 0.00330558 0.00330644
 0.00330676 0.00330833 0.00339337 0.00343113 0.00345471 0.00347689
 0.00347976 0.00348767 0.00354423 0.00357469 0.00358846 0.00360075
 0.00362083 0.00363336 0.00366042 0.00366339 0.00367595 0.00370119
 0.00371013 0.00372907 0.00373289 0.00376071 0.00376419 0.00377157
 0.00378008 0.00378305 0.00378931 0.00379213 0.00379394 0.00379576
 0.00380216 0.003803   0.00380736 0.00382696 0.00386819 0.00387967
 0.00388591 0.00390074 0.00390307 0.00390614 0.00392434 0.00393698
 0.0039376  0.00393769 0.00396245 0.00398356 0.00399236 0.00400617
 0.00402897 0.00402936 0.004056   0.00405762 0.00407967 0.00408284
 0.00408302 0.0041019  0.00412317 0.00416125 0.0041624  0.00418398
 0.00418924 0.00419594 0.00420123 0.00420992 0.00424906 0.00428203
 0.00428996 0.00429071 0.00429133 0.00429337 0.00429419 0.00429686
 0.00434625 0.00437138 0.00437926 0.00448327 0.00449047 0.0045073
 0.00452522 0.00453583 0.00455502 0.0045767  0.00459201 0.00463757
 0.0046666  0.00467498 0.00467977 0.00468836 0.00470515 0.00472962
 0.00486158 0.00489647 0.0049064  0.00491533 0.00492147 0.00495276
 0.00496533 0.00497054 0.00503605 0.00504708 0.0050472  0.00505815
 0.00508952 0.00509153 0.00509944 0.00513306 0.00513433 0.00517247
 0.00517992 0.00519457 0.00523279 0.00525636 0.00525974 0.00526538
 0.00532655 0.00535013 0.00537123 0.00537152 0.00537947 0.00540267
 0.00542425 0.00542459 0.00546199 0.00547883 0.00547967 0.00553612
 0.00563158 0.00564689 0.00571809 0.00576339 0.00582908 0.00584357
 0.00587689 0.00591532 0.00599206 0.00602319 0.00602571 0.0060285
 0.00602982 0.0061748  0.00617825 0.0062265  0.00623297 0.00623312
 0.00628657 0.00636612 0.00636645 0.00639178 0.00651437 0.00653325
 0.00653443 0.00664449 0.00671852 0.00680252 0.00684289 0.0068756
 0.00700103 0.00706818 0.00715655 0.00722704 0.00725369 0.00733965
 0.00744251 0.00746104 0.00752968 0.00753941 0.00754683 0.00761658
 0.00787516 0.00792018 0.00798705 0.00817768 0.00827626 0.00834143
 0.00843421 0.00858807 0.00875731 0.00905321 0.00907471 0.00909404
 0.01015292 0.01175228]


Trech: 0.002, n=200, acc : 91.2225%
Trech: 0.002, n=199, acc : 91.0475%
...
...
"""
    