### <<37>>

# 47-0 카피

# 컬럼 중요도순으로 나열하고 1~n개까지 각각 한번씩 학습돌리는 방법

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score

seed = 123
random.seed(seed)
np.random.seed(seed)
metric_name = 'mlogloss'
verbose = 1

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (150, 4) (150,)

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
          verbose = 1,
          )
print(f"ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ 그냥 모델훈련 끝 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
print("\n")

print("=========", model.__class__.__name__, "========")
print('acc :', model.score(x_test, y_test))     
# acc : 0.9333333333333333

print(model.feature_importances_)               # 부스팅모델(사이킷런기반)에서 중요도를 선정하는 기준 : 빈도수
# [0.07114094 0.12324659 0.49751213 0.3081003 ]

threshold = np.sort(model.feature_importances_) # default : 오름차순
print(threshold)    
# [0.07114094 0.12324659 0.3081003  0.49751213]
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
    # (120, 4)
    # (120, 3)
    # (120, 2)
    # (120, 1)
    
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
    print(f"ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ threshold : {i} 훈련 시작 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
    select_model.fit(select_x_train, y_train,
        eval_set=[(select_x_test, y_test)],
        verbose = 1,
    )
    print(f"ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ threshold : {i} 훈련 끝 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")

    select_y_pred = select_model.predict(select_x_test)
    score = accuracy_score(y_test, select_y_pred)
    print('Trech: %.3f, n=%d, ACC: %.4f%%' %(i, select_x_train.shape[1], score*100))
    # print("\n\n")
print(f"verbose : {verbose}")

"""
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ 그냥 모델훈련 시작 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
[0]     validation_0-mlogloss:0.77627
[1]     validation_0-mlogloss:0.61601
[2]     validation_0-mlogloss:0.47369
[3]     validation_0-mlogloss:0.38918
[4]     validation_0-mlogloss:0.35025
[5]     validation_0-mlogloss:0.32003
[6]     validation_0-mlogloss:0.28476
[7]     validation_0-mlogloss:0.26394
[8]     validation_0-mlogloss:0.25019
[9]     validation_0-mlogloss:0.24792
[10]    validation_0-mlogloss:0.23421
[11]    validation_0-mlogloss:0.23179
[12]    validation_0-mlogloss:0.22704
[13]    validation_0-mlogloss:0.21865
[14]    validation_0-mlogloss:0.22456
[15]    validation_0-mlogloss:0.22782
[16]    validation_0-mlogloss:0.23552
[17]    validation_0-mlogloss:0.23851
[18]    validation_0-mlogloss:0.23927
[19]    validation_0-mlogloss:0.23858
[20]    validation_0-mlogloss:0.23976
[21]    validation_0-mlogloss:0.23224
[22]    validation_0-mlogloss:0.23819
[23]    validation_0-mlogloss:0.23808
[24]    validation_0-mlogloss:0.23852
[25]    validation_0-mlogloss:0.23460
[26]    validation_0-mlogloss:0.24064
[27]    validation_0-mlogloss:0.24503
[28]    validation_0-mlogloss:0.24743
[29]    validation_0-mlogloss:0.24914
[30]    validation_0-mlogloss:0.25477
[31]    validation_0-mlogloss:0.25726
[32]    validation_0-mlogloss:0.25813
[33]    validation_0-mlogloss:0.26539
[34]    validation_0-mlogloss:0.27047
[35]    validation_0-mlogloss:0.27104
[36]    validation_0-mlogloss:0.27158
[37]    validation_0-mlogloss:0.27188
[38]    validation_0-mlogloss:0.26517
[39]    validation_0-mlogloss:0.26982
[40]    validation_0-mlogloss:0.27293
[41]    validation_0-mlogloss:0.27522
[42]    validation_0-mlogloss:0.27557
[43]    validation_0-mlogloss:0.28193
[44]    validation_0-mlogloss:0.28622
[45]    validation_0-mlogloss:0.28734
[46]    validation_0-mlogloss:0.29148
[47]    validation_0-mlogloss:0.28769
[48]    validation_0-mlogloss:0.28827
[49]    validation_0-mlogloss:0.28478
[50]    validation_0-mlogloss:0.28982
[51]    validation_0-mlogloss:0.29352
[52]    validation_0-mlogloss:0.29461
[53]    validation_0-mlogloss:0.29187
[54]    validation_0-mlogloss:0.29681
[55]    validation_0-mlogloss:0.29767
[56]    validation_0-mlogloss:0.30068
[57]    validation_0-mlogloss:0.29958
[58]    validation_0-mlogloss:0.29990
[59]    validation_0-mlogloss:0.30063
[60]    validation_0-mlogloss:0.30516
[61]    validation_0-mlogloss:0.30664
[62]    validation_0-mlogloss:0.31038
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ 그냥 모델훈련 끝 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ


========= XGBClassifier ========
acc : 0.9333333333333333
[0.07114094 0.12324659 0.49751213 0.3081003 ]
[0.07114094 0.12324659 0.3081003  0.49751213]


ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ threshold : 0.07114094495773315 훈련 시작 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
[0]     validation_0-mlogloss:0.77627
[1]     validation_0-mlogloss:0.61601
[2]     validation_0-mlogloss:0.47369
[3]     validation_0-mlogloss:0.38918
[4]     validation_0-mlogloss:0.35025
[5]     validation_0-mlogloss:0.32003
[6]     validation_0-mlogloss:0.28476
[7]     validation_0-mlogloss:0.26394
[8]     validation_0-mlogloss:0.25019
[9]     validation_0-mlogloss:0.24792
[10]    validation_0-mlogloss:0.23421
[11]    validation_0-mlogloss:0.23179
[12]    validation_0-mlogloss:0.22704
[13]    validation_0-mlogloss:0.21865
[14]    validation_0-mlogloss:0.22456
[15]    validation_0-mlogloss:0.22782
[16]    validation_0-mlogloss:0.23552
[17]    validation_0-mlogloss:0.23851
[18]    validation_0-mlogloss:0.23927
[19]    validation_0-mlogloss:0.23858
[20]    validation_0-mlogloss:0.23976
[21]    validation_0-mlogloss:0.23224
[22]    validation_0-mlogloss:0.23819
[23]    validation_0-mlogloss:0.23808
[24]    validation_0-mlogloss:0.23852
[25]    validation_0-mlogloss:0.23460
[26]    validation_0-mlogloss:0.24064
[27]    validation_0-mlogloss:0.24503
[28]    validation_0-mlogloss:0.24743
[29]    validation_0-mlogloss:0.24914
[30]    validation_0-mlogloss:0.25477
[31]    validation_0-mlogloss:0.25726
[32]    validation_0-mlogloss:0.25813
[33]    validation_0-mlogloss:0.26539
[34]    validation_0-mlogloss:0.27047
[35]    validation_0-mlogloss:0.27104
[36]    validation_0-mlogloss:0.27158
[37]    validation_0-mlogloss:0.27188
[38]    validation_0-mlogloss:0.26517
[39]    validation_0-mlogloss:0.26982
[40]    validation_0-mlogloss:0.27293
[41]    validation_0-mlogloss:0.27522
[42]    validation_0-mlogloss:0.27557
[43]    validation_0-mlogloss:0.28193
[44]    validation_0-mlogloss:0.28622
[45]    validation_0-mlogloss:0.28734
[46]    validation_0-mlogloss:0.29148
[47]    validation_0-mlogloss:0.28769
[48]    validation_0-mlogloss:0.28827
[49]    validation_0-mlogloss:0.28478
[50]    validation_0-mlogloss:0.28982
[51]    validation_0-mlogloss:0.29352
[52]    validation_0-mlogloss:0.29461
[53]    validation_0-mlogloss:0.29187
[54]    validation_0-mlogloss:0.29681
[55]    validation_0-mlogloss:0.29767
[56]    validation_0-mlogloss:0.30068
[57]    validation_0-mlogloss:0.29958
[58]    validation_0-mlogloss:0.29990
[59]    validation_0-mlogloss:0.30063
[60]    validation_0-mlogloss:0.30516
[61]    validation_0-mlogloss:0.30664
[62]    validation_0-mlogloss:0.31038
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ threshold : 0.07114094495773315 훈련 끝 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
Trech: 0.071, n=4, ACC: 93.3333%
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ threshold : 0.12324658781290054 훈련 시작 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
[0]     validation_0-mlogloss:0.78051
[1]     validation_0-mlogloss:0.59419
[2]     validation_0-mlogloss:0.45597
[3]     validation_0-mlogloss:0.37187
[4]     validation_0-mlogloss:0.32217
[5]     validation_0-mlogloss:0.29140
[6]     validation_0-mlogloss:0.25783
[7]     validation_0-mlogloss:0.23711
[8]     validation_0-mlogloss:0.22070
[9]     validation_0-mlogloss:0.21765
[10]    validation_0-mlogloss:0.20237
[11]    validation_0-mlogloss:0.19887
[12]    validation_0-mlogloss:0.19498
[13]    validation_0-mlogloss:0.19100
[14]    validation_0-mlogloss:0.18576
[15]    validation_0-mlogloss:0.18913
[16]    validation_0-mlogloss:0.19470
[17]    validation_0-mlogloss:0.19708
[18]    validation_0-mlogloss:0.19850
[19]    validation_0-mlogloss:0.19927
[20]    validation_0-mlogloss:0.20403
[21]    validation_0-mlogloss:0.19765
[22]    validation_0-mlogloss:0.20244
[23]    validation_0-mlogloss:0.20406
[24]    validation_0-mlogloss:0.19369
[25]    validation_0-mlogloss:0.18979
[26]    validation_0-mlogloss:0.19739
[27]    validation_0-mlogloss:0.19581
[28]    validation_0-mlogloss:0.19496
[29]    validation_0-mlogloss:0.19694
[30]    validation_0-mlogloss:0.19586
[31]    validation_0-mlogloss:0.19517
[32]    validation_0-mlogloss:0.19934
[33]    validation_0-mlogloss:0.20635
[34]    validation_0-mlogloss:0.20972
[35]    validation_0-mlogloss:0.21543
[36]    validation_0-mlogloss:0.21337
[37]    validation_0-mlogloss:0.21599
[38]    validation_0-mlogloss:0.20597
[39]    validation_0-mlogloss:0.20879
[40]    validation_0-mlogloss:0.20587
[41]    validation_0-mlogloss:0.21073
[42]    validation_0-mlogloss:0.21039
[43]    validation_0-mlogloss:0.21572
[44]    validation_0-mlogloss:0.22029
[45]    validation_0-mlogloss:0.22133
[46]    validation_0-mlogloss:0.22059
[47]    validation_0-mlogloss:0.21280
[48]    validation_0-mlogloss:0.21491
[49]    validation_0-mlogloss:0.21490
[50]    validation_0-mlogloss:0.21792
[51]    validation_0-mlogloss:0.22038
[52]    validation_0-mlogloss:0.22155
[53]    validation_0-mlogloss:0.21856
[54]    validation_0-mlogloss:0.21932
[55]    validation_0-mlogloss:0.22159
[56]    validation_0-mlogloss:0.22387
[57]    validation_0-mlogloss:0.22451
[58]    validation_0-mlogloss:0.22621
[59]    validation_0-mlogloss:0.22676
[60]    validation_0-mlogloss:0.22704
[61]    validation_0-mlogloss:0.22874
[62]    validation_0-mlogloss:0.23009
[63]    validation_0-mlogloss:0.23175
[64]    validation_0-mlogloss:0.23425
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ threshold : 0.12324658781290054 훈련 끝 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
Trech: 0.123, n=3, ACC: 96.6667%
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ threshold : 0.30810031294822693 훈련 시작 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
[0]     validation_0-mlogloss:0.77705
[1]     validation_0-mlogloss:0.58963
[2]     validation_0-mlogloss:0.45225
[3]     validation_0-mlogloss:0.36848
[4]     validation_0-mlogloss:0.31810
[5]     validation_0-mlogloss:0.28636
[6]     validation_0-mlogloss:0.25007
[7]     validation_0-mlogloss:0.23111
[8]     validation_0-mlogloss:0.21377
[9]     validation_0-mlogloss:0.21224
[10]    validation_0-mlogloss:0.19897
[11]    validation_0-mlogloss:0.19630
[12]    validation_0-mlogloss:0.19150
[13]    validation_0-mlogloss:0.18845
[14]    validation_0-mlogloss:0.18392
[15]    validation_0-mlogloss:0.19014
[16]    validation_0-mlogloss:0.19038
[17]    validation_0-mlogloss:0.19579
[18]    validation_0-mlogloss:0.19383
[19]    validation_0-mlogloss:0.19266
[20]    validation_0-mlogloss:0.19811
[21]    validation_0-mlogloss:0.19076
[22]    validation_0-mlogloss:0.19559
[23]    validation_0-mlogloss:0.19549
[24]    validation_0-mlogloss:0.18680
[25]    validation_0-mlogloss:0.18343
[26]    validation_0-mlogloss:0.18843
[27]    validation_0-mlogloss:0.18791
[28]    validation_0-mlogloss:0.18521
[29]    validation_0-mlogloss:0.18713
[30]    validation_0-mlogloss:0.18399
[31]    validation_0-mlogloss:0.18739
[32]    validation_0-mlogloss:0.18949
[33]    validation_0-mlogloss:0.19502
[34]    validation_0-mlogloss:0.19736
[35]    validation_0-mlogloss:0.20122
[36]    validation_0-mlogloss:0.20241
[37]    validation_0-mlogloss:0.20163
[38]    validation_0-mlogloss:0.19396
[39]    validation_0-mlogloss:0.19694
[40]    validation_0-mlogloss:0.19195
[41]    validation_0-mlogloss:0.19439
[42]    validation_0-mlogloss:0.19717
[43]    validation_0-mlogloss:0.20062
[44]    validation_0-mlogloss:0.20401
[45]    validation_0-mlogloss:0.20658
[46]    validation_0-mlogloss:0.20624
[47]    validation_0-mlogloss:0.20063
[48]    validation_0-mlogloss:0.20293
[49]    validation_0-mlogloss:0.20298
[50]    validation_0-mlogloss:0.20801
[51]    validation_0-mlogloss:0.21038
[52]    validation_0-mlogloss:0.21257
[53]    validation_0-mlogloss:0.21038
[54]    validation_0-mlogloss:0.21083
[55]    validation_0-mlogloss:0.21286
[56]    validation_0-mlogloss:0.21542
[57]    validation_0-mlogloss:0.21760
[58]    validation_0-mlogloss:0.21951
[59]    validation_0-mlogloss:0.22048
[60]    validation_0-mlogloss:0.22196
[61]    validation_0-mlogloss:0.22376
[62]    validation_0-mlogloss:0.22475
[63]    validation_0-mlogloss:0.22664
[64]    validation_0-mlogloss:0.23054
[65]    validation_0-mlogloss:0.23227
[66]    validation_0-mlogloss:0.23534
[67]    validation_0-mlogloss:0.23318
[68]    validation_0-mlogloss:0.23573
[69]    validation_0-mlogloss:0.23724
[70]    validation_0-mlogloss:0.23946
[71]    validation_0-mlogloss:0.24299
[72]    validation_0-mlogloss:0.24437
[73]    validation_0-mlogloss:0.24576
[74]    validation_0-mlogloss:0.24761
[75]    validation_0-mlogloss:0.25076
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ threshold : 0.30810031294822693 훈련 끝 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
Trech: 0.308, n=2, ACC: 96.6667%
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ threshold : 0.4975121319293976 훈련 시작 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
[0]     validation_0-mlogloss:0.79545
[1]     validation_0-mlogloss:0.61853
[2]     validation_0-mlogloss:0.50572
[3]     validation_0-mlogloss:0.42246
[4]     validation_0-mlogloss:0.37221
[5]     validation_0-mlogloss:0.33838
[6]     validation_0-mlogloss:0.30894
[7]     validation_0-mlogloss:0.29088
[8]     validation_0-mlogloss:0.27530
[9]     validation_0-mlogloss:0.27134
[10]    validation_0-mlogloss:0.25128
[11]    validation_0-mlogloss:0.25847
[12]    validation_0-mlogloss:0.24606
[13]    validation_0-mlogloss:0.24067
[14]    validation_0-mlogloss:0.24291
[15]    validation_0-mlogloss:0.24000
[16]    validation_0-mlogloss:0.24231
[17]    validation_0-mlogloss:0.25066
[18]    validation_0-mlogloss:0.25108
[19]    validation_0-mlogloss:0.24565
[20]    validation_0-mlogloss:0.25731
[21]    validation_0-mlogloss:0.24531
[22]    validation_0-mlogloss:0.24757
[23]    validation_0-mlogloss:0.24430
[24]    validation_0-mlogloss:0.24396
[25]    validation_0-mlogloss:0.23558
[26]    validation_0-mlogloss:0.23846
[27]    validation_0-mlogloss:0.24721
[28]    validation_0-mlogloss:0.24233
[29]    validation_0-mlogloss:0.24345
[30]    validation_0-mlogloss:0.24831
[31]    validation_0-mlogloss:0.25486
[32]    validation_0-mlogloss:0.25442
[33]    validation_0-mlogloss:0.25557
[34]    validation_0-mlogloss:0.25722
[35]    validation_0-mlogloss:0.26986
[36]    validation_0-mlogloss:0.26668
[37]    validation_0-mlogloss:0.26376
[38]    validation_0-mlogloss:0.25462
[39]    validation_0-mlogloss:0.25382
[40]    validation_0-mlogloss:0.25109
[41]    validation_0-mlogloss:0.26064
[42]    validation_0-mlogloss:0.26288
[43]    validation_0-mlogloss:0.26869
[44]    validation_0-mlogloss:0.28087
[45]    validation_0-mlogloss:0.28249
[46]    validation_0-mlogloss:0.27946
[47]    validation_0-mlogloss:0.27137
[48]    validation_0-mlogloss:0.27916
[49]    validation_0-mlogloss:0.27363
[50]    validation_0-mlogloss:0.28158
[51]    validation_0-mlogloss:0.28441
[52]    validation_0-mlogloss:0.28947
[53]    validation_0-mlogloss:0.28353
[54]    validation_0-mlogloss:0.28560
[55]    validation_0-mlogloss:0.28524
[56]    validation_0-mlogloss:0.29051
[57]    validation_0-mlogloss:0.28769
[58]    validation_0-mlogloss:0.28862
[59]    validation_0-mlogloss:0.29366
[60]    validation_0-mlogloss:0.29522
[61]    validation_0-mlogloss:0.29395
[62]    validation_0-mlogloss:0.29899
[63]    validation_0-mlogloss:0.29516
[64]    validation_0-mlogloss:0.30062
[65]    validation_0-mlogloss:0.30260
[66]    validation_0-mlogloss:0.30908
[67]    validation_0-mlogloss:0.30545
[68]    validation_0-mlogloss:0.31083
[69]    validation_0-mlogloss:0.30941
[70]    validation_0-mlogloss:0.31130
[71]    validation_0-mlogloss:0.30778
[72]    validation_0-mlogloss:0.31090
[73]    validation_0-mlogloss:0.31267
[74]    validation_0-mlogloss:0.31956
[75]    validation_0-mlogloss:0.31454
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ threshold : 0.4975121319293976 훈련 끝 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
Trech: 0.498, n=1, ACC: 90.0000%
verbose : 1
"""

    