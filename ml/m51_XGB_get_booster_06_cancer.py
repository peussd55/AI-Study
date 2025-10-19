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

# pandas 컬럼명 불일치경고 무시 (x가 pd.dataframe일때 사용)
import warnings
warnings.filterwarnings('ignore', message='X has feature names, but SelectFromModel was fitted without feature names')

seed = 123
random.seed(seed)
np.random.seed(seed)
metric_name = 'logloss'
verbose = 0

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (569, 30) (569,)

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
# [0.00155126 0.00362445 0.00518073 0.00835012 0.0084746  0.00886395
#  0.01093888 0.01102794 0.01111294 0.01190981 0.01278746 0.01279194
#  0.01309709 0.01564693 0.02007128 0.02283325 0.02301165 0.0233588
#  0.02553026 0.02664254 0.03085927 0.0322782  0.03605012 0.04718359
#  0.04727729 0.05134182 0.05179162 0.05474903 0.06334361 0.3083195 ]
print("\n")

######### threshhold 변형 ######
# 1. 첫번째 방식
aaa = model.get_booster().get_score(importance_type='gain')
print(aaa)
# {'f0': 0.09054344147443771, 'f1': 0.44619157910346985, 'f2': 0.9568471312522888, 'f3': 0.22348596155643463, 'f4': 0.20814745128154755, 'f5': 0.39905595779418945, 'f6': 0.027111342176795006, 'f7': 0.824626088142395, 
# 'f8': 0.19422045350074768, 'f9': 0.15491503477096558, 'f10': 0.40824103355407715, 'f11': 0.35078516602516174, 'f12': 0.27346092462539673, 'f13': 0.539326548576355, 'f14': 0.14593474566936493, 'f15': 0.4656308889389038, 'f16': 0.4021739661693573, 'f17': 0.1481102705001831, 'f18': 0.19117841124534607, 'f19': 0.22889749705791473, 'f20': 0.5641250610351562, 'f21': 0.8972993493080139, 'f22': 5.388490676879883, 'f23': 0.6300467848777771, 'f24': 0.8262636661529541, 'f25': 0.1927350014448166, 'f26': 1.107054352760315, 'f27': 0.9051604866981506, 'f28': 0.22356440126895905, 'f29': 0.06334438920021057}
model_gain = []
total_sum = 0
for i in aaa:
    model_gain.append(aaa[i])
    total_sum += aaa[i]
print(sum)  # 17.476968063041568
print(model_gain)
model_gain_normalized = [x / total_sum for x in model_gain]
# SelectModel이 threshold 계산할때 1이하 부동소수점으로 계산하므로 정규화 해줘야한다.
print(model_gain_normalized)
threshold = np.sort(model_gain_normalized)
print(threshold)
# [0.00155126 0.00362445 0.00518073 0.00835012 0.0084746  0.00886395
#  0.01093888 0.01102794 0.01111294 0.01190981 0.01278746 0.01279194
#  0.01309709 0.01564693 0.02007128 0.02283325 0.02301166 0.0233588
#  0.02553026 0.02664254 0.03085927 0.0322782  0.03605012 0.04718359
#  0.04727729 0.05134182 0.05179162 0.05474903 0.06334362 0.30831954]

# threshold[-1] = threshold[-1] * 0.99999
# threshold[-1] = threshold[-1] - 0.00000002

threshold = threshold - 0.00000002  
# 부동소수점 8번째 자리에서 threshold와 select_model의 임계값이 일치하지않아 제대로 매칭시키지 못하는 문제 발생
# -> threshold값을 조금씩 줄여서 select_model에서 매칭시키도록 하기위해 적용함. (내 환경에서만 이러는지 모르겠음)

# print(threshold[-1])    
# 0.00000001를 빼면 : 0.3083195 25599249     -> 마지막 threshold에서 학습안됨
# 0.00000002를 빼면 : 0.3083195 15599249     -> 마지막 threshold에서 학습됨
# 0.00000003를 빼면 : 0.3083195 0559924903   -> 마지막 threshold에서 학습됨
# exit()
"""
# 2. 2번째 방식
######
score_dict = model.get_booster().get_score(importance_type='gain')
print(score_dict)
# {'f0': 0.09054344147443771, 'f1': 0.44619157910346985, 'f2': 0.9568471312522888, 'f3': 0.22348596155643463, 'f4': 0.20814745128154755, 'f5': 0.39905595779418945, 'f6': 0.027111342176795006, 'f7': 0.824626088142395, 
# 'f8': 0.19422045350074768, 'f9': 0.15491503477096558, 'f10': 0.40824103355407715, 'f11': 0.35078516602516174, 'f12': 0.27346092462539673, 'f13': 0.539326548576355, 'f14': 0.14593474566936493, 'f15': 0.4656308889389038, 'f16': 0.4021739661693573, 'f17': 0.1481102705001831, 'f18': 0.19117841124534607, 'f19': 0.22889749705791473, 'f20': 0.5641250610351562, 'f21': 0.8972993493080139, 'f22': 5.388490676879883, 'f23': 0.6300467848777771, 'f24': 0.8262636661529541, 'f25': 0.1927350014448166, 'f26': 1.107054352760315, 'f27': 0.9051604866981506, 'f28': 0.22356440126895905, 'f29': 0.06334438920021057}
total = sum(score_dict.values())
print(total)    # 17.476968063041568
score_list = [score_dict.get(f"f{i}", 0) / total for i in range(x.shape[1])]
print(score_list)
# [0.005180729354647581, 0.02553026231403537, 0.05474903471819734, 0.01278745608221594, 0.011909814707604554, 0.022833248670750306, 0.0015512611843771225, 
# 0.047183589577314985, 0.011112937484360598, 0.008863953645287217, 0.023358801828869954, 0.02007128265954579, 0.015646931643920708, 0.03085927413902331, 
# 0.008350118003475225, 0.026642543904601534, 0.023011655380879938, 0.008474597536937254, 0.010938877416022166, 0.013097094200335743, 0.03227819945658125, 
# 0.051341820049756064, 0.308319535599249, 0.03605011936882422, 0.04727728878215716, 0.011027942647122648, 0.06334361593881925, 0.051791619886992164, 
# 0.012791944258439726, 0.0036244495596558617]
threshold = np.sort(score_list)     # np가 8자리를 맟추기 위해서 반올림함 : 0.308319535599249 -> 0.30831954
print(threshold)
# [0.00155126 0.00362445 0.00518073 0.00835012 0.0084746  0.00886395
#  0.01093888 0.01102794 0.01111294 0.01190981 0.01278746 0.01279194
#  0.01309709 0.01564693 0.02007128 0.02283325 0.02301166 0.0233588
#  0.02553026 0.02664254 0.03085927 0.0322782  0.03605012 0.04718359
#  0.04727729 0.05134182 0.05179162 0.05474903 0.06334362 0.30831954]
# threshold[-1] = threshold[-1] * 0.99999
threshold[-1] = threshold[-1] - 0.00000002
# print(threshold[-1])    
# 0.00000001를 빼면 : 0.3083195 25599249     -> 마지막 threshold에서 학습안됨
# 0.00000002를 빼면 : 0.3083195 15599249     -> 마지막 threshold에서 학습됨
# 0.00000003를 빼면 : 0.3083195 0559924903   -> 마지막 threshold에서 학습됨
# exit()
####
"""

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
    print('Trech: %.10f, n=%d, acc : %.4f%%' %(i, select_x_train.shape[1], score*100))
    # print("\n\n")
print(f"verbose : {verbose}")

"""
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ 그냥 모델훈련 시작 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ 그냥 모델훈련 끝 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ


========= XGBClassifier ========
accuracy_score : 0.9736842105263158
[0.00518073 0.02553026 0.05474903 0.01278746 0.01190981 0.02283325
 0.00155126 0.04718359 0.01111294 0.00886395 0.0233588  0.02007128
 0.01564693 0.03085927 0.00835012 0.02664254 0.02301165 0.0084746
 0.01093888 0.01309709 0.0322782  0.05134182 0.3083195  0.03605012
 0.04727729 0.01102794 0.06334361 0.05179162 0.01279194 0.00362445]
[0.00155126 0.00362445 0.00518073 0.00835012 0.0084746  0.00886395
 0.01093888 0.01102794 0.01111294 0.01190981 0.01278746 0.01279194
 0.01309709 0.01564693 0.02007128 0.02283325 0.02301165 0.0233588
 0.02553026 0.02664254 0.03085927 0.0322782  0.03605012 0.04718359
 0.04727729 0.05134182 0.05179162 0.05474903 0.06334361 0.3083195 ]


Trech: 0.002, n=30, acc : 97.3684%
Trech: 0.004, n=29, acc : 96.4912%
Trech: 0.005, n=28, acc : 96.4912%
Trech: 0.008, n=27, acc : 99.1228%
Trech: 0.008, n=26, acc : 98.2456%
Trech: 0.009, n=25, acc : 98.2456%
Trech: 0.011, n=24, acc : 97.3684%
Trech: 0.011, n=23, acc : 98.2456%
Trech: 0.011, n=22, acc : 98.2456%
Trech: 0.012, n=21, acc : 98.2456%
Trech: 0.013, n=20, acc : 96.4912%
Trech: 0.013, n=19, acc : 98.2456%
Trech: 0.013, n=18, acc : 98.2456%
Trech: 0.016, n=17, acc : 96.4912%
Trech: 0.020, n=16, acc : 96.4912%
Trech: 0.023, n=15, acc : 96.4912%
Trech: 0.023, n=14, acc : 98.2456%
Trech: 0.023, n=13, acc : 97.3684%
Trech: 0.026, n=12, acc : 98.2456%
Trech: 0.027, n=11, acc : 99.1228%
Trech: 0.031, n=10, acc : 98.2456%
Trech: 0.032, n=9, acc : 98.2456%
Trech: 0.036, n=8, acc : 97.3684%
Trech: 0.047, n=7, acc : 96.4912%
Trech: 0.047, n=6, acc : 98.2456%
Trech: 0.051, n=5, acc : 97.3684%
Trech: 0.052, n=4, acc : 94.7368%
Trech: 0.055, n=3, acc : 96.4912%
Trech: 0.063, n=2, acc : 92.9825%
Trech: 0.308, n=1, acc : 90.3509%
verbose : 0
"""
    