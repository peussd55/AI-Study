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
feature_names = datasets.feature_names

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
# 1. 기본방식
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
# model_gain_normalized = model_gain / total_sum
model_gain_normalized = [x / total_sum for x in model_gain]
# SelectModel이 threshold 계산할때 1이하 부동소수점으로 계산하므로 정규화 해줘야한다.
print(model_gain_normalized)
threshold = np.sort(model_gain_normalized)

# 2. 선생님 방식
######
score_dict = model.get_booster().get_score(importance_type='gain')
print(score_dict)
# {'f0': 0.09054344147443771, 'f1': 0.44619157910346985, 'f2': 0.9568471312522888, 'f3': 0.22348596155643463, 'f4': 0.20814745128154755, 'f5': 0.39905595779418945, 
# 'f6': 0.027111342176795006, 'f7': 0.824626088142395, 'f8': 0.19422045350074768, 'f9': 0.15491503477096558, 'f10': 0.40824103355407715, 
# 'f11': 0.35078516602516174, 'f12': 0.27346092462539673, 'f13': 0.539326548576355, 'f14': 0.14593474566936493, 'f15': 0.4656308889389038, 
# 'f16': 0.4021739661693573, 'f17': 0.1481102705001831, 'f18': 0.19117841124534607, 'f19': 0.22889749705791473, 'f20': 0.5641250610351562, 
# 'f21': 0.8972993493080139, 'f22': 5.388490676879883, 'f23': 0.6300467848777771, 'f24': 0.8262636661529541, 'f25': 0.1927350014448166, 
# 'f26': 1.107054352760315, 'f27': 0.9051604866981506, 'f28': 0.22356440126895905, 'f29': 0.06334438920021057}
total = sum(score_dict.values())
print(total)    # 17.476968063041568
score_list = [score_dict.get(f"f{i}", 0) / total for i in range(x.shape[1])]
print(score_list)
# [0.005180729354647581, 0.02553026231403537, 0.05474903471819734, 0.01278745608221594, 0.011909814707604554, 0.022833248670750306, 0.0015512611843771225, 
# 0.047183589577314985, 0.011112937484360598, 0.008863953645287217, 0.023358801828869954, 0.02007128265954579, 0.015646931643920708, 0.03085927413902331, 
# 0.008350118003475225, 0.026642543904601534, 0.023011655380879938, 0.008474597536937254, 0.010938877416022166, 0.013097094200335743, 0.03227819945658125, 
# 0.051341820049756064, 0.308319535599249, 0.03605011936882422, 0.04727728878215716, 0.011027942647122648, 0.06334361593881925, 0.051791619886992164, 
# 0.012791944258439726, 0.0036244495596558617]
threshold = np.sort(score_list)
# threshold[-1] = threshold[-1] - 0.00000002

threshold = threshold - 0.00000002
# 부동소수점 8번째 자리에서 threshold와 select_model의 임계값이 일치하지않아 제대로 매칭시키지 못하는 문제 발생
# -> threshold값을 조금씩 줄여서 select_model에서 매칭시키도록 하기위해 적용함. (내 환경에서만 이러는지 모르겠음)

print(threshold)
# [0.00155126 0.00362445 0.00518073 0.00835012 0.0084746  0.00886395
#  0.01093888 0.01102794 0.01111294 0.01190981 0.01278746 0.01279194
#  0.01309709 0.01564693 0.02007128 0.02283325 0.02301166 0.0233588
#  0.02553026 0.02664254 0.03085927 0.0322782  0.03605012 0.04718359
#  0.04727729 0.05134182 0.05179162 0.05474903 0.06334362 0.30831952]
# exit()

#### 컬럼명매칭 ####
print(feature_names)
# ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
#  'mean smoothness' 'mean compactness' 'mean concavity'
#  'mean concave points' 'mean symmetry' 'mean fractal dimension'
#  'radius error' 'texture error' 'perimeter error' 'area error'
#  'smoothness error' 'compactness error' 'concavity error'
#  'concave points error' 'symmetry error' 'fractal dimension error'
#  'worst radius' 'worst texture' 'worst perimeter' 'worst area'
#  'worst smoothness' 'worst compactness' 'worst concavity'
#  'worst concave points' 'worst symmetry' 'worst fractal dimension']
score_df = pd.DataFrame({
    # 'feature' : [feature_names[int(f[1:])]   for f in score_dict.keys()],
    'feature' : feature_names,
    'gain' : list(score_dict.values())
}).sort_values(by='gain', ascending=True)   # 오름차순
# print(score_df.index[0])
# exit()
#                     feature      gain
# 6            mean concavity  0.027111
# 29  worst fractal dimension  0.063344
# 0               mean radius  0.090543
# 14         smoothness error  0.145935
# 17     concave points error  0.148110
# 9    mean fractal dimension  0.154915
# 18           symmetry error  0.191178
# 25        worst compactness  0.192735
# 8             mean symmetry  0.194220
# 4           mean smoothness  0.208147
# 3                 mean area  0.223486
# 28           worst symmetry  0.223564
# 19  fractal dimension error  0.228897
# 12          perimeter error  0.273461
# 11            texture error  0.350785
# 5          mean compactness  0.399056
# 16          concavity error  0.402174
# 10             radius error  0.408241
# 1              mean texture  0.446192
# 15        compactness error  0.465631
# 13               area error  0.539327
# 20             worst radius  0.564125
# 23               worst area  0.630047
# 7       mean concave points  0.824626
# 24         worst smoothness  0.826264
# 21            worst texture  0.897299
# 27     worst concave points  0.905160
# 2            mean perimeter  0.956847
# 26          worst concavity  1.107054
# 22          worst perimeter  5.388491
# exit()
####

# threshold = np.sort(model_gain) # default : 오름차순
# print(threshold)    
# # [0.02711134 0.06334439 0.09054344 0.14593475 0.14811027 0.15491503
# #  0.19117841 0.192735   0.19422045 0.20814745 0.22348596 0.2235644
# #  0.2288975  0.27346092 0.35078517 0.39905596 0.40217397 0.40824103
# #  0.44619158 0.46563089 0.53932655 0.56412506 0.63004678 0.82462609
# #  0.82626367 0.89729935 0.90516049 0.95684713 1.10705435 5.38849068]
# print("\n")
# threshold = threshold/sum
# print(threshold)
# # [0.00155126 0.00362445 0.00518073 0.00835012 0.0084746  0.00886395
# #  0.01093888 0.01102794 0.01111294 0.01190981 0.01278746 0.01279194
# #  0.01309709 0.01564693 0.02007128 0.02283325 0.02301166 0.0233588
# #  0.02553026 0.02664254 0.03085927 0.0322782  0.03605012 0.04718359
# #  0.04727729 0.05134182 0.05179162 0.05474903 0.06334362 0.30831954]
# ######### threshhold 변형 ######
# exit()
from sklearn.feature_selection import SelectFromModel
delete_columns = []
max_acc = 0
nn = 0
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
    
    mask = selection.get_support()
    print("선택된 피처 :", mask)
    
    not_select_features = [feature_names[j] 
                            for j, selected in enumerate(mask) 
                            if not selected]
    
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
    print('threshold :', i)
    print('정규화 안 된 threshold :', i * total)
    print('삭제할 컬럼명: ', score_df[score_df['gain'] < i * total]['feature'].tolist())
    print('삭제할 컬럼 :', not_select_features)
    print('Trech: %.10f, n=%d, acc : %.4f%%' %(i, select_x_train.shape[1], score*100))
    if score >= max_acc:
        max_acc = score
        nn = select_x_train.shape[1]
        delete_columns = not_select_features
    print("\n")
    # print("\n\n")
print('최대의 acc보장하면서 가장 많이 삭제된 컬럼 리스트 :', delete_columns)
print('그때 acc :', max_acc*100)
print('그때 남아있는 컬럼의 갯수 :', nn)


print(f"verbose : {verbose}")

"""
선택된 피처 : [ True  True  True  True  True  True  True  True  True  True  True  True
  True  True  True  True  True  True  True  True  True  True  True  True
  True  True  True  True  True  True]
threshold : 0.0015512411843771224
정규화 안 된 threshold : 0.027110992637433745
삭제할 컬럼명:  []
삭제할 컬럼 : []
Trech: 0.0015512412, n=30, acc : 98.2456%


선택된 피처 : [ True  True  True  True  True  True False  True  True  True  True  True
  True  True  True  True  True  True  True  True  True  True  True  True
  True  True  True  True  True  True]
threshold : 0.0036244295596558618
정규화 안 된 threshold : 0.06334403966084931
삭제할 컬럼명:  ['mean concavity']
삭제할 컬럼 : ['mean concavity']
Trech: 0.0036244296, n=29, acc : 98.2456%


선택된 피처 : [ True  True  True  True  True  True False  True  True  True  True  True
  True  True  True  True  True  True  True  True  True  True  True  True
  True  True  True  True  True False]
threshold : 0.005180709354647581
정규화 안 된 threshold : 0.09054309193507647
삭제할 컬럼명:  ['mean concavity', 'worst fractal dimension']
삭제할 컬럼 : ['mean concavity', 'worst fractal dimension']
Trech: 0.0051807094, n=28, acc : 98.2456%


선택된 피처 : [False  True  True  True  True  True False  True  True  True  True  True
  True  True  True  True  True  True  True  True  True  True  True  True
  True  True  True  True  True False]
threshold : 0.008350098003475226
정규화 안 된 threshold : 0.14593439613000367
삭제할 컬럼명:  ['mean concavity', 'worst fractal dimension', 'mean radius']
삭제할 컬럼 : ['mean radius', 'mean concavity', 'worst fractal dimension']
Trech: 0.0083500980, n=27, acc : 98.2456%


선택된 피처 : [False  True  True  True  True  True False  True  True  True  True  True
  True  True False  True  True  True  True  True  True  True  True  True
  True  True  True  True  True False]
threshold : 0.008474577536937254
정규화 안 된 threshold : 0.14810992096082184
삭제할 컬럼명:  ['mean concavity', 'worst fractal dimension', 'mean radius', 'smoothness error']
삭제할 컬럼 : ['mean radius', 'mean concavity', 'smoothness error', 'worst fractal dimension']
Trech: 0.0084745775, n=26, acc : 99.1228%


선택된 피처 : [False  True  True  True  True  True False  True  True  True  True  True
  True  True False  True  True False  True  True  True  True  True  True
  True  True  True  True  True False]
threshold : 0.008863933645287217
정규화 안 된 threshold : 0.15491468523160432
삭제할 컬럼명:  ['mean concavity', 'worst fractal dimension', 'mean radius', 'smoothness error', 'concave points error']
삭제할 컬럼 : ['mean radius', 'mean concavity', 'smoothness error', 'concave points error', 'worst fractal dimension']
Trech: 0.0088639336, n=25, acc : 98.2456%


선택된 피처 : [False  True  True  True  True  True False  True  True False  True  True
  True  True False  True  True False  True  True  True  True  True  True
  True  True  True  True  True False]
threshold : 0.010938857416022166
정규화 안 된 threshold : 0.1911780617059848
삭제할 컬럼명:  ['mean concavity', 'worst fractal dimension', 'mean radius', 'smoothness error', 'concave points error', 'mean fractal dimension']
삭제할 컬럼 : ['mean radius', 'mean concavity', 'mean fractal dimension', 'smoothness error', 'concave points error', 'worst fractal dimension']
Trech: 0.0109388574, n=24, acc : 99.1228%


선택된 피처 : [False  True  True  True  True  True False  True  True False  True  True
  True  True False  True  True False False  True  True  True  True  True
  True  True  True  True  True False]
threshold : 0.011027922647122648
정규화 안 된 threshold : 0.19273465190545536
삭제할 컬럼명:  ['mean concavity', 'worst fractal dimension', 'mean radius', 'smoothness error', 'concave points error', 'mean fractal dimension', 'symmetry error']
삭제할 컬럼 : ['mean radius', 'mean concavity', 'mean fractal dimension', 'smoothness error', 'concave points error', 'symmetry error', 'worst fractal dimension']
Trech: 0.0110279226, n=23, acc : 98.2456%


선택된 피처 : [False  True  True  True  True  True False  True  True False  True  True
  True  True False  True  True False False  True  True  True  True  True
  True False  True  True  True False]
threshold : 0.011112917484360598
정규화 안 된 threshold : 0.19422010396138642
삭제할 컬럼명:  ['mean concavity', 'worst fractal dimension', 'mean radius', 'smoothness error', 'concave points error', 'mean fractal dimension', 'symmetry error', 'worst compactness']
삭제할 컬럼 : ['mean radius', 'mean concavity', 'mean fractal dimension', 'smoothness error', 'concave points error', 'symmetry error', 'worst compactness', 'worst fractal dimension']
Trech: 0.0111129175, n=22, acc : 98.2456%


선택된 피처 : [False  True  True  True  True  True False  True False False  True  True
  True  True False  True  True False False  True  True  True  True  True
  True False  True  True  True False]
threshold : 0.011909794707604554
정규화 안 된 threshold : 0.20814710174218629
삭제할 컬럼명:  ['mean concavity', 'worst fractal dimension', 'mean radius', 'smoothness error', 'concave points error', 'mean fractal dimension', 'symmetry error', 'worst compactness', 'mean symmetry']        
삭제할 컬럼 : ['mean radius', 'mean concavity', 'mean symmetry', 'mean fractal dimension', 'smoothness error', 'concave points error', 'symmetry error', 'worst compactness', 'worst fractal dimension']
Trech: 0.0119097947, n=21, acc : 98.2456%


선택된 피처 : [False  True  True  True False  True False  True False False  True  True
  True  True False  True  True False False  True  True  True  True  True
  True False  True  True  True False]
threshold : 0.01278743608221594
정규화 안 된 threshold : 0.22348561201707337
삭제할 컬럼명:  ['mean concavity', 'worst fractal dimension', 'mean radius', 'smoothness error', 'concave points error', 'mean fractal dimension', 'symmetry error', 'worst compactness', 'mean symmetry', 'mean smoothness']
삭제할 컬럼 : ['mean radius', 'mean smoothness', 'mean concavity', 'mean symmetry', 'mean fractal dimension', 'smoothness error', 'concave points error', 'symmetry error', 'worst compactness', 'worst fractal dimension']
Trech: 0.0127874361, n=20, acc : 98.2456%


선택된 피처 : [False  True  True False False  True False  True False False  True  True
  True  True False  True  True False False  True  True  True  True  True
  True False  True  True  True False]
threshold : 0.012791924258439726
정규화 안 된 threshold : 0.22356405172959778
삭제할 컬럼명:  ['mean concavity', 'worst fractal dimension', 'mean radius', 'smoothness error', 'concave points error', 'mean fractal dimension', 'symmetry error', 'worst compactness', 'mean symmetry', 'mean smoothness', 'mean area']
삭제할 컬럼 : ['mean radius', 'mean area', 'mean smoothness', 'mean concavity', 'mean symmetry', 'mean fractal dimension', 'smoothness error', 'concave points error', 'symmetry error', 'worst compactness', 'worst fractal dimension']
Trech: 0.0127919243, n=19, acc : 98.2456%


선택된 피처 : [False  True  True False False  True False  True False False  True  True
  True  True False  True  True False False  True  True  True  True  True
  True False  True  True False False]
threshold : 0.013097074200335743
정규화 안 된 threshold : 0.22889714751855347
삭제할 컬럼명:  ['mean concavity', 'worst fractal dimension', 'mean radius', 'smoothness error', 'concave points error', 'mean fractal dimension', 'symmetry error', 'worst compactness', 'mean symmetry', 'mean smoothness', 'mean area', 'worst symmetry']
삭제할 컬럼 : ['mean radius', 'mean area', 'mean smoothness', 'mean concavity', 'mean symmetry', 'mean fractal dimension', 'smoothness error', 'concave points error', 'symmetry error', 'worst compactness', 'worst symmetry', 'worst fractal dimension']
Trech: 0.0130970742, n=18, acc : 98.2456%


선택된 피처 : [False  True  True False False  True False  True False False  True  True
  True  True False  True  True False False False  True  True  True  True
  True False  True  True False False]
threshold : 0.015646911643920708
정규화 안 된 threshold : 0.27346057508603544
삭제할 컬럼명:  ['mean concavity', 'worst fractal dimension', 'mean radius', 'smoothness error', 'concave points error', 'mean fractal dimension', 'symmetry error', 'worst compactness', 'mean symmetry', 'mean smoothness', 'mean area', 'worst symmetry', 'fractal dimension error']
삭제할 컬럼 : ['mean radius', 'mean area', 'mean smoothness', 'mean concavity', 'mean symmetry', 'mean fractal dimension', 'smoothness error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst compactness', 'worst symmetry', 'worst fractal dimension']
Trech: 0.0156469116, n=17, acc : 97.3684%


선택된 피처 : [False  True  True False False  True False  True False False  True  True
 False  True False  True  True False False False  True  True  True  True
  True False  True  True False False]
threshold : 0.02007126265954579
정규화 안 된 threshold : 0.3507848164858005
삭제할 컬럼명:  ['mean concavity', 'worst fractal dimension', 'mean radius', 'smoothness error', 'concave points error', 'mean fractal dimension', 'symmetry error', 'worst compactness', 'mean symmetry', 'mean smoothness', 'mean area', 'worst symmetry', 'fractal dimension error', 'perimeter error']
삭제할 컬럼 : ['mean radius', 'mean area', 'mean smoothness', 'mean concavity', 'mean symmetry', 'mean fractal dimension', 'perimeter error', 'smoothness error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst compactness', 'worst symmetry', 'worst fractal dimension']
Trech: 0.0200712627, n=16, acc : 96.4912%


선택된 피처 : [False  True  True False False  True False  True False False  True False
 False  True False  True  True False False False  True  True  True  True
  True False  True  True False False]
threshold : 0.022833228670750307
정규화 안 된 threshold : 0.39905560825482816
삭제할 컬럼명:  ['mean concavity', 'worst fractal dimension', 'mean radius', 'smoothness error', 'concave points error', 'mean fractal dimension', 'symmetry error', 'worst compactness', 'mean symmetry', 'mean smoothness', 'mean area', 'worst symmetry', 'fractal dimension error', 'perimeter error', 'texture error']
삭제할 컬럼 : ['mean radius', 'mean area', 'mean smoothness', 'mean concavity', 'mean symmetry', 'mean fractal dimension', 'texture error', 'perimeter error', 'smoothness error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst compactness', 'worst symmetry', 'worst fractal dimension']
Trech: 0.0228332287, n=15, acc : 99.1228%


선택된 피처 : [False  True  True False False False False  True False False  True False
 False  True False  True  True False False False  True  True  True  True
  True False  True  True False False]
threshold : 0.023011635380879938
정규화 안 된 threshold : 0.40217361662999607
삭제할 컬럼명:  ['mean concavity', 'worst fractal dimension', 'mean radius', 'smoothness error', 'concave points error', 'mean fractal dimension', 'symmetry error', 'worst compactness', 'mean symmetry', 'mean smoothness', 'mean area', 'worst symmetry', 'fractal dimension error', 'perimeter error', 'texture error', 'mean compactness']
삭제할 컬럼 : ['mean radius', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean symmetry', 'mean fractal dimension', 'texture error', 'perimeter error', 'smoothness error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst compactness', 'worst symmetry', 'worst fractal dimension']
Trech: 0.0230116354, n=14, acc : 98.2456%


선택된 피처 : [False  True  True False False False False  True False False  True False
 False  True False  True False False False False  True  True  True  True
  True False  True  True False False]
threshold : 0.023358781828869954
정규화 안 된 threshold : 0.4082406840147159
삭제할 컬럼명:  ['mean concavity', 'worst fractal dimension', 'mean radius', 'smoothness error', 'concave points error', 'mean fractal dimension', 'symmetry error', 'worst compactness', 'mean symmetry', 'mean smoothness', 'mean area', 'worst symmetry', 'fractal dimension error', 'perimeter error', 'texture error', 'mean compactness', 'concavity error']
삭제할 컬럼 : ['mean radius', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean symmetry', 'mean fractal dimension', 'texture error', 'perimeter error', 'smoothness error', 'concavity 
error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst compactness', 'worst symmetry', 'worst fractal dimension']
Trech: 0.0233587818, n=13, acc : 98.2456%


선택된 피처 : [False  True  True False False False False  True False False False False
 False  True False  True False False False False  True  True  True  True
  True False  True  True False False]
threshold : 0.02553024231403537
정규화 안 된 threshold : 0.4461912295641086
삭제할 컬럼명:  ['mean concavity', 'worst fractal dimension', 'mean radius', 'smoothness error', 'concave points error', 'mean fractal dimension', 'symmetry error', 'worst compactness', 'mean symmetry', 'mean smoothness', 'mean area', 'worst symmetry', 'fractal dimension error', 'perimeter error', 'texture error', 'mean compactness', 'concavity error', 'radius error']
삭제할 컬럼 : ['mean radius', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'smoothness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst compactness', 'worst symmetry', 'worst fractal dimension']
Trech: 0.0255302423, n=12, acc : 98.2456%


선택된 피처 : [False False  True False False False False  True False False False False
 False  True False  True False False False False  True  True  True  True
  True False  True  True False False]
threshold : 0.026642523904601534
정규화 안 된 threshold : 0.4656305393995425
삭제할 컬럼명:  ['mean concavity', 'worst fractal dimension', 'mean radius', 'smoothness error', 'concave points error', 'mean fractal dimension', 'symmetry error', 'worst compactness', 'mean symmetry', 'mean smoothness', 'mean area', 'worst symmetry', 'fractal dimension error', 'perimeter error', 'texture error', 'mean compactness', 'concavity error', 'radius error', 'mean texture']
삭제할 컬럼 : ['mean radius', 'mean texture', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'smoothness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst compactness', 'worst symmetry', 'worst fractal dimension']
Trech: 0.0266425239, n=11, acc : 99.1228%


선택된 피처 : [False False  True False False False False  True False False False False
 False  True False False False False False False  True  True  True  True
  True False  True  True False False]
threshold : 0.03085925413902331
정규화 안 된 threshold : 0.5393261990369936
삭제할 컬럼명:  ['mean concavity', 'worst fractal dimension', 'mean radius', 'smoothness error', 'concave points error', 'mean fractal dimension', 'symmetry error', 'worst compactness', 'mean symmetry', 'mean smoothness', 'mean area', 'worst symmetry', 'fractal dimension error', 'perimeter error', 'texture error', 'mean compactness', 'concavity error', 'radius error', 'mean texture', 'compactness error']
삭제할 컬럼 : ['mean radius', 'mean texture', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'smoothness error', 'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst compactness', 'worst symmetry', 'worst fractal dimension']
Trech: 0.0308592541, n=10, acc : 98.2456%


선택된 피처 : [False False  True False False False False  True False False False False
 False False False False False False False False  True  True  True  True
  True False  True  True False False]
threshold : 0.03227817945658125
정규화 안 된 threshold : 0.5641247114957949
삭제할 컬럼명:  ['mean concavity', 'worst fractal dimension', 'mean radius', 'smoothness error', 'concave points error', 'mean fractal dimension', 'symmetry error', 'worst compactness', 'mean symmetry', 'mean smoothness', 'mean area', 'worst symmetry', 'fractal dimension error', 'perimeter error', 'texture error', 'mean compactness', 'concavity error', 'radius error', 'mean texture', 'compactness error', 'area error']
삭제할 컬럼 : ['mean radius', 'mean texture', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst compactness', 'worst symmetry', 'worst fractal dimension'] 
Trech: 0.0322781795, n=9, acc : 98.2456%


선택된 피처 : [False False  True False False False False  True False False False False
 False False False False False False False False False  True  True  True
  True False  True  True False False]
threshold : 0.036050099368824216
정규화 안 된 threshold : 0.6300464353384158
삭제할 컬럼명:  ['mean concavity', 'worst fractal dimension', 'mean radius', 'smoothness error', 'concave points error', 'mean fractal dimension', 'symmetry error', 'worst compactness', 'mean symmetry', 'mean smoothness', 'mean area', 'worst symmetry', 'fractal dimension error', 'perimeter error', 'texture error', 'mean compactness', 'concavity error', 'radius error', 'mean texture', 'compactness error', 'area error', 'worst radius']
삭제할 컬럼 : ['mean radius', 'mean texture', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius', 'worst compactness', 'worst symmetry', 'worst fractal dimension']
Trech: 0.0360500994, n=8, acc : 97.3684%


선택된 피처 : [False False  True False False False False  True False False False False
 False False False False False False False False False  True  True False
  True False  True  True False False]
threshold : 0.04718356957731498
정규화 안 된 threshold : 0.8246257386030337
삭제할 컬럼명:  ['mean concavity', 'worst fractal dimension', 'mean radius', 'smoothness error', 'concave points error', 'mean fractal dimension', 'symmetry error', 'worst compactness', 'mean symmetry', 'mean smoothness', 'mean area', 'worst symmetry', 'fractal dimension error', 'perimeter error', 'texture error', 'mean compactness', 'concavity error', 'radius error', 'mean texture', 'compactness error', 'area error', 'worst radius', 'worst area']
삭제할 컬럼 : ['mean radius', 'mean texture', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius', 'worst area', 'worst compactness', 'worst symmetry', 'worst fractal dimension']
Trech: 0.0471835696, n=7, acc : 96.4912%


선택된 피처 : [False False  True False False False False False False False False False
 False False False False False False False False False  True  True False
  True False  True  True False False]
threshold : 0.047277268782157156
정규화 안 된 threshold : 0.8262633166135928
삭제할 컬럼명:  ['mean concavity', 'worst fractal dimension', 'mean radius', 'smoothness error', 'concave points error', 'mean fractal dimension', 'symmetry error', 'worst compactness', 'mean symmetry', 'mean smoothness', 'mean area', 'worst symmetry', 'fractal dimension error', 'perimeter error', 'texture error', 'mean compactness', 'concavity error', 'radius error', 'mean texture', 'compactness error', 'area error', 'worst radius', 'worst area', 'mean concave points']
삭제할 컬럼 : ['mean radius', 'mean texture', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius', 'worst area', 'worst compactness', 'worst symmetry', 'worst fractal dimension']
Trech: 0.0472772688, n=6, acc : 98.2456%


선택된 피처 : [False False  True False False False False False False False False False
 False False False False False False False False False  True  True False
 False False  True  True False False]
threshold : 0.05134180004975606
정규화 안 된 threshold : 0.8972989997686527
삭제할 컬럼명:  ['mean concavity', 'worst fractal dimension', 'mean radius', 'smoothness error', 'concave points error', 'mean fractal dimension', 'symmetry error', 'worst compactness', 'mean symmetry', 'mean smoothness', 'mean area', 'worst symmetry', 'fractal dimension error', 'perimeter error', 'texture error', 'mean compactness', 'concavity error', 'radius error', 'mean texture', 'compactness error', 'area error', 'worst radius', 'worst area', 'mean concave points', 'worst smoothness']
삭제할 컬럼 : ['mean radius', 'mean texture', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius', 'worst area', 'worst smoothness', 'worst compactness', 'worst symmetry', 'worst fractal dimension']
Trech: 0.0513418000, n=5, acc : 97.3684%


선택된 피처 : [False False  True False False False False False False False False False
 False False False False False False False False False False  True False
 False False  True  True False False]
threshold : 0.05179159988699216
정규화 안 된 threshold : 0.9051601371587893
삭제할 컬럼명:  ['mean concavity', 'worst fractal dimension', 'mean radius', 'smoothness error', 'concave points error', 'mean fractal dimension', 'symmetry error', 'worst compactness', 'mean symmetry', 'mean smoothness', 'mean area', 'worst symmetry', 'fractal dimension error', 'perimeter error', 'texture error', 'mean compactness', 'concavity error', 'radius error', 'mean texture', 'compactness error', 'area error', 'worst radius', 'worst area', 'mean concave points', 'worst smoothness', 'worst texture']
삭제할 컬럼 : ['mean radius', 'mean texture', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius', 'worst texture', 'worst area', 'worst smoothness', 'worst compactness', 'worst symmetry', 'worst fractal dimension']
Trech: 0.0517915999, n=4, acc : 94.7368%


선택된 피처 : [False False  True False False False False False False False False False
 False False False False False False False False False False  True False
 False False  True False False False]
threshold : 0.054749014718197334
정규화 안 된 threshold : 0.9568467817129276
삭제할 컬럼명:  ['mean concavity', 'worst fractal dimension', 'mean radius', 'smoothness error', 'concave points error', 'mean fractal dimension', 'symmetry error', 'worst compactness', 'mean symmetry', 'mean smoothness', 'mean area', 'worst symmetry', 'fractal dimension error', 'perimeter error', 'texture error', 'mean compactness', 'concavity error', 'radius error', 'mean texture', 'compactness error', 'area error', 'worst radius', 'worst area', 'mean concave points', 'worst smoothness', 'worst texture', 'worst concave points']
삭제할 컬럼 : ['mean radius', 'mean texture', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius', 'worst texture', 'worst area', 'worst smoothness', 'worst compactness', 'worst concave points', 'worst symmetry', 'worst fractal dimension']
Trech: 0.0547490147, n=3, acc : 97.3684%


선택된 피처 : [False False False False False False False False False False False False
 False False False False False False False False False False  True False
 False False  True False False False]
threshold : 0.06334359593881925
정규화 안 된 threshold : 1.1070540032209535
삭제할 컬럼명:  ['mean concavity', 'worst fractal dimension', 'mean radius', 'smoothness error', 'concave points error', 'mean fractal dimension', 'symmetry error', 'worst compactness', 'mean symmetry', 'mean smoothness', 'mean area', 'worst symmetry', 'fractal dimension error', 'perimeter error', 'texture error', 'mean compactness', 'concavity error', 'radius error', 'mean texture', 'compactness error', 'area error', 'worst radius', 'worst area', 'mean concave points', 'worst smoothness', 'worst texture', 'worst concave points', 'mean perimeter']
삭제할 컬럼 : ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius', 'worst texture', 'worst area', 'worst smoothness', 'worst compactness', 'worst concave points', 'worst symmetry', 'worst fractal dimension']
Trech: 0.0633435959, n=2, acc : 93.8596%


선택된 피처 : [False False False False False False False False False False False False
 False False False False False False False False False False  True False
 False False False False False False]
threshold : 0.308319515599249
정규화 안 된 threshold : 5.388490327340522
삭제할 컬럼명:  ['mean concavity', 'worst fractal dimension', 'mean radius', 'smoothness error', 'concave points error', 'mean fractal dimension', 'symmetry error', 'worst compactness', 'mean symmetry', 'mean smoothness', 'mean area', 'worst symmetry', 'fractal dimension error', 'perimeter error', 'texture error', 'mean compactness', 'concavity error', 'radius error', 'mean texture', 'compactness error', 'area error', 'worst radius', 'worst area', 'mean concave points', 'worst smoothness', 'worst texture', 'worst concave points', 'mean perimeter', 'worst concavity']
삭제할 컬럼 : ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius', 'worst texture', 'worst area', 'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension']
Trech: 0.3083195156, n=1, acc : 85.9649%


최대의 acc보장하면서 가장 많이 삭제된 컬럼 리스트 : ['mean radius', 'mean texture', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'smoothness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst compactness', 'worst symmetry', 'worst fractal dimension']
그때 acc : 99.12280701754386
그때 남아있는 컬럼의 갯수 : 11
verbose : 0
"""