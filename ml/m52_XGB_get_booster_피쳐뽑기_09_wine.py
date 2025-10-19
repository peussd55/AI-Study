### <<37>>

from sklearn.datasets import load_wine
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
metric_name = 'mlogloss'
verbose = 0

# 1. 데이터
datasets = load_wine()
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
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ 그냥 모델훈련 시작 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ 그냥 모델훈련 끝 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ


========= XGBClassifier ========
accuracy_score : 0.9722222222222222
[0.05517212 0.05668676 0.01071067 0.02118997 0.05122918 0.02677136
 0.07888021 0.00055143 0.00498382 0.17661686 0.18855742 0.16711283
 0.16153738]
[0.00055143 0.00498382 0.01071067 0.02118997 0.02677136 0.05122918
 0.05517212 0.05668676 0.07888021 0.16153738 0.16711283 0.17661686
 0.18855742]


{'f0': 0.21766085922718048, 'f1': 0.22363629937171936, 'f2': 0.042254917323589325, 'f3': 0.0835970789194107, 'f4': 0.20210546255111694, 'f5': 0.10561633110046387, 'f6': 0.31119221448898315, 'f7': 0.002175477799028158, 'f8': 0.0196617990732193, 'f9': 0.696775496006012, 'f10': 0.7438824772834778, 'f11': 0.6592808961868286, 'f12': 0.637285053730011}
<built-in function sum>
[0.21766085922718048, 0.22363629937171936, 0.042254917323589325, 0.0835970789194107, 0.20210546255111694, 0.10561633110046387, 0.31119221448898315, 0.002175477799028158, 0.0196617990732193, 0.696775496006012, 0.7438824772834778, 0.6592808961868286, 0.637285053730011]
[0.0551721160593012, 0.05668675529361485, 0.010710668013214046, 0.021189973046767868, 0.05122917402641836, 0.026771356586213078, 0.07888020398107999, 0.0005514345300233311, 0.004983822375111038, 0.17661686473817026, 0.18855742147157456, 0.16711282979056444, 0.16153738008794696]
{'f0': 0.21766085922718048, 'f1': 0.22363629937171936, 'f2': 0.042254917323589325, 'f3': 0.0835970789194107, 'f4': 0.20210546255111694, 'f5': 0.10561633110046387, 'f6': 0.31119221448898315, 'f7': 0.002175477799028158, 'f8': 0.0196617990732193, 'f9': 0.696775496006012, 'f10': 0.7438824772834778, 'f11': 0.6592808961868286, 'f12': 0.637285053730011}
3.9451243630610406
[0.0551721160593012, 0.05668675529361485, 0.010710668013214046, 0.021189973046767868, 0.05122917402641836, 0.026771356586213078, 0.07888020398107999, 0.0005514345300233311, 0.004983822375111038, 0.17661686473817026, 0.18855742147157456, 0.16711282979056444, 0.16153738008794696]
[0.00055141 0.0049838  0.01071065 0.02118995 0.02677134 0.05122915
 0.0551721  0.05668674 0.07888018 0.16153736 0.16711281 0.17661684
 0.1885574 ]
['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'] 
선택된 피처 : [ True  True  True  True  True  True  True  True  True  True  True  True
  True]
threshold : 0.0005514145300233311
정규화 안 된 threshold : 0.002175398896540897
삭제할 컬럼명:  []
삭제할 컬럼 : []
Trech: 0.0005514145, n=13, acc : 97.2222%


선택된 피처 : [ True  True  True  True  True  True  True False  True  True  True  True
  True]
threshold : 0.0049838023751110384
정규화 안 된 threshold : 0.019661720170732037
삭제할 컬럼명:  ['nonflavanoid_phenols']
삭제할 컬럼 : ['nonflavanoid_phenols']
Trech: 0.0049838024, n=12, acc : 97.2222%


선택된 피처 : [ True  True  True  True  True  True  True False False  True  True  True
  True]
threshold : 0.010710648013214046
정규화 안 된 threshold : 0.042254838421102066
삭제할 컬럼명:  ['nonflavanoid_phenols', 'proanthocyanins']
삭제할 컬럼 : ['nonflavanoid_phenols', 'proanthocyanins']
Trech: 0.0107106480, n=11, acc : 97.2222%


선택된 피처 : [ True  True False  True  True  True  True False False  True  True  True
  True]
threshold : 0.02118995304676787
정규화 안 된 threshold : 0.08359700001692344
삭제할 컬럼명:  ['nonflavanoid_phenols', 'proanthocyanins', 'ash']
삭제할 컬럼 : ['ash', 'nonflavanoid_phenols', 'proanthocyanins']
Trech: 0.0211899530, n=10, acc : 97.2222%


선택된 피처 : [ True  True False False  True  True  True False False  True  True  True
  True]
threshold : 0.026771336586213078
정규화 안 된 threshold : 0.1056162521979766
삭제할 컬럼명:  ['nonflavanoid_phenols', 'proanthocyanins', 'ash', 'alcalinity_of_ash']
삭제할 컬럼 : ['ash', 'alcalinity_of_ash', 'nonflavanoid_phenols', 'proanthocyanins']
Trech: 0.0267713366, n=9, acc : 97.2222%


선택된 피처 : [ True  True False False  True False  True False False  True  True  True
  True]
threshold : 0.05122915402641836
정규화 안 된 threshold : 0.20210538364862968
삭제할 컬럼명:  ['nonflavanoid_phenols', 'proanthocyanins', 'ash', 'alcalinity_of_ash', 'total_phenols']
삭제할 컬럼 : ['ash', 'alcalinity_of_ash', 'total_phenols', 'nonflavanoid_phenols', 'proanthocyanins']
Trech: 0.0512291540, n=8, acc : 97.2222%


선택된 피처 : [ True  True False False False False  True False False  True  True  True
  True]
threshold : 0.0551720960593012
정규화 안 된 threshold : 0.2176607803246932
삭제할 컬럼명:  ['nonflavanoid_phenols', 'proanthocyanins', 'ash', 'alcalinity_of_ash', 'total_phenols', 'magnesium']
삭제할 컬럼 : ['ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'nonflavanoid_phenols', 'proanthocyanins']
Trech: 0.0551720961, n=7, acc : 97.2222%


선택된 피처 : [False  True False False False False  True False False  True  True  True
  True]
threshold : 0.05668673529361485
정규화 안 된 threshold : 0.2236362204692321
삭제할 컬럼명:  ['nonflavanoid_phenols', 'proanthocyanins', 'ash', 'alcalinity_of_ash', 'total_phenols', 'magnesium', 'alcohol']
삭제할 컬럼 : ['alcohol', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'nonflavanoid_phenols', 'proanthocyanins']
Trech: 0.0566867353, n=6, acc : 97.2222%


선택된 피처 : [False False False False False False  True False False  True  True  True
  True]
threshold : 0.07888018398107999
정규화 안 된 threshold : 0.3111921355864959
삭제할 컬럼명:  ['nonflavanoid_phenols', 'proanthocyanins', 'ash', 'alcalinity_of_ash', 'total_phenols', 'magnesium', 'alcohol', 'malic_acid']
삭제할 컬럼 : ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'nonflavanoid_phenols', 'proanthocyanins']
Trech: 0.0788801840, n=5, acc : 97.2222%


선택된 피처 : [False False False False False False False False False  True  True  True
  True]
threshold : 0.16153736008794697
정규화 안 된 threshold : 0.6372849748275238
삭제할 컬럼명:  ['nonflavanoid_phenols', 'proanthocyanins', 'ash', 'alcalinity_of_ash', 'total_phenols', 'magnesium', 'alcohol', 'malic_acid', 'flavanoids']
삭제할 컬럼 : ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins']
Trech: 0.1615373601, n=4, acc : 97.2222%


선택된 피처 : [False False False False False False False False False  True  True  True
 False]
threshold : 0.16711280979056445
정규화 안 된 threshold : 0.6592808172843414
삭제할 컬럼명:  ['nonflavanoid_phenols', 'proanthocyanins', 'ash', 'alcalinity_of_ash', 'total_phenols', 'magnesium', 'alcohol', 'malic_acid', 'flavanoids', 'proline']
삭제할 컬럼 : ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'proline']
Trech: 0.1671128098, n=3, acc : 94.4444%


선택된 피처 : [False False False False False False False False False  True  True False
 False]
threshold : 0.17661684473817027
정규화 안 된 threshold : 0.6967754171035246
삭제할 컬럼명:  ['nonflavanoid_phenols', 'proanthocyanins', 'ash', 'alcalinity_of_ash', 'total_phenols', 'magnesium', 'alcohol', 'malic_acid', 'flavanoids', 'proline', 'od280/od315_of_diluted_wines']
삭제할 컬럼 : ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'od280/od315_of_diluted_wines', 'proline']
Trech: 0.1766168447, n=2, acc : 80.5556%


선택된 피처 : [False False False False False False False False False False  True False
 False]
threshold : 0.18855740147157457
정규화 안 된 threshold : 0.7438823983809906
삭제할 컬럼명:  ['nonflavanoid_phenols', 'proanthocyanins', 'ash', 'alcalinity_of_ash', 'total_phenols', 'magnesium', 'alcohol', 'malic_acid', 'flavanoids', 'proline', 'od280/od315_of_diluted_wines', 'color_intensity']
삭제할 컬럼 : ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'od280/od315_of_diluted_wines', 'proline']
Trech: 0.1885574015, n=1, acc : 61.1111%


최대의 acc보장하면서 가장 많이 삭제된 컬럼 리스트 : ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins']
그때 acc : 97.22222222222221
그때 남아있는 컬럼의 갯수 : 4
verbose : 0
"""