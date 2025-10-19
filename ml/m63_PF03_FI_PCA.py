### <<41>>

# 52 카피

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
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (20640, 8) (20640,)
feature_names = datasets.feature_names
print(feature_names)    # ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
print(type(feature_names))  # <class 'list'>

# PolynomialFeatures 적용여부
PolynomialFeatures_yn = 1

if PolynomialFeatures_yn ==  1:
  # PolynomialFeatures 적용
  from sklearn.preprocessing import PolynomialFeatures
  pf = PolynomialFeatures(degree=2, include_bias=True) # 회귀데이터라 바이어스추가해봄
  x = pf.fit_transform(x)
  col_names = pf.get_feature_names_out(input_features=feature_names)  # PolynomialFeatures로 생성된 컬럼명 추출
  print(type(col_names))      # <class 'numpy.ndarray'>
  print(col_names)  
  # ['1' 'MedInc' 'HouseAge' 'AveRooms' 'AveBedrms' 'Population' 'AveOccup'
  #  'Latitude' 'Longitude' 'MedInc^2' 'MedInc HouseAge' 'MedInc AveRooms'
  #  'MedInc AveBedrms' 'MedInc Population' 'MedInc AveOccup'
  #  'MedInc Latitude' 'MedInc Longitude' 'HouseAge^2' 'HouseAge AveRooms'
  #  'HouseAge AveBedrms' 'HouseAge Population' 'HouseAge AveOccup'
  #  'HouseAge Latitude' 'HouseAge Longitude' 'AveRooms^2'
  #  'AveRooms AveBedrms' 'AveRooms Population' 'AveRooms AveOccup'
  #  'AveRooms Latitude' 'AveRooms Longitude' 'AveBedrms^2'
  #  'AveBedrms Population' 'AveBedrms AveOccup' 'AveBedrms Latitude'
  #  'AveBedrms Longitude' 'Population^2' 'Population AveOccup'
  #  'Population Latitude' 'Population Longitude' 'AveOccup^2'
  #  'AveOccup Latitude' 'AveOccup Longitude' 'Latitude^2'
  #  'Latitude Longitude' 'Longitude^2']
  # print(x.shape)   # (20640, 45)
  feature_names = col_names

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
# r2_score : 0.8159363442715617

print(model.feature_importances_)   

threshold = np.sort(model.feature_importances_) # default : 오름차순 / 기본중요도(importance_type='weight') 방식으로 추출
print(threshold)     
# [0.         0.         0.         0.         0.         0.
#  0.         0.         0.0060931  0.00687761 0.00753524 0.0080223
#  0.00832582 0.00894856 0.00908208 0.00943227 0.0096259  0.00973663
#  0.00973969 0.00996039 0.01011265 0.01056076 0.01060014 0.0106761
#  0.0106936  0.01123067 0.01168875 0.011936   0.01198031 0.01198275
#  0.01435473 0.01577365 0.01867449 0.01883463 0.02196978 0.02203034
#  0.02928408 0.03135761 0.03152722 0.03744828 0.03987031 0.04034287
#  0.04802623 0.08690465 0.32875976]
print("\n")

######### threshhold 변형 ######
# 1. 기본방식
aaa = model.get_booster().get_score(importance_type='gain')
print(aaa)
# {'f1': 9.669477462768555, 'f2': 0.677950382232666, 'f3': 0.8384106159210205, 'f4': 0.8926036953926086, 'f5': 0.7652400135993958, 'f6': 2.4444758892059326, 'f7': 3.507888078689575, 'f8': 3.25830340385437, 'f10': 4.436184406280518, 'f11': 1.1794286966323853, 'f12': 4.166696071624756, 'f13': 1.1251877546310425, 'f14': 1.049485206604004, 'f15': 1.1750472784042358, 'f16': 36.57957077026367, 'f18': 1.1082464456558228, 'f19': 1.3005534410476685, 'f20': 1.3329936265945435, 'f21': 1.1878796815872192, 'f22': 1.0710293054580688, 'f23': 2.095642328262329, 'f25': 1.249584674835205, 'f26': 0.9263758659362793, 'f27': 3.489015817642212, 'f28': 1.755061149597168, 'f29': 1.5971840620040894, 'f31': 1.3280631303787231, 'f32': 1.3332653045654297, 
# 'f33': 1.0105206966400146, 'f34': 0.9956651926040649, 'f36': 1.0836902856826782, 'f37': 1.1898276805877686, 'f38': 1.083349347114563, 'f40': 5.343655586242676, 'f41': 2.077824115753174, 'f43': 2.4512133598327637, 'f44': 4.48876428604126}
model_gain = []
total_sum = 0
for i in aaa:
    model_gain.append(aaa[i])
    total_sum += aaa[i]
print(total_sum)  # 111.26535511016846
print(model_gain)
# model_gain_normalized = model_gain / total_sum
model_gain_normalized = [x / total_sum for x in model_gain]
# SelectModel이 threshold 계산할때 1이하 부동소수점으로 계산하므로 정규화 해줘야한다.
print(model_gain_normalized)
# [0.08690465646916241, 0.006093095029997425, 0.007535235160044879, 0.008022296738358536, 0.006877612648085289, 
# 0.021969784635887384, 0.03152722673842428, 0.029284078594169663, 0.03987031184943478, 0.010600143193400892, 
# 0.03744827909369576, 0.010112651449473624, 0.009432273015844555, 0.010560765093867473, 0.3287597539597543, 
# 0.009960391035992307, 0.01168875468702667, 0.011980311618784581, 0.010676096619752395, 0.00962590111178451, 
# 0.01883463478984403, 0.01123067170008076, 0.008325824916651153, 0.031357611847709396, 0.015773653423919863,
# 0.01435472938025184, 0.011935998667902984, 0.01198275333094752, 0.009082078564701978, 0.00894856437224521, 
# 0.009739691969793036, 0.010693604306656557, 0.009736627776381011, 0.04802623045557802, 0.018674493185195283, 
# 0.022030337811852714, 0.04034287475734695]

threshold = np.sort(model_gain_normalized)
print(threshold)
# [0.0060931  0.00687761 0.00753524 0.0080223  0.00832582 0.00894856
#  0.00908208 0.00943227 0.0096259  0.00973663 0.00973969 0.00996039
#  0.01011265 0.01056077 0.01060014 0.0106761  0.0106936  0.01123067
#  0.01168875 0.011936   0.01198031 0.01198275 0.01435473 0.01577365
#  0.01867449 0.01883463 0.02196978 0.02203034 0.02928408 0.03135761
#  0.03152723 0.03744828 0.03987031 0.04034287 0.04802623 0.08690466
#  0.32875975]

# 2. 선생님 방식
######
score_dict = model.get_booster().get_score(importance_type='gain')
print(score_dict)
# {'f1': 9.669477462768555, 'f2': 0.677950382232666, 'f3': 0.8384106159210205, 'f4': 0.8926036953926086, 
# 'f5': 0.7652400135993958, 'f6': 2.4444758892059326, 'f7': 3.507888078689575, 'f8': 3.25830340385437, 
# 'f10': 4.436184406280518, 'f11': 1.1794286966323853, 'f12': 4.166696071624756, 'f13': 1.1251877546310425, 
# 'f14': 1.049485206604004, 'f15': 1.1750472784042358, 'f16': 36.57957077026367, 'f18': 1.1082464456558228, 
# 'f19': 1.3005534410476685, 'f20': 1.3329936265945435, 'f21': 1.1878796815872192, 'f22': 1.0710293054580688,
# 'f23': 2.095642328262329, 'f25': 1.249584674835205, 'f26': 0.9263758659362793, 'f27': 3.489015817642212, 
# 'f28': 1.755061149597168, 'f29': 1.5971840620040894, 'f31': 1.3280631303787231, 'f32': 1.3332653045654297, 
# 'f33': 1.0105206966400146, 'f34': 0.9956651926040649, 'f36': 1.0836902856826782, 'f37': 1.1898276805877686, 
# 'f38': 1.083349347114563, 'f40': 5.343655586242676, 'f41': 2.077824115753174, 'f43': 2.4512133598327637, 'f44': 4.48876428604126}
print(len(score_dict.values()))     # 37 : 45(컬럼갯수만큼)이 아닌 이유? feature(=중요도가 0인 feature) 것들은 split에 사용되지 않아 score_dict에 포함안됨.
total = sum(score_dict.values())
print(total)    # 111.26535511016846
score_list = [score_dict.get(f"f{i}", 0) / total for i in range(x.shape[1])]
print(score_list)
# [0.0, 0.08690465646916241, 0.006093095029997425, 0.007535235160044879, 0.008022296738358536, 
# 0.006877612648085289, 0.021969784635887384, 0.03152722673842428, 0.029284078594169663, 0.0, 
# 0.03987031184943478, 0.010600143193400892, 0.03744827909369576, 0.010112651449473624, 
# 0.009432273015844555, 0.010560765093867473, 0.3287597539597543, 0.0, 0.009960391035992307, 
# 0.01168875468702667, 0.011980311618784581, 0.010676096619752395, 0.00962590111178451, 0.01883463478984403,
# 0.0, 0.01123067170008076, 0.008325824916651153, 0.031357611847709396, 0.015773653423919863, 0.01435472938025184, 
# 0.0, 0.011935998667902984, 0.01198275333094752, 0.009082078564701978, 0.00894856437224521, 0.0, 
# 0.009739691969793036, 0.010693604306656557, 0.009736627776381011, 0.0, 0.04802623045557802, 0.018674493185195283, 
# 0.0, 0.022030337811852714, 0.04034287475734695]
print(len(score_list))  # 45
threshold = np.sort(score_list)
# threshold[-1] = threshold[-1] - 0.00000002
print(threshold)
# [0.         0.         0.         0.         0.         0.
#  0.         0.         0.0060931  0.00687761 0.00753524 0.0080223
#  0.00832582 0.00894856 0.00908208 0.00943227 0.0096259  0.00973663
#  0.00973969 0.00996039 0.01011265 0.01056077 0.01060014 0.0106761
#  0.0106936  0.01123067 0.01168875 0.011936   0.01198031 0.01198275
#  0.01435473 0.01577365 0.01867449 0.01883463 0.02196978 0.02203034
#  0.02928408 0.03135761 0.03152723 0.03744828 0.03987031 0.04034287
#  0.04802623 0.08690466 0.32875975]
threshold = threshold - 0.00000002
# 부동소수점 8번째 자리에서 threshold와 select_model의 임계값이 일치하지않아 제대로 매칭시키지 못하는 문제 발생
# -> threshold값을 조금씩 줄여서 select_model에서 매칭시키도록 하기위해 적용함. (내 환경에서만 이러는지 모르겠음)
# print(threshold)

# 컬럼명매칭
print(feature_names)
print(len(feature_names))   # 45
# ['1' 'MedInc' 'HouseAge' 'AveRooms' 'AveBedrms' 'Population' 'AveOccup'
#  'Latitude' 'Longitude' 'MedInc^2' 'MedInc HouseAge' 'MedInc AveRooms'
#  'MedInc AveBedrms' 'MedInc Population' 'MedInc AveOccup'
#  'MedInc Latitude' 'MedInc Longitude' 'HouseAge^2' 'HouseAge AveRooms'
#  'HouseAge AveBedrms' 'HouseAge Population' 'HouseAge AveOccup'
#  'HouseAge Latitude' 'HouseAge Longitude' 'AveRooms^2'
#  'AveRooms AveBedrms' 'AveRooms Population' 'AveRooms AveOccup'
#  'AveRooms Latitude' 'AveRooms Longitude' 'AveBedrms^2'
#  'AveBedrms Population' 'AveBedrms AveOccup' 'AveBedrms Latitude'
#  'AveBedrms Longitude' 'Population^2' 'Population AveOccup'
#  'Population Latitude' 'Population Longitude' 'AveOccup^2'
#  'AveOccup Latitude' 'AveOccup Longitude' 'Latitude^2'
#  'Latitude Longitude' 'Longitude^2']
print(type(feature_names))          # <class 'numpy.ndarray'>

score_df = pd.DataFrame({
    # 'feature' : [feature_names[int(f[1:])]   for f in score_dict.keys()],
    'feature' : feature_names,
    
    # score_dict.values()는 중요도가 0 인 것은 카운트 되지 않아서 컬럼증폭으로 생성된 컬럼 중요도가 0인게 있으면 score_list로 직접 매핑해야한다.
    # 'gain' : list(score_dict.values()),    
    'gain' : score_list,
}).sort_values(by='gain', ascending=True)   # 오름차순
print(score_df)
#                  feature      gain
# 0                      1  0.000000
# 42            Latitude^2  0.000000
# 39            AveOccup^2  0.000000
# 35          Population^2  0.000000
# 9               MedInc^2  0.000000
# 30           AveBedrms^2  0.000000
# 24            AveRooms^2  0.000000
# 17            HouseAge^2  0.000000
# 2               HouseAge  0.006093
# 5             Population  0.006878
# 3               AveRooms  0.007535
# 4              AveBedrms  0.008022
# 26   AveRooms Population  0.008326
# 34   AveBedrms Longitude  0.008949
# 33    AveBedrms Latitude  0.009082
# 14       MedInc AveOccup  0.009432
# 22     HouseAge Latitude  0.009626
# 38  Population Longitude  0.009737
# 36   Population AveOccup  0.009740
# 18     HouseAge AveRooms  0.009960
# 13     MedInc Population  0.010113
# 15       MedInc Latitude  0.010561
# 11       MedInc AveRooms  0.010600
# 21     HouseAge AveOccup  0.010676
# 37   Population Latitude  0.010694
# 25    AveRooms AveBedrms  0.011231
# 19    HouseAge AveBedrms  0.011689
# 31  AveBedrms Population  0.011936
# 20   HouseAge Population  0.011980
# 32    AveBedrms AveOccup  0.011983
# 29    AveRooms Longitude  0.014355
# 28     AveRooms Latitude  0.015774
# 41    AveOccup Longitude  0.018674
# 23    HouseAge Longitude  0.018835
# 6               AveOccup  0.021970
# 43    Latitude Longitude  0.022030
# 8              Longitude  0.029284
# 27     AveRooms AveOccup  0.031358
# 7               Latitude  0.031527
# 12      MedInc AveBedrms  0.037448
# 10       MedInc HouseAge  0.039870
# 44           Longitude^2  0.040343
# 40     AveOccup Latitude  0.048026
# 1                 MedInc  0.086905
# 16      MedInc Longitude  0.328760

from sklearn.feature_selection import SelectFromModel
delete_columns = []
max_r2 = 0
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
    print('threshold :', i)
    print('정규화 안 된 threshold :', i * total)
    print('삭제할 컬럼명: ', score_df[score_df['gain'] < i * total]['feature'].tolist())
    print('삭제할 컬럼 :', not_select_features)
    print('Trech: %.10f, n=%d, r2 : %.4f%%' %(i, select_x_train.shape[1], score))
    if score >= max_r2:
        max_r2 = score
        nn = select_x_train.shape[1]
        delete_columns = not_select_features
    print("\n")
    # print("\n\n")
print(f"PolynomialFeatures 적용여부 : {PolynomialFeatures_yn}")
print('최대의 r2보장하면서 가장 많이 삭제된 컬럼 리스트 :', delete_columns)
print('그때 r2 :', max_r2)
print('그때 남아있는 컬럼의 갯수 :', nn)


print(f"verbose : {verbose}")

"""
PolynomialFeatures 적용여부 : 0
최대의 r2보장하면서 가장 많이 삭제된 컬럼 리스트 : ['age', 'sex', 's1', 's2', 's3', 's4', 's6']
그때 r2 : 0.521254588437533
그때 남아있는 컬럼의 갯수 : 3
verbose : 0

PolynomialFeatures 적용여부 : 1
최대의 r2보장하면서 가장 많이 삭제된 컬럼 리스트 : ['1', 'age', 'sex', 'sex^2', 'sex s2', 's1 s4', 's4^2']
그때 r2 : 0.5402128226652021
그때 남아있는 컬럼의 갯수 : 59
verbose : 0
"""