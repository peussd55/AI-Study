### <<37>>

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
# [0.04990303 0.05100336 0.05491458 0.05768418 0.06528616 0.0768147
#  0.16232552 0.18228628 0.29978216]
print("\n")

######### threshhold 변형 ######
# 1. 기본방식
aaa = model.get_booster().get_score(importance_type='gain')
print(aaa)
# skelarn 데이터셋은 넘파이라 컬럼명이 없어서 f1, f2 .. 이런식으로 컬럼명을 임의로 부여했지만 여기선 csv파일을 import 해왔기때문에 실제 컬럼명이 추출됨
# {'hour': 7416.6376953125, 'hour_bef_temperature': 6604.4990234375, 'hour_bef_precipitation': 12197.1640625, 'hour_bef_windspeed': 2030.392578125, 'hour_bef_humidity': 2075.161376953125, 'hour_bef_visibility': 3125.34130859375, 'hour_bef_ozone': 2656.2822265625, 'hour_bef_pm10': 2234.29638671875, 'hour_bef_pm2.5': 2346.982421875}
model_gain = []
total_sum = 0
for i in aaa:
    model_gain.append(aaa[i])
    total_sum += aaa[i]
print(total_sum) # 40686.757080078125
print(model_gain)   # [7416.6376953125, 6604.4990234375, 12197.1640625, 2030.392578125, 2075.161376953125, 3125.34130859375, 2656.2822265625, 2234.29638671875, 2346.982421875]
# model_gain_normalized = model_gain / total_sum
model_gain_normalized = [x / total_sum for x in model_gain]
# SelectModel이 threshold 계산할때 1이하 부동소수점으로 계산하므로 정규화 해줘야한다.
print(model_gain_normalized)
# [0.1822862825050263, 0.16232552057267127, 0.2997821634812036, 0.049903032923682235, 0.05100336143450488, 0.07681470662413749, 0.06528616230913922, 0.054914585163946414, 0.05768418498568855]
threshold = np.sort(model_gain_normalized)
print(threshold)
# [0.04990303 0.05100336 0.05491459 0.05768418 0.06528616 0.07681471
#  0.16232552 0.18228628 0.29978216]


# 2. 선생님 방식
score_dict = model.get_booster().get_score(importance_type='gain')
print(score_dict)
# {'hour': 7416.6376953125, 'hour_bef_temperature': 6604.4990234375, 'hour_bef_precipitation': 12197.1640625, 'hour_bef_windspeed': 2030.392578125, 'hour_bef_humidity': 2075.161376953125, 'hour_bef_visibility': 3125.34130859375, 'hour_bef_ozone': 2656.2822265625, 'hour_bef_pm10': 2234.29638671875, 'hour_bef_pm2.5': 2346.982421875}

total = sum(score_dict.values())
print(total)    # 40686.757080078125

# score_list = [score_dict.get(f"f{i}", 0) / total for i in range(x.shape[1])]
score_list = [score_dict.get(key, 0) / total for key in score_dict.keys()]
# 딕셔너리가 컬럼명을 직접가지기때문에 이렇게 추출해야함

print(score_list)
# [0.1822862825050263, 0.16232552057267127, 0.2997821634812036, 0.049903032923682235, 0.05100336143450488, 0.07681470662413749, 0.06528616230913922, 0.054914585163946414, 0.05768418498568855]

threshold = np.sort(score_list)
# threshold[-1] = threshold[-1] - 0.00000002

threshold = threshold - 0.00000002
# 부동소수점 8번째 자리에서 threshold와 select_model의 임계값이 일치하지않아 제대로 매칭시키지 못하는 문제 발생
# -> threshold값을 조금씩 줄여서 select_model에서 매칭시키도록 하기위해 적용함. (내 환경에서만 이러는지 모르겠음)

print(threshold)

#### 컬럼명매칭 ####
feature_names = x.columns.tolist()
print(feature_names)
# exit()
# ['hour', 'hour_bef_temperature', 'hour_bef_precipitation', 'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility', 'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5']

score_df = pd.DataFrame({
    # 'feature' : [feature_names[int(f[1:])]   for f in score_dict.keys()],
    'feature' : feature_names,
    'gain' : list(score_dict.values())
}).sort_values(by='gain', ascending=True)   # 오름차순



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
print('최대의 r2보장하면서 가장 많이 삭제된 컬럼 리스트 :', delete_columns)
print('그때 r2 :', max_r2)
print('그때 남아있는 컬럼의 갯수 :', nn)


print(f"verbose : {verbose}")

"""
(442, 10) (442,)
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ 그냥 모델훈련 시작 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ 그냥 모델훈련 끝 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ


========= XGBRegressor ========
r2_score : 0.46896612405522065
[0.04560075 0.04596306 0.13227104 0.11751472 0.08156345 0.09924012
 0.07758132 0.09443878 0.21943946 0.08638725]
[0.04560075 0.04596306 0.07758132 0.08156345 0.08638725 0.09443878
 0.09924012 0.11751472 0.13227104 0.21943946]


{'f0': 957.58837890625, 'f1': 965.19677734375, 'f2': 2777.612548828125, 'f3': 2467.73876953125, 'f4': 1712.7835693359375, 'f5': 2083.98291015625, 'f6': 1629.1611328125, 'f7': 1983.15771484375, 'f8': 4608.09716796875, 'f9': 1814.080322265625}
<built-in function sum>
[957.58837890625, 965.19677734375, 2777.612548828125, 2467.73876953125, 1712.7835693359375, 2083.98291015625, 1629.1611328125, 1983.15771484375, 4608.09716796875, 1814.080322265625]
[0.04560075103059792, 0.04596306608217186, 0.13227104786217989, 0.11751473150340476, 0.08156345548365673, 0.0992401202138647, 0.07758132078729303, 0.09443878309414298, 0.21943947557233126, 0.08638724837035686]       
{'f0': 957.58837890625, 'f1': 965.19677734375, 'f2': 2777.612548828125, 'f3': 2467.73876953125, 'f4': 1712.7835693359375, 'f5': 2083.98291015625, 'f6': 1629.1611328125, 'f7': 1983.15771484375, 'f8': 4608.09716796875, 'f9': 1814.080322265625}
20999.399291992188
[0.04560075103059792, 0.04596306608217186, 0.13227104786217989, 0.11751473150340476, 0.08156345548365673, 0.0992401202138647, 0.07758132078729303, 0.09443878309414298, 0.21943947557233126, 0.08638724837035686]       
[0.04560073 0.04596305 0.0775813  0.08156344 0.08638723 0.09443876
 0.0992401  0.11751471 0.13227103 0.21943946]
['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
선택된 피처 : [ True  True  True  True  True  True  True  True  True  True]
threshold : 0.04560073103059792
정규화 안 된 threshold : 957.5879589182641
삭제할 컬럼명:  []
삭제할 컬럼 : []
Trech: 0.0456007310, n=10, r2 : 0.4690%


선택된 피처 : [False  True  True  True  True  True  True  True  True  True]
threshold : 0.04596304608217186
정규화 안 된 threshold : 965.196357355764
삭제할 컬럼명:  ['age']
삭제할 컬럼 : ['age']
Trech: 0.0459630461, n=9, r2 : 0.4179%


선택된 피처 : [False False  True  True  True  True  True  True  True  True]
threshold : 0.07758130078729303
정규화 안 된 threshold : 1629.1607128245141
삭제할 컬럼명:  ['age', 'sex']
삭제할 컬럼 : ['age', 'sex']
Trech: 0.0775813008, n=8, r2 : 0.5033%


선택된 피처 : [False False  True  True  True  True False  True  True  True]
threshold : 0.08156343548365673
정규화 안 된 threshold : 1712.7831493479516
삭제할 컬럼명:  ['age', 'sex', 's3']
삭제할 컬럼 : ['age', 'sex', 's3']
Trech: 0.0815634355, n=7, r2 : 0.4496%


선택된 피처 : [False False  True  True False  True False  True  True  True]
threshold : 0.08638722837035685
정규화 안 된 threshold : 1814.0799022776391
삭제할 컬럼명:  ['age', 'sex', 's3', 's1']
삭제할 컬럼 : ['age', 'sex', 's1', 's3']
Trech: 0.0863872284, n=6, r2 : 0.4812%


선택된 피처 : [False False  True  True False  True False  True  True False]
threshold : 0.09443876309414298
정규화 안 된 threshold : 1983.1572948557641
삭제할 컬럼명:  ['age', 'sex', 's3', 's1', 's6']
삭제할 컬럼 : ['age', 'sex', 's1', 's3', 's6']
Trech: 0.0944387631, n=5, r2 : 0.4893%


선택된 피처 : [False False  True  True False  True False False  True False]
threshold : 0.0992401002138647
정규화 안 된 threshold : 2083.982490168264
삭제할 컬럼명:  ['age', 'sex', 's3', 's1', 's6', 's4']
삭제할 컬럼 : ['age', 'sex', 's1', 's3', 's4', 's6']
Trech: 0.0992401002, n=4, r2 : 0.4622%


선택된 피처 : [False False  True  True False False False False  True False]
threshold : 0.11751471150340476
정규화 안 된 threshold : 2467.738349543264
삭제할 컬럼명:  ['age', 'sex', 's3', 's1', 's6', 's4', 's2']
삭제할 컬럼 : ['age', 'sex', 's1', 's2', 's3', 's4', 's6']
Trech: 0.1175147115, n=3, r2 : 0.5213%


선택된 피처 : [False False  True False False False False False  True False]
threshold : 0.1322710278621799
정규화 안 된 threshold : 2777.6121288401396
삭제할 컬럼명:  ['age', 'sex', 's3', 's1', 's6', 's4', 's2', 'bp']
삭제할 컬럼 : ['age', 'sex', 'bp', 's1', 's2', 's3', 's4', 's6']
Trech: 0.1322710279, n=2, r2 : 0.5129%


선택된 피처 : [False False False False False False False False  True False]
threshold : 0.21943945557233127
정규화 안 된 threshold : 4608.096747980764
삭제할 컬럼명:  ['age', 'sex', 's3', 's1', 's6', 's4', 's2', 'bp', 'bmi']
삭제할 컬럼 : ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's6']
Trech: 0.2194394556, n=1, r2 : 0.3184%


최대의 r2보장하면서 가장 많이 삭제된 컬럼 리스트 : ['age', 'sex', 's1', 's2', 's3', 's4', 's6']
그때 r2 : 0.521254588437533
그때 남아있는 컬럼의 갯수 : 3
verbose : 0
"""