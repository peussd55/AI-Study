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
path = './_data/kaggle/bike/'
# 맨 앞, 공통 컬럼 datetime을 인덱스 컬럼으로 지정
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)   
print(x)
y = train_csv['count']
print(y)
print(y.shape)

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
print("\n")

######### threshhold 변형 ######
# 1. 기본방식
aaa = model.get_booster().get_score(importance_type='gain')
print(aaa)
# skelarn 데이터셋은 넘파이라 컬럼명이 없어서 f1, f2 .. 이런식으로 컬럼명을 임의로 부여했지만 여기선 csv파일을 import 해왔기때문에 실제 컬럼명이 추출됨
model_gain = []
total_sum = 0
for i in aaa:
    model_gain.append(aaa[i])
    total_sum += aaa[i]
print(total_sum) 
print(model_gain)  
model_gain_normalized = [x / total_sum for x in model_gain]
# SelectModel이 threshold 계산할때 1이하 부동소수점으로 계산하므로 정규화 해줘야한다.
print(model_gain_normalized)
threshold = np.sort(model_gain_normalized)
print(threshold)

# 2. 선생님 방식
score_dict = model.get_booster().get_score(importance_type='gain')
print(score_dict)

total = sum(score_dict.values())
print(total)

# score_list = [score_dict.get(f"f{i}", 0) / total for i in range(x.shape[1])]
score_list = [score_dict.get(key, 0) / total for key in score_dict.keys()]
# 딕셔너리가 컬럼명을 직접가지기때문에 이렇게 추출해야함

print(score_list)

threshold = np.sort(score_list)
# threshold[-1] = threshold[-1] - 0.00000002

threshold = threshold - 0.00000002
# 부동소수점 8번째 자리에서 threshold와 select_model의 임계값이 일치하지않아 제대로 매칭시키지 못하는 문제 발생
# -> threshold값을 조금씩 줄여서 select_model에서 매칭시키도록 하기위해 적용함. (내 환경에서만 이러는지 모르겠음)

print(threshold)

#### 컬럼명매칭 ####
feature_names = x.columns.tolist()
print(feature_names)

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
(10886,)
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ 그냥 모델훈련 시작 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ 그냥 모델훈련 끝 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ


========= XGBRegressor ========
r2_score : 0.32508665323257446
[0.13232872 0.08218841 0.10296488 0.08582555 0.11200627 0.2504472
 0.14085309 0.09338577]
[0.08218841 0.08582555 0.09338577 0.10296488 0.11200627 0.13232872
 0.14085309 0.2504472 ]


{'season': 79276.8984375, 'holiday': 49238.3046875, 'workingday': 61685.29296875, 'weather': 51417.28125, 'temp': 67101.90625, 'atemp': 150040.578125, 'humidity': 84383.765625, 'windspeed': 55946.5390625}
599090.56640625
[79276.8984375, 49238.3046875, 61685.29296875, 51417.28125, 67101.90625, 150040.578125, 84383.765625, 55946.5390625]
[0.13232873772834783, 0.08218841599003072, 0.10296488782786227, 0.08582555649045785, 0.11200628087423004, 0.2504472387623206, 0.14085310361535291, 0.09338577871139775]
[0.08218842 0.08582556 0.09338578 0.10296489 0.11200628 0.13232874
 0.1408531  0.25044724]
{'season': 79276.8984375, 'holiday': 49238.3046875, 'workingday': 61685.29296875, 'weather': 51417.28125, 'temp': 67101.90625, 'atemp': 150040.578125, 'humidity': 84383.765625, 'windspeed': 55946.5390625}
599090.56640625
[0.13232873772834783, 0.08218841599003072, 0.10296488782786227, 0.08582555649045785, 0.11200628087423004, 0.2504472387623206, 0.14085310361535291, 0.09338577871139775]
[0.0821884  0.08582554 0.09338576 0.10296487 0.11200626 0.13232872
 0.14085308 0.25044722]
['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed']
선택된 피처 : [ True  True  True  True  True  True  True  True]
threshold : 0.08218839599003072
정규화 안 된 threshold : 49238.29270568867
삭제할 컬럼명:  []
삭제할 컬럼 : []
Trech: 0.0821883960, n=8, r2 : 0.3251%


선택된 피처 : [ True False  True  True  True  True  True  True]
threshold : 0.08582553649045785
정규화 안 된 threshold : 51417.26926818867
삭제할 컬럼명:  ['holiday']
삭제할 컬럼 : ['holiday']
Trech: 0.0858255365, n=7, r2 : 0.3287%


선택된 피처 : [ True False  True False  True  True  True  True]
threshold : 0.09338575871139775
정규화 안 된 threshold : 55946.52708068867
삭제할 컬럼명:  ['holiday', 'weather']
삭제할 컬럼 : ['holiday', 'weather']
Trech: 0.0933857587, n=6, r2 : 0.3205%


선택된 피처 : [ True False  True False  True  True  True False]
threshold : 0.10296486782786227
정규화 안 된 threshold : 61685.28098693868
삭제할 컬럼명:  ['holiday', 'weather', 'windspeed']
삭제할 컬럼 : ['holiday', 'weather', 'windspeed']
Trech: 0.1029648678, n=5, r2 : 0.3169%


선택된 피처 : [ True False False False  True  True  True False]
threshold : 0.11200626087423003
정규화 안 된 threshold : 67101.89426818867
삭제할 컬럼명:  ['holiday', 'weather', 'windspeed', 'workingday']
삭제할 컬럼 : ['holiday', 'workingday', 'weather', 'windspeed']
Trech: 0.1120062609, n=4, r2 : 0.3047%


선택된 피처 : [ True False False False False  True  True False]
threshold : 0.13232871772834784
정규화 안 된 threshold : 79276.88645568868
삭제할 컬럼명:  ['holiday', 'weather', 'windspeed', 'workingday', 'temp']
삭제할 컬럼 : ['holiday', 'workingday', 'weather', 'temp', 'windspeed']
Trech: 0.1323287177, n=3, r2 : 0.3076%


선택된 피처 : [False False False False False  True  True False]
threshold : 0.14085308361535293
정규화 안 된 threshold : 84383.75364318867
삭제할 컬럼명:  ['holiday', 'weather', 'windspeed', 'workingday', 'temp', 'season']
삭제할 컬럼 : ['season', 'holiday', 'workingday', 'weather', 'temp', 'windspeed']
Trech: 0.1408530836, n=2, r2 : 0.2646%


선택된 피처 : [False False False False False  True False False]
threshold : 0.2504472187623206
정규화 안 된 threshold : 150040.56614318865
삭제할 컬럼명:  ['holiday', 'weather', 'windspeed', 'workingday', 'temp', 'season', 'humidity']
삭제할 컬럼 : ['season', 'holiday', 'workingday', 'weather', 'temp', 'humidity', 'windspeed']
Trech: 0.2504472188, n=1, r2 : 0.1654%


최대의 r2보장하면서 가장 많이 삭제된 컬럼 리스트 : ['holiday']
그때 r2 : 0.32866472005844116
그때 남아있는 컬럼의 갯수 : 7
verbose : 0
"""