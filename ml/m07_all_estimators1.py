### <<31>>

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.utils import  all_estimators
import sklearn as sk
print(sk.__version__)   # 1.6.1

# 1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, test_size=0.2,
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
# model = RandomForestRegressor()
allAlgorithms = all_estimators(type_filter='regressor')     # 모든 회귀모델
print('allAlgorithms :', allAlgorithms)
print('모델의 갯수 :', len(allAlgorithms))  # 55
print(type(allAlgorithms))  # <class 'list'>

max_name = "default model"
max_score = 0
for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        # 3. 훈련
        model.fit(x_train, y_train)     # fit에서 필수 파라미터 빠져서 에러일으키는 모델은 예외처리로 이동
        
        # 4. 평가, 예측
        results = model.score(x_test, y_test)   # 기본 평가지표 : 회귀 : r2_score
        print(name, '의 정답률 :', results)
        if results > max_score:
            max_score = results
            max_name = name
    except:
        print('예외처리된 모델 :', name)
        
print("===========================================================")
print("최고성능 모델 :", max_name, max_score)
print("===========================================================")
"""
ARDRegression 의 정답률 : 0.5986791915523428
AdaBoostRegressor 의 정답률 : 0.3726309491002264
BaggingRegressor 의 정답률 : 0.7770766970452605
BayesianRidge 의 정답률 : 0.5985153953397588
예외처리된 모델 : CCA
DecisionTreeRegressor 의 정답률 : 0.5900326774343962
DummyRegressor 의 정답률 : -0.0010660727393365654
ElasticNet 의 정답률 : 0.14627258235198126
ElasticNetCV 의 정답률 : 0.5980937620269564
ExtraTreeRegressor 의 정답률 : 0.5269210750807781
ExtraTreesRegressor 의 정답률 : 0.7999026922194576
GammaRegressor 의 정답률 : 0.29486255950095597
GaussianProcessRegressor 의 정답률 : -85.96608645360213
GradientBoostingRegressor 의 정답률 : 0.7782542019993143
HistGradientBoostingRegressor 의 정답률 : 0.8264862885230226
HuberRegressor 의 정답률 : 0.5367504110521399
예외처리된 모델 : IsotonicRegression
KNeighborsRegressor 의 정답률 : 0.6760040682925643
KernelRidge 의 정답률 : -1.1752443401609494
Lars 의 정답률 : 0.5985244794247394
LarsCV 의 정답률 : 0.59841544104116
Lasso 의 정답률 : -0.0010660727393365654
LassoCV 의 정답률 : 0.5983293165837233
LassoLars 의 정답률 : -0.0010660727393365654
LassoLarsCV 의 정답률 : 0.59841544104116
LassoLarsIC 의 정답률 : 0.5985244794247394
LinearRegression 의 정답률 : 0.5985244794247393
LinearSVR 의 정답률 : -0.6411778924458362
MLPRegressor 의 정답률 : 0.6783806872874278
예외처리된 모델 : MultiOutputRegressor
예외처리된 모델 : MultiTaskElasticNet
예외처리된 모델 : MultiTaskElasticNetCV
예외처리된 모델 : MultiTaskLasso
예외처리된 모델 : MultiTaskLassoCV
NuSVR 의 정답률 : 0.6684287485246247
OrthogonalMatchingPursuit 의 정답률 : 0.46116651181193535
OrthogonalMatchingPursuitCV 의 정답률 : 0.46116651181193535
예외처리된 모델 : PLSCanonical
PLSRegression 의 정답률 : 0.5129282286025278
PassiveAggressiveRegressor 의 정답률 : -2.5332590736544414
PoissonRegressor 의 정답률 : 0.39298556687869635
QuantileRegressor 의 정답률 : -0.04197299297850554
RANSACRegressor 의 정답률 : -0.2808952399830771
예외처리된 모델 : RadiusNeighborsRegressor
RandomForestRegressor 의 정답률 : 0.8002925974688015
예외처리된 모델 : RegressorChain
Ridge 의 정답률 : 0.5985108864106219
RidgeCV 의 정답률 : 0.5985232136984577
SGDRegressor 의 정답률 : -6.238750882982826e+21
SVR 의 정답률 : 0.666530403479588
예외처리된 모델 : StackingRegressor
TheilSenRegressor 의 정답률 : -4.109312193968673
TransformedTargetRegressor 의 정답률 : 0.5985244794247393
TweedieRegressor 의 정답률 : 0.3398871596066627
예외처리된 모델 : VotingRegressor
===========================================================
최고성능 모델 : HistGradientBoostingRegressor 0.8237808001967954
===========================================================
"""