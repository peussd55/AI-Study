### <<31>>

import numpy as np
from sklearn.datasets import fetch_california_housing, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.utils import  all_estimators
import sklearn as sk
print(sk.__version__)   # 1.6.1

# 1. 데이터
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, test_size=0.2,
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
# model = RandomForestRegressor()
allAlgorithms = all_estimators(type_filter='classifier')     # 모든 분류모델
print('allAlgorithms :', allAlgorithms)
print('모델의 갯수 :', len(allAlgorithms))  # 44
print(type(allAlgorithms))  # <class 'list'>

max_name = "default model"
max_score = 0
for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        # 3. 훈련
        model.fit(x_train, y_train)     # fit에서 필수 파라미터 빠져서 에러일으키는 모델은 예외처리로 이동
        
        # 4. 평가, 예측
        results = model.score(x_test, y_test)   # 기본 평가지표 : 분류 : acc
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
AdaBoostClassifier 의 정답률 : 0.775
BaggingClassifier 의 정답률 : 0.9305555555555556
BernoulliNB 의 정답률 : 0.8666666666666667
CalibratedClassifierCV 의 정답률 : 0.9527777777777777
예외처리된 모델 : CategoricalNB
예외처리된 모델 : ClassifierChain
예외처리된 모델 : ComplementNB
DecisionTreeClassifier 의 정답률 : 0.8416666666666667
DummyClassifier 의 정답률 : 0.08888888888888889
ExtraTreeClassifier 의 정답률 : 0.775
ExtraTreesClassifier 의 정답률 : 0.975
예외처리된 모델 : FixedThresholdClassifier
GaussianNB 의 정답률 : 0.8305555555555556
GaussianProcessClassifier 의 정답률 : 0.95
GradientBoostingClassifier 의 정답률 : 0.9277777777777778
HistGradientBoostingClassifier 의 정답률 : 0.9638888888888889
KNeighborsClassifier 의 정답률 : 0.9194444444444444
LabelPropagation 의 정답률 : 0.9305555555555556
LabelSpreading 의 정답률 : 0.9305555555555556
LinearDiscriminantAnalysis 의 정답률 : 0.9388888888888889
LinearSVC 의 정답률 : 0.9527777777777777
LogisticRegression 의 정답률 : 0.9611111111111111
LogisticRegressionCV 의 정답률 : 0.95
MLPClassifier 의 정답률 : 0.9777777777777777  
예외처리된 모델 : MultiOutputClassifier       
예외처리된 모델 : MultinomialNB
NearestCentroid 의 정답률 : 0.6722222222222223
NuSVC 의 정답률 : 0.8805555555555555
예외처리된 모델 : OneVsOneClassifier
예외처리된 모델 : OneVsRestClassifier
예외처리된 모델 : OutputCodeClassifier
PassiveAggressiveClassifier 의 정답률 : 0.95
Perceptron 의 정답률 : 0.9333333333333333
QuadraticDiscriminantAnalysis 의 정답률 : 0.8888888888888888
예외처리된 모델 : RadiusNeighborsClassifier
RandomForestClassifier 의 정답률 : 0.9694444444444444
RidgeClassifier 의 정답률 : 0.9027777777777778
RidgeClassifierCV 의 정답률 : 0.9027777777777778
SGDClassifier 의 정답률 : 0.9472222222222222
SVC 의 정답률 : 0.9777777777777777
예외처리된 모델 : SelfTrainingClassifier
예외처리된 모델 : StackingClassifier
예외처리된 모델 : TunedThresholdClassifierCV
예외처리된 모델 : VotingClassifier
===========================================================
최고성능 모델 : MLPClassifier 0.977777777777777
===========================================================
"""