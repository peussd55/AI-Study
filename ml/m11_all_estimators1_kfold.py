### <<32>>

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor
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

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333, )

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
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        print(name, '의 평가 r2_score :', scores, '\n평균 r2_score :', round(np.mean(scores), 4))
        
        # 4. 평가, 예측
        y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)
        r2 = r2_score(y_test, y_pred)
        
        print(name, '의 예측 r2_score :', r2)
        if r2 > max_score:
            max_score = r2
            max_name = name
    except:
        print('예외처리된 모델 :', name)
        
print("===========================================================")
print("최고성능 모델 :", max_name, max_score)
print("===========================================================")
"""
AdaBoostClassifier 의 평가 acc : [0.85772173 0.86105431 0.8584359  0.85813293 0.85404279] 
평균 acc : 0.8579
AdaBoostClassifier 의 예측 acc : 0.8536371072802739
BaggingClassifier 의 평가 acc : [0.84829205 0.85033704 0.84983905 0.84756675 0.84798334] 
평균 acc : 0.8488
BaggingClassifier 의 예측 acc : 0.8430029993637713
BernoulliNB 의 평가 acc : [0.82178293 0.82424449 0.82196554 0.82526037 0.81605756] 
평균 acc : 0.8219
BernoulliNB 의 예측 acc : 0.8214621140970098
CalibratedClassifierCV 의 평가 acc : [0.82341135 0.82632735 0.82639652 0.82942625 0.81855709] 
평균 acc : 0.8248
CalibratedClassifierCV 의 예측 acc : 0.825430969188354
예외처리된 모델 : CategoricalNB
예외처리된 모델 : ClassifierChain
예외처리된 모델 : ComplementNB
DecisionTreeClassifier 의 평가 acc : [0.79853064 0.79534954 0.79844726 0.79803068 0.79564476] 
평균 acc : 0.7972
DecisionTreeClassifier 의 예측 acc : 0.7955282212863938
DummyClassifier 의 평가 acc : [0.78838143 0.78838143 0.78841129 0.78841129 0.78841129] 
평균 acc : 0.7884
DummyClassifier 의 예측 acc : 0.7884085194049747
ExtraTreeClassifier 의 평가 acc : [0.79724305 0.7999697  0.79352395 0.79477372 0.79515243] 
평균 acc : 0.7961
ExtraTreeClassifier 의 예측 acc : 0.7922258914775654
ExtraTreesClassifier 의 평가 acc : [0.8552223  0.85908506 0.8551032  0.85608786 0.85525469] 
평균 acc : 0.8562
ExtraTreesClassifier 의 예측 acc : 0.8533341412427666
예외처리된 모델 : FixedThresholdClassifier
GaussianNB 의 평가 acc : [0.82697114 0.82871317 0.83056239 0.83060027 0.82170044] 
평균 acc : 0.8277
GaussianNB 의 예측 acc : 0.8263398673008756
예외처리된 모델 : GaussianProcessClassifier
GradientBoostingClassifier 의 평가 acc : [0.86249337 0.86669696 0.86597235 0.86563151 0.86157925] 
평균 acc : 0.8645
GradientBoostingClassifier 의 예측 acc : 0.8614536310479596
HistGradientBoostingClassifier 의 평가 acc : [0.86389457 0.8685526  0.86729786 0.86619958 0.86494982] 
평균 acc : 0.8662
HistGradientBoostingClassifier 의 예측 acc : 0.8621807495379767
KNeighborsClassifier 의 평가 acc : [0.84844354 0.85162463 0.84900587 0.84597614 0.84627911] 
평균 acc : 0.8483
KNeighborsClassifier 의 예측 acc : 0.8415790589874875
예외처리된 모델 : LabelPropagation
예외처리된 모델 : LabelSpreading
LinearDiscriminantAnalysis 의 평가 acc : [0.82363857 0.82697114 0.82806287 0.82984283 0.81950388] 
평균 acc : 0.8256
LinearDiscriminantAnalysis 의 예측 acc : 0.8247038506983367
LinearSVC 의 평가 acc : [0.82163145 0.82307051 0.8229502  0.82560121 0.8165499 ] 
평균 acc : 0.822
LinearSVC 의 예측 acc : 0.8230072408882965
LogisticRegression 의 평가 acc : [0.82519124 0.82731197 0.82783564 0.82969135 0.82045067] 
평균 acc : 0.8261
LogisticRegression 의 예측 acc : 0.8257036386221105
LogisticRegressionCV 의 평가 acc : [0.82522911 0.82742559 0.82783564 0.82969135 0.82048854] 
평균 acc : 0.8261
LogisticRegressionCV 의 예측 acc : 0.8258248250371133
MLPClassifier 의 평가 acc : [0.86264485 0.86669696 0.86608597 0.86540428 0.86339708] 
평균 acc : 0.8648
MLPClassifier 의 예측 acc : 0.8600296906716757
예외처리된 모델 : MultiOutputClassifier
예외처리된 모델 : MultinomialNB
NearestCentroid 의 평가 acc : [0.74093009 0.74327804 0.73815565 0.73985988 0.73690589] 
평균 acc : 0.7398
NearestCentroid 의 예측 acc : 0.7370860726512558
예외처리된 모델 : NuSVC
예외처리된 모델 : OneVsOneClassifier
예외처리된 모델 : OneVsRestClassifier
예외처리된 모델 : OutputCodeClassifier
PassiveAggressiveClassifier 의 평가 acc : [0.73691585 0.70264334 0.6780534  0.70501799 0.74557849] 
평균 acc : 0.7136
PassiveAggressiveClassifier 의 예측 acc : 0.7805616990335383
Perceptron 의 평가 acc : [0.72922821 0.78815421 0.771975   0.79844726 0.73016474]
평균 acc : 0.7636
Perceptron 의 예측 acc : 0.7566273820704699
QuadraticDiscriminantAnalysis 의 평가 acc : [0.8318564  0.83594638 0.83790949 0.84074986 0.83325128]
평균 acc : 0.8359
QuadraticDiscriminantAnalysis 의 예측 acc : 0.8351864755960857
RadiusNeighborsClassifier 의 평가 acc : [nan nan nan nan nan]
평균 acc : nan
평균 acc : 0.8209
SVC 의 평가 acc : [0.86018329 0.86408392 0.86301837 0.86423026 0.85923121]
SVC 의 예측 acc : 0.8589087163328991
예외처리된 모델 : SelfTrainingClassifier
예외처리된 모델 : StackingClassifier
예외처리된 모델 : TunedThresholdClassifierCV
예외처리된 모델 : VotingClassifier
===========================================================
최고성능 모델 : HistGradientBoostingClassifier 0.8621807495379767
===========================================================
"""