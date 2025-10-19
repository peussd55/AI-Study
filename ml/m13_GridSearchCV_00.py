### <<32>>

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import  LinearSVC, SVC
import time
# GridSearch : 모든 경우의 수를 탐색하는 방법

# 1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8,
    stratify=y
)

n_split = 5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333,)

parameters = [
    {"C":[1,10,100,1000], "kernel":['linear', 'sigmoid'], "degree":[3,4,5]},                    # 4*2*3=24번
    {"C":[1,10,100], "kernel":['rbf'], 'gamma':[0.001, 0.0001]},                                # 3*1*2=6번
    {"C":[1,10,100,1000], "kernel":['sigmoid'], "gamma":[0.01,0.001,0.0001], "degree":[3,4]}    # 4*1*3*2=24번
]   # 총 24+6+24=54번

# 2. 모델 (SVC(C=10, degree=3, kernel='linear'))
model = GridSearchCV(SVC(), param_grid=parameters, cv=kfold, 
                     verbose = 1,
                     refit=True,    # best 파라미터로 전체 훈련 데이터(x_train, y_train)를 다시 fit 1번 (디폴트)
                     n_jobs=-1,     # cpu를 풀 가동(모든 쓰레드를 가동)
                     )  # 54 * 5 + 1 = 271번 훈련

# Fitting 5 folds for each of 54 candidates, totalling 210 fits
# -> Fold 5번으로 교차검증 210번을 수행한다는 의미

# 3. 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()

print('최적의 매개변수 :', model.best_estimator_)
# 최적의 매개변수 : SVC(C=10, kernel='linear')

print('최적의 파라미터 :', model.best_params_)
# 최적의 파라미터 : {'C': 10, 'degree': 3, 'kernel': 'linear'}

# 4. 평가, 예측
print('best_score :', model.best_score_)                # (x_train)에서의 교차검증 평균 성능
# best_score : 0.9916666666666668

print('model.score :', model.score(x_test, y_test))     # 실제 테스트 데이터에서의 모델 성능
# model.score : 0.9

y_pred = model.predict(x_test)
print('accuracy_score :', accuracy_score(y_test, y_pred))
# accuracy_score : 0.9

print('걸린시간 :', round(end - start), '초')
# 걸린시간 : 2 초
