### <<33>>

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import  LinearSVC, SVC
import time
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.utils import  all_estimators
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# GridSearch : 모든 경우의 수를 탐색하는 방법

# 1. 데이터
x = datasets = load_diabetes()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8, 
    # stratify=y,   # 분류데이터에서만 사용가능
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333, )

parameters = [
    {'n_estimators':[100,500], 'max_depth':[6,10,12], 'learning_rate':[0.1, 0.01, 0.001]},  # 18
    {'max_depth':[6,8,10,12], 'learning_rate':[0.1, 0.01, 0.001]},  # 12
    {'min_child_weight':[2,3,5,10], 'learning_rate':[0.1, 0.01, 0.001]},    # 12
]

# 2. 모델
# 모델불러오기
import joblib
path = './_save/m15_cv_results/'
model = joblib.load(path + 'm15_best_model_03_diabetes.joblib') 

# 3. 훈련

# 4. 평가, 예측     # XGBoost 파라미터만쓸수있다.
print('model.score :', model.score(x_test, y_test))
# model.score : 0.439408774401704

y_pred = model.predict(x_test)
print('accuracy_score :', r2_score(y_test, y_pred))
# accuracy_score : 0.439408774401704

# path = './_save/m15_cv_results/'
# joblib.dump(model.best_estimator_, path + 'm15_best_model.joblib')
# 모델은 GridSearch로 돌렸지만 저장은 XGBoost로 저장된다.

print(model)    # 파라미터확인

print(type(model))
# <class 'xgboost.sklearn.XGBRegressor'>