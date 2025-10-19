### <<33>>

import numpy as np
import pandas as pd
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
path = './_data/dacon/따릉이/'          
train_csv =  pd.read_csv(path + 'train.csv', index_col=0)    
test_csv = pd.read_csv(path + 'test.csv', index_col=0)  
submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)
x = train_csv.drop(['count'], axis=1) 
y = train_csv['count'] 

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
xgb = XGBRegressor()
model = GridSearchCV(xgb, param_grid=parameters, cv=kfold, 
                     verbose = 1,
                     refit=True,    # best 파라미터로 전체 훈련 데이터(x_train, y_train)를 다시 fit 1번 (디폴트)
                     n_jobs=-1,     # cpu를 풀 가동(모든 쓰레드를 가동)
                     )  # 12+18+12+1

# 3. 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()

print('최적의 매개변수 :', model.best_estimator_)

print('최적의 파라미터 :', model.best_params_)

# 4. 평가, 예측
print('best_score :', model.best_score_)

print('model.score :', model.score(x_test, y_test))

y_pred = model.predict(x_test)
print('r2_score :', r2_score(y_test, y_pred))

print('걸린시간 :', round(end - start), '초')
"""
최적의 파라미터 : {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 500}
best_score : 0.7831301943268064
model.score : 0.7990606612608195
r2_score : 0.7990606612608195
걸린시간 : 22 초
"""

# 결과 csv출력 생성
import pandas as pd
print(pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True))   # rank_test_score를 기준으로 오름차순 정렬
print(pd.DataFrame(model.cv_results_).columns)
path = './_save/m15_cv_results/'
pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True).to_csv(path + 'm15_gs_cv_results_04_ddareung.csv')