### <<31>>

# KFold : 전체 데이터셋을 n등분 한후 각각 한 조각을 test로 지정한 후 train을 한번씩 다돌려보는 방식. 
# 과적합이 될수도있지만, 일반방식으로 test데이터셋을 고정해서 손실되는 학습데이터를 보완할 수 있다는 장점이있음
# 따라서 test data에 중요한 데이터가 들어가있었다면 kfold방식으로 바꿔서 효과를 볼 수있다.

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import warnings     # 경고무시
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# 1. 데이터
datasets = load_breast_cancer()
x, y = load_breast_cancer(return_X_y=True)
print(x.shape, y.shape) # (569, 30) (569,)

n_split = 5
# kfold = KFold(n_splits = n_split, shuffle=True, random_state=333) 
kfold = StratifiedKFold(n_splits = n_split, shuffle=True, random_state=333) # StratifiedKFold : 분류모델에만 적용

# 2. 모델
# model_list = [
#             HistGradientBoostingRegressor(),
#             RandomForestRegressor(),
#             ]     # 회귀 : 분류문제를 회귀모델로 적용할 수도있지만 분류모델보다 성능은 보장할 수없다.
model_list = [DecisionTreeClassifier(), RandomForestClassifier()]    # 분류

# 3. 훈련
for model in model_list:
    print(f"======{type(model).__name__}=====")      
    scores = cross_val_score(model, x, y, cv=kfold)     
    print('acc :', scores, '\n평균 acc :', round(np.mean(scores),4))
    
"""
======DecisionTreeClassifier=====
acc : [0.92105263 0.92105263 0.92105263 0.93859649 0.9380531 ] 
평균 acc : 0.928
======RandomForestClassifier=====
acc : [0.96491228 0.95614035 0.95614035 0.98245614 0.9380531 ] 
평균 acc : 0.9595
"""
