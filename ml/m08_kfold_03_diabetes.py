### <<31>>

# KFold : 전체 데이터셋을 n등분 한후 각각 한 조각을 test로 지정한 후 train을 한번씩 다돌려보는 방식. 
# 과적합이 될수도있지만, 일반방식으로 test데이터셋을 고정해서 손실되는 학습데이터를 보완할 수 있다는 장점이있음
# 따라서 test data에 중요한 데이터가 들어가있었다면 kfold방식으로 바꿔서 효과를 볼 수있다.

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)  # (442, 10) (442,)

n_split = 5
kfold = KFold(n_splits = n_split, shuffle=True, random_state=333)

# 2. 모델
model_list = [
            HistGradientBoostingRegressor(),
            RandomForestRegressor(),
            ]

# 3. 훈련
for model in model_list:
    print(f"======{type(model).__name__}=====")      
    scores = cross_val_score(model, x, y, cv=kfold)     
    print('r2_score :', scores, '\n평균 r2_score :', round(np.mean(scores),4))
    
"""
======HistGradientBoostingRegressor=====
HistGradientBoostingRegressor 
r2_score : [0.32241346 0.36710717 0.47591973 0.25545567 0.36580256]
평균 r2_score : 0.3573
======RandomForestRegressor=====
RandomForestRegressor 
r2_score : [0.38796304 0.39113586 0.48263514 0.33821056 0.42178942]
평균 r2_score : 0.4043
"""
