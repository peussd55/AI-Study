### <<31>>

# KFold : 전체 데이터셋을 n등분 한후 각각 한 조각을 test로 지정한 후 train을 한번씩 다돌려보는 방식. 
# 과적합이 될수도있지만, 일반방식으로 test데이터셋을 고정해서 손실되는 학습데이터를 보완할 수 있다는 장점이있음
# 따라서 test data에 중요한 데이터가 들어가있었다면 kfold방식으로 바꿔서 효과를 볼 수있다.

import numpy as np
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor

# 1. 데이터
x, y = fetch_california_housing(return_X_y=True)

n_split = 5
kfold = KFold(n_splits = n_split, shuffle=True, random_state=333)   # shuffle=False : 데이터를 앞에서 순서로 짜름. y라벨이 000 11 222 로 될 수있으므로 shuffle=True 필요

# 2. 모델
model = HistGradientBoostingRegressor()
# model = RandomForestRegressor()

# 3. 훈련
scores = cross_val_score(model, x, y, cv=kfold)     # cross_val_score : fit까지 포함된 함수
print('r2_score :', scores, '\n평균 r2_score :', round(np.mean(scores),4))

# r2_score : [0.82844582 0.84063748 0.82240123 0.84105292 0.84572463] 
# 평균 r2_score : 0.8357

# r2_score : [0.80165251 0.80840463 0.79580501 0.81847315 0.8227951 ] 
# 평균 r2_score : 0.8094

