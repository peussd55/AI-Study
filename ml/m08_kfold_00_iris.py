### <<31>>

# KFold : 전체 데이터셋을 n등분 한후 각각 한 조각을 test로 지정한 후 train을 한번씩 다돌려보는 방식. 
# 과적합이 될수도있지만, 일반방식으로 test데이터셋을 고정해서 손실되는 학습데이터를 보완할 수 있다는 장점이있음
# 따라서 test data에 중요한 데이터가 들어가있었다면 kfold방식으로 바꿔서 효과를 볼 수있다.

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
x, y = load_iris(return_X_y=True)

n_split = 5
# kfold = KFold(n_splits = n_split, shuffle=True, random_state=333)   # shuffle=False : 데이터를 앞에서 순서로 짜름. y라벨이 000 11 222 로 될 수있으므로 shuffle=True 필요
kfold = StratifiedKFold(n_splits = n_split, shuffle=True, random_state=333) # KFold에 Stratified를 적용해주는 클래스. 분류데이터에서만 쓰이는 클래스(연속데이터에서만 쓰면 효과가없어서 KFold나 다름없다)

# 2. 모델
model = MLPClassifier()

# 3. 훈련
scores = cross_val_score(model, x, y, cv=kfold)     # cross_val_score : fit까지 포함된 함수
print('acc :', scores, '\n평균 acc :', round(np.mean(scores)))

# acc : [0.96666667 1.         0.96666667 0.96666667 0.93333333] 
# 평균 acc : 1

# acc : [0.93333333 1.         0.96666667 1.         1.        ] 
# 평균 acc : 1