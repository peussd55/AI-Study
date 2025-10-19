### <<31>>

# KFold : 전체 데이터셋을 n등분 한후 각각 한 조각을 test로 지정한 후 train을 한번씩 다돌려보는 방식. 
# 과적합이 될수도있지만, 일반방식으로 test데이터셋을 고정해서 손실되는 학습데이터를 보완할 수 있다는 장점이있음
# 따라서 test data에 중요한 데이터가 들어가있었다면 kfold방식으로 바꿔서 효과를 볼 수있다.

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

# 1. 데이터
path = './_data/kaggle/bike/'
# 맨 앞, 공통 컬럼 datetime을 인덱스 컬럼으로 지정
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)   
print(x)
y = train_csv['count']
print(y)
print(y.shape)

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
r2_score : [0.35352498 0.33077443 0.37605704 0.37007788 0.35513083] 
평균 r2_score : 0.3571
======RandomForestRegressor=====
r2_score : [0.29471966 0.27429453 0.29817731 0.31326766 0.29852586] 
평균 r2_score : 0.2958
"""
