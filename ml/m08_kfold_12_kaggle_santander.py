### <<31>>

# KFold : 전체 데이터셋을 n등분 한후 각각 한 조각을 test로 지정한 후 train을 한번씩 다돌려보는 방식. 
# 과적합이 될수도있지만, 일반방식으로 test데이터셋을 고정해서 손실되는 학습데이터를 보완할 수 있다는 장점이있음
# 따라서 test data에 중요한 데이터가 들어가있었다면 kfold방식으로 바꿔서 효과를 볼 수있다.

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
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
from sklearn.preprocessing import LabelEncoder

# 1. 데이터
path = './_data/kaggle/santander/'           
# 맨 앞, 공통 컬럼 datetime을 인덱스 컬럼으로 지정
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')
print(train_csv.shape, test_csv.shape, submission_csv.shape)  # (200000, 202) (200000, 201) (200000, 2)
print(train_csv['target'].value_counts())
# 이진분류 (불균형)
# 0    179902
# 1     20098
# train data x로 y 분리
x = train_csv.drop(['target'], axis=1)
print(x)
y = train_csv['target']

n_split = 5
# kfold = KFold(n_splits = n_split, shuffle=True, random_state=333) 
kfold = StratifiedKFold(n_splits = n_split, shuffle=True, random_state=333) # StratifiedKFold : 분류모델에만 적용

# scikit-learn 분류모델은 입력데이터가 전부 1차원이어야한다.(y 원핫인코딩 X, but x 데이터 스케일러 적용가능)
# 2. 모델
# model_list = [
#             HistGradientBoostingRegressor(),
#             RandomForestRegressor(),
#             ]     # 회귀
model_list = [DecisionTreeClassifier(), RandomForestClassifier()]    # 분류

# 3. 훈련
for model in model_list:
    print(f"======{type(model).__name__}=====")      
    scores = cross_val_score(model, x, y, cv=kfold)     
    print('acc :', scores, '\n평균 acc :', round(np.mean(scores),4))
    
"""
======DecisionTreeClassifier=====
acc : [0.87777778 0.86111111 0.83008357 0.82451253 0.87465181] 
평균 acc : 0.8536
======RandomForestClassifier=====
acc : [0.97777778 0.98333333 0.97771588 0.97214485 0.97493036] 
평균 acc : 0.9772
"""
