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
path = './_data/dacon/diabetes/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

# 데이터 전처리
x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']
x = x.replace(0, np.nan).fillna(x.median())  # 0값을 NaN으로 변환 후 중앙값으로 대체

n_split = 5
# kfold = KFold(n_splits = n_split, shuffle=True, random_state=333) 
kfold = StratifiedKFold(n_splits = n_split, shuffle=True, random_state=333) # StratifiedKFold : 분류모델에만 적용

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
acc : [0.6870229  0.75572519 0.63076923 0.70769231 0.68461538] 
평균 acc : 0.6932
======RandomForestClassifier=====
acc : [0.77862595 0.78625954 0.75384615 0.76153846 0.69230769] 
평균 acc : 0.7545
"""
