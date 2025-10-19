### <<39>>

# 54-12 카피

import numpy as np
import pandas as pd
import random

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

seed = 123
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

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

# 훈련데이터와 테스트데이터 분할
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.8,
    random_state=8282,
    shuffle=True,
    stratify=y,
)

###################### SMOTE 적용 #####################
from imblearn.over_sampling import SMOTE, RandomOverSampler
import sklearn as sk
print('sklearn version :', sk.__version__)
import imblearn
print('imblearn version :', imblearn.__version__)

# 증폭은 train데이터에만 해야한다.
# 증폭은 분류에서 쓰이는 기법이다.
# smote = SMOTE(random_state=seed, 
#                k_neighbors=5, # 디폴트
#             #   sampling_strategy = 'auto'    # 디폴트 : 가장 많은 클래스 갯만큼으로 맞춰준다.
#             # sampling_strategy = 0.75, # 최대값의 75%만큼 지정(이진분류일때만 사용가능)
#             # sampling_strategy = {0:50, 2:33},
#             # sampling_strategy={0:5000, 1:50000, 2:50000},  #직접지정
#             )
# x_train, y_train, = smote.fit_resample(x_train, y_train)

ros  = RandomOverSampler(random_state=seed, 
              sampling_strategy = 'auto'    # 디폴트 : 가장 많은 클래스 갯만큼으로 맞춰준다.
            # sampling_strategy = 0.75, # 최대값의 75%만큼 지정(이진분류일때만 사용가능)
            # sampling_strategy = {0:50, 2:33},
            # sampling_strategy={0:5000, 1:50000, 2:50000},  #직접지정
            )
x_train, y_train, = ros.fit_resample(x_train, y_train)

print(np.unique(y_train, return_counts=True))   # (array([0, 1], dtype=int64), array([143922, 143922], dtype=int64))

# scikit-learn 분류모델은 입력데이터가 전부 1차원이어야한다.(y 원핫인코딩 X, but x 데이터 스케일러는 적용가능)
# 2. 모델구성 (전부 분류)
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model_list = [LinearSVC, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier]

for index, value in enumerate(model_list):
    if value.__name__ == 'LogisticRegression':
        model = value(solver='liblinear')   # 경고로그안뜨게하려고 적용하는거임
    else:
        model = value()
    model.fit(x_train, y_train)
    results = model.score(x_test, y_test)
    print(f"{value.__name__} :", results)

# 그냥
# LinearSVC : 0.9029
# LogisticRegression : 0.91455
# DecisionTreeClassifier : 0.835775
# RandomForestClassifier : 0.89955

# 디폴트 SMOTE
# LinearSVC : 0.7856
# LogisticRegression : 0.786975
# DecisionTreeClassifier : 0.70015