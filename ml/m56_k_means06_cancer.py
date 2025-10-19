### <<38>>

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.cluster import KMeans

seed = 123
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    stratify=y,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
model = KMeans(n_clusters=2, init='k-means++',
               n_init='auto', random_state=333,
               )    # n_cluster=2 : 이진분류 / init='k=means++' : 랜덤값 k-means++ 로 사용 / n_init=10 : 횟수(디폴트 : auto)

y_train_pred = model.fit_predict(x_train)

print(y_train_pred[:20])    
print(y_train[:20])         
# [1 1 1 1 1 1 1 1 1 1 0 1 1 0 1 1 1 1 0 0] : 군집화로 알아낸 분류. 실제 0, 1라벨이 아니라 k-means가 나눈 군집임.
# [1 1 0 0 1 1 1 1 1 1 0 0 1 0 1 1 1 0 0 0] : 실제 라벨
# -> 하는이유? : 라벨이 제공되지 않는 데이터에서 비지도학습으로 라벨을 부여하기위함