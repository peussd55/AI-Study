### <<39>>

# 53-9 카피

import numpy as np
import pandas as pd
import random

from sklearn.datasets import load_wine
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
datasets = load_wine()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape)     # (178, 13) (178,)
print(np.unique(y, return_counts=True))     # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))
# print(pd.value_counts(y))
print(y)

## 데이터 삭제, 라벨이 2인것을 8개만 남기고 지우기 (리스트슬라이싱)
x = x[:-40, :]
y = y[:-40]
print(np.unique(y, return_counts=True))     # (138, 13) (138,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, 
    train_size=0.75, 
    shuffle=True,
    stratify=y,
)

#################### SMOTE 적용 #####################
# RandomOverSampler : 단순복사증폭
# 장점 : 연산량이 낮다
# 단점 : 과적합위험
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

print(np.unique(y_train, return_counts=True))

# 2. 모델
model = Sequential()
model.add(Dense(10, input_shape=(13,)))
model.add(Dense(3, activation='softmax'))

# 3. 컴파일, 훈려
model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam',
              metrics=['acc']
              )

model.fit(x_train, y_train,
          epochs =100,
          validation_split=0.2,
          verbose = 1,
          )

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss :', results[0])
print('acc :', results[1])
# loss : 253.90240478515625
# acc : 0.05714285746216774

y_pred = model.predict(x_test)
print(y_pred)
print(y_pred.shape) # (35, 3)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)
print(y_pred.shape) # (35,)

acc = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average='macro')  # 이진분류가 아니라 average 옵션 별도로 줘야한다.
print('accuracy_score :', acc)
print('f1_score :', f1)


#################################### 결과 ####################################
# 그냥 (2클래스 40개 삭제)
# accuracy_score : 0.8857142857142857
# f1_score : 0.615890083632019

# 디폴트 smote
# accuracy_score : 0.9142857142857143
# f1_score : 0.6354354354354355

# 50000 smote
# accuracy_score : 1.0
# f1_score : 1.0

# ros
# accuracy_score : 0.9714285714285714
# f1_score : 0.9218390804597701
# -> f1_score는 ros가 smote보다 훨씬 잘나온다.