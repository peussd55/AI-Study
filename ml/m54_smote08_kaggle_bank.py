### <<38>>

# KNN : 최근접 이웃 : 라벨을 알아야하기때문에 지도학습이다.
# 라벨 구분하는법 : K 갯수만큼 이웃과의 거리를 각각 계산한다. (유클리드 거리계산)
# K가 1일경우 -> 가장 가까운라벨 1개와 같은 라벨을 가진다.
# K가 3이상일경우 -> 이웃의 갯수가 가장 많은 라벨을 가진다.
# 다수결로 뽑아야하기때문에 K는 홀수로 정한다.(1, 3, 5, ...)

# SMOTE : (0, 1, 2) 라벨 중 2라벨을 증폭시킨다고하면 2군집 샘플 중 하나를 선택하고 거기에서 가까운 k개의 이웃의 거리를 각각 벡터 계산한후,
# 랜덤으로 하나의 이웃을 선택하고, 그 이웃과 샘플간의 직선거리 위에서 랜덤한 지점을 선택(0~1 사이의 값을 곱)한 후 그 지점에 샘플 생성
# 평가지표가 macro F1-Score면 증폭은 거의 필수

import numpy as np
import pandas as pd
import random

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

seed = 123
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. 데이터
path = './_data/kaggle/bank/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

# 인코딩
le_geo = LabelEncoder()
le_gen = LabelEncoder()
train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])
train_csv['Gender'] = le_gen.fit_transform(train_csv['Gender'])
test_csv['Geography'] = le_geo.transform(test_csv['Geography'])
test_csv['Gender'] = le_gen.transform(test_csv['Gender'])

# 불필요 컬럼 제거
train_csv = train_csv.drop(["CustomerId", "Surname"], axis=1)
test_csv = test_csv.drop(["CustomerId", "Surname"], axis=1)

x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']

print(x.shape, y.shape)     # (165034, 10) (165034,)
print(np.unique(y, return_counts=True))     # (array([0, 1], dtype=int64), array([130113,  34921], dtype=int64))
# print(pd.value_counts(y))
# print(y)
# exit()
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, 
    train_size=0.75, 
    shuffle=True,
    stratify=y,
)

#################### SMOTE 적용 #####################
from imblearn.over_sampling import SMOTE
import sklearn as sk
print('sklearn version :', sk.__version__)
import imblearn
print('imblearn version :', imblearn.__version__)

# 증폭은 train데이터에만 해야한다.
# 증폭은 분류에서 쓰이는 기법이다.
smote = SMOTE(random_state=seed, 
               k_neighbors=5, # 디폴트
              sampling_strategy = 'auto'    # 디폴트 : 가장 많은 클래스 갯만큼으로 맞춰준다.
            # sampling_strategy = 0.75, # 증폭되는게 최대값의 75%만큼 지정(이진분류일때만 사용가능)
            # sampling_strategy={0:10000, 1:10000},  #직접지정
            )
x_train, y_train, = smote.fit_resample(x_train, y_train)

print(np.unique(y_train, return_counts=True))

# 2. 모델
model = Sequential()
model.add(Dense(128, input_shape=(10,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈려
model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['acc']
              )
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping( 
    monitor = 'val_loss',      
    mode = 'min',              
    patience=30,              
    restore_best_weights=True, 
)

model.fit(x_train, y_train,
          epochs =100,
          validation_split=0.2,
          verbose = 1,
          callbacks=[es],
          )

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss :', results[0])
print('acc :', results[1])

y_pred = model.predict(x_test)
# print(y_pred)
print(y_pred.shape)
# y_pred = np.argmax(y_pred, axis=1)
# print(y_pred)
# print(y_pred.shape)

acc = accuracy_score(np.round(y_pred), y_test)
f1 = f1_score(np.round(y_pred), y_test)  
print('accuracy_score :', acc)
print('f1_score :', f1)


#################################### 결과 ####################################
# 그냥
# accuracy_score : 0.6687116564417178
# f1_score : 0.6142857142857143

# 디폴트smote
# accuracy_score : 0.6791245546426234
# f1_score : 0.2910843373493976


