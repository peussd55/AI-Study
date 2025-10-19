### <<38>>

# 54-11 카피

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
datasets = load_digits()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (1797, 64) (1797,)
print(np.unique(y, return_counts=True))     # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, 
    train_size=0.75, 
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

print(np.unique(y_train, return_counts=True))

# 2. 모델
model = Sequential()
model.add(Dense(64, input_shape=(64,), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈려
model.compile(loss='sparse_categorical_crossentropy', 
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
import time
start_time = time.time()
model.fit(x_train, y_train,
          epochs =100,
          validation_split=0.2,
          verbose = 1,
          callbacks=[es],
          )
end_time = time.time()

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss :', results[0])
print('acc :', results[1])


y_pred = model.predict(x_test)
# print(y_pred)
print(y_pred.shape) 
y_pred = np.argmax(y_pred, axis=1)
# print(y_pred)
print(y_pred.shape)

acc = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average='macro')  # 이진분류가 아니라 average 옵션 별도로 줘야한다.
print('걸린시간 :', end_time-start_time)
print('accuracy_score :', acc)
print('f1_score :', f1)


#################################### 결과 ####################################
# 그냥 
# 걸린시간 : 5.310043096542358
# accuracy_score : 0.9777777777777777
# f1_score : 0.9779667908256844

# 디폴트 smote
# 걸린시간 : 4.9515180587768555
# accuracy_score : 0.9733333333333334
# f1_score : 0.9734421088088947

# ros
# 걸린시간 : 4.993192434310913
# accuracy_score : 0.9733333333333334
# f1_score : 0.9734421088088947
