### <<26>>

# 36-6 카피

import numpy as np
import pandas as pd 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Reshape
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

# 스케일링 2(많이 씀) (이미지 스케일링) : 픽셀의 값은 0~255이므로 255만 나누면 0~1로 스케일링(정규화(0~1로 변환))된다.
# 이미지는 연산량이 상당하기때문에 0~1정규화로 부동소수점연산으로 부담을 줄여야한다.
x_train = x_train/255.  # 255. : 부동소수점(실수연산)
x_test = x_test/255.
print(x_train.shape, x_test.shape)      # (60000, 28, 28) (10000, 28, 28)
print(np.max(x_train), np.min(x_train)) # 1.0 0.0
print(np.max(x_test), np.min(x_test))   # 1.0 0.0
# 문제점 : 픽셀값이 0이 많으므로 0에 쏠린 데이터불균형영향을 받는다.

# 원핫인코딩
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

# 2. 모델구성
model = Sequential()
model.add(Dense(100, input_shape=(28, 28)))     # output : (None, 28, 100)
model.add(Reshape(target_shape=(28, 10, 10)))   # output : (None, 28, 10, 10)
# model.add
model.add(Conv2D(64, (3,3), strides=1,))        # input_shape=(28, 28, 1)))    # output : (None, 26, 8, 64)
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(Dropout(0.2))              
model.add(Conv2D(32, (3,3), activation='relu'))         # activation의 역할 : 레이어의 출력을 한정시킴
model.add(Flatten())                                    # Flatten 하는 이유 : 다중분류를 위해선 softmax를 써야하고 softmax를 쓰기위해선 Flatten으로 차원변환을 해야하기때문.
model.add(Dense(units=16, activation='relu'))           # 원하는 평가를 하기위해서 그에 맞는 데이터타입변환 하기위해 Flatten을 사용한다.
model.add(Dropout(0.2))
model.add(Dense(units=16, input_shape=(16,)))
model.add(Dense(units=10, activation='softmax'))
model.summary()
# 입력 데이터가 [−1,1] 범위라도, Conv 레이어 이후의 가중치 연산을 거치면 값이 더 넓은 범위로 확장될 수 있으므로 ReLU를 그대로 사용하는 것이 일반적입니다
# tanh는 성능이랑 속도가 느려서 잘 안쓰인다
"""
[tanh]
장) sigmoid를 개선한 함수 : sigmoid에 비해 향상된 최대 기울기(Max 1), zero-centered 특성
단) tanh은 갈수록 기울기가 0에 수렴(vanishing gradient)해서 가중치 업데이트에 문제가 있다.
"""

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['acc'],
              )

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', 
                   mode='min',
                   patience=50,
                   verbose=1,   # stop이 어느 epoch에서 걸렸는지 출력해줌(modelcheckpoint도 적용가능)
                   restore_best_weights=True,
                   )

####################### mcp 세이브 파일명 만들기 #######################
import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path = './_save/keras36_cnn5/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'                # 04d : 정수 4자리, .4f : 소수점 4자리
filepath = "".join([path, 'k36_', date, '_', filename])     # 구분자를 공백("")으로 하겠다.
# ./_save/keras27_mcp2/k27_0602_1442_{epoch:04d}-{val_loss:.4f}.hdf5
print(filepath)

mcp = ModelCheckpoint(          # 모델+가중치 저장
    monitor = 'val_loss',
    mode = 'auto',
    verbose=1,
    save_best_only=True,
    filepath = filepath,    # filepath가 고정되지 않았기때문에 val_loss갱신될때 마다 신규파일저장
)
start = time.time()
hist = model.fit(x_train, y_train, 
                 epochs=10, 
                 batch_size=64, 
                 verbose=0, 
                 validation_split=0.2,
                 callbacks=[es, mcp],
                 )
end = time.time()
print("걸린 시간 :", round(end-start, 2), "초") # 496.32 초
# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)   # evaluation도 verbose 옵션사용가능
print('loss :', loss[0])
print('acc :', loss[1])

y_pred = model.predict(x_test)
print(y_pred.shape) # (10000, 10)
print(y_test.shape) # (10000, 10)

# argmax이용하여 최대값있는 인덱스만 벡터(시리즈)로 반환후 얼마나 일치하는지 계산
y_pred = np.argmax(y_pred, axis=1)      # axis = 1 : 행 방향
print(y_pred)           # [1 6 1 ... 1 1 6]
y_test = y_test.values  # nparray로 변환. values : 속성. 값만 반환
y_test = np.argmax(y_test, axis=1)
print(y_test)

acc = accuracy_score(y_test, y_pred)
print("accuracy : ", acc) 

"""
    (1) 스케일링3 + relu : 0.9874
    (2) 스케일링3 + tanh : 0.9739 / 1313.05초 / 200 epochs
    
    (3) 스케일링2 + relu : 0.9878 / 302.37초 / 55 epochs
    (4) 스케일링2 + tanh : 0.9803 / 1162.98초 / 200 epochs
    
    (5) 스케일링1 + tanh : 0.9777 / 826.87초 / 134 epochs
    (6) 스케일링1 + relu : 0.986 / 322.8초 / 59 epochs
    
    :: tanh이 더 느리고 성능도 떨어짐
"""