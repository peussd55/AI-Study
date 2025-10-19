### <<17>>

import numpy as np
import pandas as pd 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

#  x reshape -> (60000, 28, 28, 1)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
print(x_train.shape, x_test.shape)  # (60000, 28, 28, 1) (10000, 28, 28, 1)

# 원핫인코딩
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

# 2. 모델구성
model = Sequential()
model.add(Conv2D(64, (3,3), strides=1, input_shape=(28, 28, 1)))    # (28, 28, 1) 에서의 채널(1 또는 3)은 여기서만 쓰이고 다음 레이어부터는 filters를 입력받는다. 
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(Dropout(0.2))              
model.add(Conv2D(32, (3,3), activation='relu'))         # activation의 역할 : 레이어의 출력을 한정시킴
model.add(Flatten())                                    # Flatten 하는 이유 : 다중분류를 위해선 softmax를 써야하고 softmax를 쓰기위해선 faltten으로 차원변환을 해야하기때문.
model.add(Dense(units=16, activation='relu'))           # 원하는 평가를 하기위해서 그에 맞는 데이터타입변환 하기위해 Flatten을 사용한다.
model.add(Dropout(0.2))
model.add(Dense(units=16, input_shape=(16,)))
model.add(Dense(units=10, activation='softmax'))
model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d (Conv2D)             (None, 26, 26, 64)        640

#  conv2d_1 (Conv2D)           (None, 24, 24, 64)        36928

#  dropout (Dropout)           (None, 24, 24, 64)        0

#  conv2d_2 (Conv2D)           (None, 22, 22, 32)        18464

#  flatten (Flatten)           (None, 15488)             0

#  dense (Dense)               (None, 16)                247824

#  dropout_1 (Dropout)         (None, 16)                0

#  dense_1 (Dense)             (None, 16)                272

#  dense_2 (Dense)             (None, 10)                170

# =================================================================
# Total params: 304,298
# Trainable params: 304,298
# Non-trainable params: 0
# _________________________________________________________________

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
                 epochs=200, 
                 batch_size=64, 
                 verbose=1, 
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