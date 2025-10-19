### <<19>>

# 36-7 카피

import numpy as np
import pandas as pd 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Input, BatchNormalization
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

# ##### 스케일링 1. MinMaxScaler() : 2차원 이하만 fit가능하기 때문에 2차원으로 reshape하고 적용해야한다.(값과 순서는 변함 없기때문에 데이터 소실X)
# x_train = x_train.reshape(60000, 28*28) # (60000, 784)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
# print(x_train.shape, x_test.shape)  # (60000, 784) (10000, 784)

# print(np.max(x_train), np.min(x_train)) # 255 0
# print(np.max(x_test), np.min(x_test))   # 255 0

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# print(x_train.shape, x_test.shape)  # (60000, 784) (10000, 784)

# print(np.max(x_train), np.min(x_train)) # 1.0 0.0
# print(np.max(x_test), np.min(x_test))   # 24.0 0.0 : 1.0이 아니고 24.0 이 나오는 이유 : x_test는 fit을 하지 않고 transform만 했기 때문

# # 스케일링 2(많이 씀) (이미지 스케일링) : 픽셀의 값은 0~255이므로 255만 나누면 0~1로 스케일링(정규화(0~1로 변환))된다.
# # 이미지는 연산량이 상당하기때문에 0~1정규화로 부동소수점연산으로 부담을 줄여야한다.
# x_train = x_train/255.  # 255. : 부동소수점(실수연산)
# x_test = x_test/255.
# print(x_train.shape, x_test.shape)      # (60000, 28, 28) (10000, 28, 28)
# print(np.max(x_train), np.min(x_train)) # 1.0 0.0
# print(np.max(x_test), np.min(x_test))   # 1.0 0.0
# # 문제점 : 픽셀값이 0이 많으므로 0에 쏠린 데이터불균형영향을 받는다.

# 스케일링 3(많이 씀) : 활성화함수로 음수를 허용하지 않는 함수(relu, sigmoid 등)를 사용하면 데이터 소실문제 발생 (그러나 실제로는 relu를 같이 써서 성능 더 좋을 수도있다.)
# 그럼에도 스케이링 3을 쓰는 이유 : Batch Normalization(이름은 normalization(정규화)이지만 실제로는 표준정규화하는 방식임)과 잘 맞는다
# why? 표준정규화는 데이터가 중앙값이 0이라서 -1 ~ 1 입력데이터와 잘 맞는다. 통상적으로 이미지 모델은 데이터가 0주변(중앙값 0)에 모이는 방식을 많이 쓴다. 
x_train = (x_train - 127.5) / 127.5     # 범위 : -1 ~ 1
x_test = (x_test - 127.5) / 127.5
print(np.max(x_train), np.min(x_train)) # 1.0 -1.0
print(np.max(x_test), np.min(x_test))   # 1.0 -1.0

#exit()

#  x reshape -> (60000, 28, 28, 1) : (60000, 784)같은 2차원 스케일링 데이터도 4차원 reshape가능하다. 값과 순서만 유지되면된다.
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
print(x_train.shape, x_test.shape)  # (60000, 28, 28, 1) (10000, 28, 28, 1)

# 원핫인코딩
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

# 2. 모델구성
# model = Sequential()
# model.add(Conv2D(64, (3,3), strides=1, input_shape=(28, 28, 1)))    # (28, 28, 1) 에서의 채널(1 또는 3)은 여기서만 쓰이고 다음 레이어부터는 filters를 입력받는다. 
# model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
# model.add(Dropout(0.2))              
# model.add(Conv2D(32, (3,3), activation='relu'))         # activation의 역할 : 레이어의 출력을 한정시킴
# model.add(Flatten())                                    # Flatten 하는 이유 : 다중분류를 위해선 softmax를 써야하고 softmax를 쓰기위해선 faltten으로 차원변환을 해야하기때문.
# model.add(Dense(units=16, activation='relu'))           # 원하는 평가를 하기위해서 그에 맞는 데이터타입변환 하기위해 Flatten을 사용한다.
# model.add(Dropout(0.2))
# model.add(Dense(units=16, input_shape=(16,)))
# model.add(Dense(units=10, activation='softmax'))
input1 = Input(shape=(28, 28, 1))
conv1 = Conv2D(64, (3,3), strides=1)(input1)
conv2 = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(conv1)     # tanh : 출력을 -1 ~ 1로 한정
batnorm1 = BatchNormalization()(conv2)
drop1 = Dropout(0.2)(batnorm1)
conv3 = Conv2D(32, (3,3), activation='relu')(drop1)
flat1 = Flatten()(conv3)
dense1 = Dense(16, activation='relu')(flat1)
batnorm2 = BatchNormalization()(dense1)
drop2 = Dropout(0.2)(batnorm2)
dense2 = Dense(16)(drop2)
output1 = Dense(10, activation='softmax')(dense2)
model = Model(inputs=input1, outputs=output1)

model.summary()

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
    (1) 스케일링2 + relu : 0.9757 / 883.4 초 / 131 epochs
    (2) 스케일링3 + relu : 0.9835 / 332.79 초 / 55 epochs
"""