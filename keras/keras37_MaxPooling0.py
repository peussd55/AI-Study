### <<18>>

# 36-6 카피

import numpy as np
import pandas as pd 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
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


# x reshape -> (60000, 28, 28, 1) : (60000, 784)같은 2차원 스케일링 데이터도 4차원 reshape가능하다. 값과 순서만 유지되면된다.
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
model.add(Conv2D(64, (3,3), strides=1, input_shape=(28, 28, 1)))                        # (None, 26, 26, 64)
model.add(MaxPool2D())                                                                  # (None, 13, 13, 64) : 디폴트는 반만큼 줄인다
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))                     # (None, 11, 11, 64)
model.add(MaxPool2D())                                                                  # (None, 5, 5, 64) : 나누어떨어지지 않아서 가장 자리 컬럼 소멸
model.add(Dropout(0.2))              
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Flatten())    
model.add(Dense(units=16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, input_shape=(16,)))
model.add(Dense(units=10, activation='softmax'))
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
# import datetime
# date = datetime.datetime.now()
# date = date.strftime('%m%d_%H%M')

path = './_save/keras37/'
filename = '.hdf5'             
filepath = "".join([path, 'k37_0', filename])     # 구분자를 공백("")으로 하겠다.

print(filepath)

mcp = ModelCheckpoint(          # 모델+가중치 저장
    monitor = 'val_loss',
    mode = 'auto',
    verbose=1,
    save_best_only=True,
    filepath = filepath,
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
print("걸린 시간 :", round(end-start, 2), "초")

"""
    (1) 스케일링2 + relu + MaxPooling
    accuracy :  0.9922
    걸린 시간 : 214.16 초
    
"""
