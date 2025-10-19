### <<36>>

# 40-1 카피
# DNN에 LDA적용

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)
# print(type(x_train.shape))              # <class 'tuple'>
# print(x_train.shape[0]) # 60000
# print(x_train.shape[1]) # 28
# print(x_train.shape[2]) # 28
# print(x_train.shape[3]) # IndexError: tuple index out of range

# 스케일링
x_train = x_train/255.
x_test = x_test/255.
print(np.max(x_train), np.min(x_train)) # 1.0 0.0
print(np.max(x_test), np.min(x_test))   # 1.0 0.0

x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

# loss='sparse_categorical_crossentropy'으로 대체적용
# # y 원핫인코딩
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder(sparse_output=False)
# # y_train, y_test 매트릭스로 변환 : ohe가 입력으로 매트릭스 받기 때문
# y_train = y_train.reshape(60000, 1)
# y_test = y_test.reshape(-1, 1)  # -1 : 맨 마지막 인덱스 (=10000) : (10000, 1)
# print(y_train.shape, y_test.shape)  # (60000, 1) (10000, 1)

# y_train = ohe.fit_transform(y_train)
# y_test = ohe.fit_transform(y_test)
# print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

# LDA 적용
lda = LinearDiscriminantAnalysis(n_components=9)   # LDA의 n_components는 최대 클래스(y)의 수-1개 까지만 적용가능
x_train = lda.fit_transform(x_train, y_train)       # train데이터 fit_trainsform할땐 y도 들어가야한다.
x_test = lda.transform(x_test)

# 2. 모델구성 // 성능 0.98이상 // 시간체크(cnn 방식이랑 시간 비교)
model = Sequential()
model.add(Dense(128, input_dim=9, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam', 
              metrics=['acc'],
              )

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_acc', 
                   mode='max',
                   patience=50,
                   verbose=1,   # stop이 어느 epoch에서 걸렸는지 출력해줌(modelcheckpoint도 적용가능)
                   restore_best_weights=True,
                   )

start = time.time()
hist = model.fit(x_train, y_train, 
                 epochs=200, 
                 batch_size=64, 
                 verbose=1, 
                 validation_split=0.2,
                 callbacks=[es],
                 )
end = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)   # evaluation도 verbose 옵션사용가능
print('loss :', loss[0])
print('acc :', loss[1])

y_pred = model.predict(x_test)
print(y_pred.shape) # (10000, 10)
print(y_test.shape) # (10000, 10)

y_pred = np.argmax(y_pred, axis=1)      # axis = 1 : 행 방향
print(y_pred)           # [1 6 1 ... 1 1 6]
print(type(y_test))     # <class 'numpy.ndarray'>
#y_test = y_test.values  # 이미 y가 nparray이기 때문에 변환 불필요
# y_test = np.argmax(y_test, axis=1)
print(y_test)

acc = accuracy_score(y_test, y_pred)
print("걸린 시간 :", round(end-start, 2), "초")
print("accuracy : ", acc) 

"""
    걸린 시간 : 118.91 초
    accuracy : 0.9815000295639038
    => cnn에 비해 성능은 별차이 없고(살짝낮음), 속도는 더 빠르다.
    
    (정리) : 속도는 dnn, 성능은 cnn이 좋다.
    
    [이미지를 다루는 모델에서 cnn이 더 성능이 좋은 이유]
    - cnn은 합성곱을 거치면서 가중치가 불필요한 데이터(0, 공백)을 제거해나가고, 지역적인 특징을 추출할 수있다.
    - 학습할때의 컬럼수가 많아질수록 성능차이는 심해진다.
    
    [LDA 적용]
    걸린 시간 : 180.81 초
    accuracy :  0.9214
"""