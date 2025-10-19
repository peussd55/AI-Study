### <<19>>

# CNN -> DNN

import numpy as np
import pandas as pd 
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,) : 흑백이기때문에 채널컬럼 생략
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

# 스케일링 2(많이 씀) (이미지 스케일링) : 픽셀의 값은 0~255이므로 255만 나누면 0~1로 스케일링(정규화(0~1로 변환))된다.
# 이미지는 연산량이 상당하기때문에 0~1정규화로 부동소수점연산으로 부담을 줄여야한다.
x_train = x_train/255.  # 255. : 부동소수점(실수연산)
x_test = x_test/255.
print(x_train.shape, x_test.shape)      # (60000, 28, 28) (10000, 28, 28)
print(np.max(x_train), np.min(x_train)) # 1.0 0.0
print(np.max(x_test), np.min(x_test))   # 1.0 0.0

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

# y 원핫인코딩(sklearn방식)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
# y_train, y_test 매트릭스로 인덱스 : ohe가 입력으로 매트릭스 받기 때문
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
print(y_train.shape, y_test.shape)

y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)
print(y_train.shape, y_test.shape)    # (60000, 10) (10000, 10)

# # y 원핫인코딩(판다스방식:판다스로 반환)
# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)
# print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)


# 2. 모델구성
model = Sequential()
model.add(Dense(256, input_dim=784, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['acc'],
              )

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_acc', 
                   mode='max',
                   patience=30,
                   verbose=1,   # stop이 어느 epoch에서 걸렸는지 출력해줌(modelcheckpoint도 적용가능)
                   restore_best_weights=True,
                   )

# ####################### mcp 세이브 파일명 만들기 #######################
# # import datetime
# # date = datetime.datetime.now()
# # date = date.strftime('%m%d_%H%M')

# path = './_save/keras39/'
# filename = '.hdf5'             
# filepath = "".join([path, 'k39_0', filename])     # 구분자를 공백("")으로 하겠다.

# print(filepath)

# mcp = ModelCheckpoint(          # 모델+가중치 저장
#     monitor = 'val_loss',
#     mode = 'auto',
#     verbose=1,
#     save_best_only=True,
#     filepath = filepath,
# )
start = time.time()
hist = model.fit(x_train, y_train, 
                 epochs=200, 
                 batch_size=64, 
                 verbose=3, 
                 validation_split=0.2,
                 callbacks=[es],
                 )
end = time.time()

## 그래프 그리기
import matplotlib.pyplot as plt
plt.figure(figsize=(18, 5))
# 첫 번째 그래프
plt.subplot(1, 2, 1)  # (행, 열, 위치)
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.title('bank Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.grid()

# 두 번째 그래프
plt.subplot(1, 2, 2)
plt.plot(hist.history['acc'], c='green', label='acc')
plt.plot(hist.history['val_acc'], c='orange', label='val_acc')
plt.title('bank Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.grid()

plt.tight_layout()  # 간격 자동 조정

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
# y_test = y_test.values  # 데이터전처리때 y가 sklearn방식으로 nparray로 반환됐기때문에 여기서 변환 불필요(에러남)
y_test = np.argmax(y_test, axis=1)
print(y_test)

acc = accuracy_score(y_test, y_pred)
print("accuracy : ", acc)
print("걸린 시간 :", round(end-start, 2), "초")
plt.show()
"""
    [cnn 방식]
    accuracy :  0.9052
    걸린 시간 : 294.86 초
    
    accuracy :  0.8927
    걸린 시간 : 84.88 초
    => cnn에 비해 성능은 별차이 없고(살짝낮음), 속도는 더 빠르다.
"""
