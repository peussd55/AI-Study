### <<13>>

# 27-1 카피

import sklearn as sk
print(sk.__version__)
from sklearn.datasets import load_boston
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time

# 1. 데이터
datasets = load_boston()
print(datasets)
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target 
print(x.shape, y.shape) # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    shuffle=True,
    random_state=4325,
)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() # 인스턴스
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))     # 0.0 1.0
print(np.min(x_test), np.max(x_test))       # -0.014613778705636737 1.0106506584043378

# a = 0.1
# b = 0.2
# print(a+b)

# 2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(11))
model.add(Dense(12))
model.add(Dense(13))
model.add(Dense(1))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# val_loss를 기준으로 최소값을 찾을건데 patience를 10으로 한다. 10번 내에 최소값을 찾으면 loss를 갱신하고 10번을 더 훈련시킨다. 10번을 훈련을 더 하는동안 최소값이 안 나타나면 훈련을 종료한다.
es = EarlyStopping( 
    monitor = 'val_loss',       # 기준을 val_loss로
    mode = 'min',               # 최대값 : max, 알아서 찾아줘 : auto
    patience=100,               # 참는 횟수는 10번
    restore_best_weights=True,  # 가장 최소지점을 save할것인지. default = False. False가 성능이 더 잘나오면 모델이 과적합됐을 수 있음. False 마지막 종료시점의 가중치를 저장한다.
)
path = './_save/keras27_mcp/'
mcp = ModelCheckpoint(          # 모델+가중치 저장
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only=True,
    filepath = path + 'keras27_mcp3.hdf5',  # 또는 .h5  # 파일명이 고정되었기때문에 val_loss 갱신될때마다 덮어쓰기
)

hist = model.fit(x_train, y_train, 
                 epochs=1000, 
                 batch_size=24, 
                 verbose=3, 
                 validation_split=0.1,
                 callbacks=[es, mcp],
                 )

path = './_save/keras27_mcp/'
model.save(path + 'keras27_3_save_model.h5')  # 학습가중치 저장

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)   # 훈련이 끝난 모델의 loss를 한번 계산해서  반환
results = model.predict(x_test)

print('loss :', loss)
print('x_test :', x_test)
#print('x_test의 예측값 :', results)

from sklearn.metrics import r2_score, mean_squared_error

def RMSE(y_test, y_predict):
    # mean_squared_error : mse를 계산해주는 함수
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, results)
print('RMSE :', rmse)

r2 = r2_score(y_test, results)
print('r2 스코어 :', r2)

# RMSE : 4.351211286660016
# r2 스코어 : 0.7823717785011496
