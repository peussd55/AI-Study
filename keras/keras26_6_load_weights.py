
### <<13>>

# 26-4 카피

import sklearn as sk
print(sk.__version__)
from sklearn.datasets import load_boston
import numpy as np
from tensorflow.keras.models import Sequential, load_model
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

# 2. 모델구성 : 가중치 save할때의 모델구성과 같아야함. 모델 형태가 맞아야 가중치가 제대로 들어감
model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(11))
model.add(Dense(12))
model.add(Dense(13))
model.add(Dense(1))

path = './_save/keras26/'
# model.load_weights(path + 'keras26_5_save1.h5') # 훈련안된 초기가중치
model.load_weights(path + 'keras26_5_save2.h5')   # 훈련된 가중치

model.summary()

# model.save(path + 'keras26_1_save.h5')

# exit()

# 3. 컴파일, 훈련 : 가중치만 불러올땐 훈려(fit)은 없어도 되지만(이미 확정된 가중치를 불러오는거기 때문에 훈련을 안하는게 맞긴하다.) 컴파일은 구성되어있어야한다.
model.compile(loss='mse', optimizer='adam')

"""
from tensorflow.keras.callbacks import EarlyStopping
# val_loss를 기준으로 최소값을 찾을건데 patience를 10으로 한다. 10번 내에 최소값을 찾으면 loss를 갱신하고 10번을 더 훈련시킨다. 10번을 훈련을 더 하는동안 최소값이 안 나타나면 훈련을 종료한다.
es = EarlyStopping( 
    monitor = 'val_loss',       # 기준을 val_loss로
    mode = 'min',               # 최대값 : max, 알아서 찾아줘 : auto
    patience=100,                # 참는 횟수는 10번
    restore_best_weights=True,  # 가장 최소지점을 save할것인지. default = False. False가 성능이 더 잘나오면 모델이 과적합됐을 수 있음. False 마지막 종료시점의 가중치를 저장한다.
)

hist = model.fit(x_train, y_train, 
                 epochs=1000, 
                 batch_size=24, 
                 verbose=3, 
                 validation_split=0.2,
                 callbacks=[es],
                 )
"""
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

# RMSE : 4.3280565668586295
# r2 스코어 : 0.7846818078075273
