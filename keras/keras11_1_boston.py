### <<05>>

from sklearn.datasets import load_boston
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from sklearn.model_selection import train_test_split

# 1. 데이터
dataset = load_boston()
print('보스톤데이터셋 :', dataset)
print('보스톤데이터셋의 설명 :', dataset.DESCR)
print('보스톤데이터셋 특성이름:', dataset.feature_names)
#exit()

x = dataset.data
y = dataset.target

print(x)
print(x.shape)  #(506, 13)
print(y)
print(y.shape)  #(506,)

#### 앵그러봐(r2 기준 0.75이상) ####

# 1. 데이터
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.3,
    random_state=163
)
print('x_test :', x_test)
print('y_test :', y_test)

# 2. 모델구성
model = Sequential()
model.add(Dense(16, input_dim=13))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1)) 

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)   # 훈련이 끝난 모델의 loss를 한번 계산해서  반환
results = model.predict(x_test)

print('loss :', loss)
print('x_test :', x_test)
print('x_test의 예측값 :', results)

from sklearn.metrics import r2_score, mean_squared_error

# def RMSE(y_test, y_predict):
#     # mean_squared_error : mse를 계산해주는 함수
#     return np.sqrt(mean_squared_error(y_test, y_predict))

# rmse = RMSE(y_test, results)
# print('RMSE :', rmse)

r2 = r2_score(y_test, results)
print('r2 스코어 :', r2)
# r2 스코어 : 0.7647148633700835