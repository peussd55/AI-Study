### <<05>>

# R2 : 회귀분석에서의 결정계수 (분류에서의 정확도에 대응)

import numpy as np 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
from sklearn.model_selection import train_test_split

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])  # (20,)
y = np.array([1,2,4,3,5,7,9,3,8,12,13, 8,14,15, 9, 6,17,23,21,20])  # (20,)

x_train, x_test, y_train, y_test = train_test_split(
        x, y, 
        test_size=0.3,
        random_state=42
)
print('x_test :', x_test)
print('y_test :', y_test)

# 2. 모델구성
model = Sequential()
model.add(Dense(2, input_dim=1))  
model.add(Dense(4))    
model.add(Dense(6))    
model.add(Dense(8))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))    
model.add(Dense(2))    
model.add(Dense(1)) 

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)     # (6,) -> (6,1)

print('loss :', loss)
print('x_test :', x_test)
print('x_test의 예측값 :', results)

from sklearn.metrics import r2_score, mean_squared_error

# def RMSE(y_test, y_predict):
#     # mean_squared_error : mse를 계산해주는 함수
#     return np.sqrt(mean_squared_error(y_test, y_predict))

# rmse = RMSE(y_test, results)
# print('RMSE :', rmse)

# r2_score는 컬럼이 많아질수록 높게 나오는 경향(단점)이 있다.

r2 = r2_score(y_test, results)
print('r2 스코어 :', r2)