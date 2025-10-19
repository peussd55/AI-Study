### <<05>>

import numpy as np 
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense 

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# print(x)
# print(y)

# print(x.shape)  #(10,)  엄밀히 말하면 (10,1)과 다르지만 (10, 1)처럼 받아들이고 연산
# print(y.shape)  #(10,)  엄밀히 말하면 (10,1)과 다르지만 (10, 1)처럼 받아들이고 연산

# #exit()
# 훈련데이터(70%)
# x_train = np.array([1,2,3,4,5,6,7])
# y_train = np.array([1,2,3,4,5,6,7])

# # 평가데이터(30%)
# x_test = np.array([8,9,10])
# y_test = np.array([8,9,10])

# [실습] 넘파이 리스트 슬라이싱
train_size = int(len(x) * 0.7)  # 70% 크기 계산
print(train_size)

x_train = x[:train_size]    # = [0:7] = [:7]
y_train = y[:train_size]
x_test = x[train_size:]     # = [7:10] = [7:]
y_test = y[train_size:]

print(x_train.shape, x_test.shape)  # (7,) (3,)
print(y_train.shape, y_test.shape)  # (7,) (3,)
# 실제로는 학습데이터를 분리할때 골고루 분포된 표본을 추출하는게 더 좋다. 앞에서부터 순서대로 70%를 추출하는 것보다 전체중에서 골고루 70%를 추출하는게 더 좋다.

# 2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(2))    
model.add(Dense(4))    
model.add(Dense(6))    
model.add(Dense(4))    
model.add(Dense(2))    
model.add(Dense(1))    

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

# 4. 평가, 예측
loss =model.evaluate(x_test, y_test)

# 이 코드는 히든레이어없이 epochs를 늘리는것보다 히든레이어를 늘리고 epochs를 줄이는게 더 잘나온다.
results = model.predict([11])
print('loss :', loss)
print('[11]의 예측값 :', results)