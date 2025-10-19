### <<08>>

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))
print(x)    # [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16]
print(y)    # [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16]

# [실습] train_test_split으로 10:3:3으로 나누기

# 1. 데이터
# x_train = x[:10]
# y_train = y[:10]
# print('x_train:',x_train)
# print('y_train:',y_train)

# x_val = x[10:14]
# y_val = y[10:14]
# print('x_val:',x_val)
# print('y_val:',y_val)

# x_test = x[14:]
# y_test = y[14:]
# print('x_test:',x_test)
# print('y_test:',y_test)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    shuffle=True,
    test_size=3/16
)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    shuffle=True,
    test_size=3/13
)

print('x_train :', x_train)
print('x_val :' ,x_val)
print('x_test :' ,x_test)


# 2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='relu'))

# 3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=1,
    verbose=1,
    validation_data=(x_val, y_val)
)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: :', loss)
results= model.predict([17])
print('[17]의 예측값 :', results)