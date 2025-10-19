### <<08>>

# 08-1 카피

import numpy as np 
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense 

# 1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10]).T  # 벡터는 transpose를 해도 변화가 없다.
# y = np.array([1,2,3,4,5,6,7,8,9,10]).T  # 벡터는 trnaspose를 해도 변화가 없다.

# print(x)
# print(y)

# print(x.shape)  #(10,)  엄밀히 말하면 (10,1)과 다르지만 (10, 1)처럼 받아들이고 연산
# print(y.shape)  #(10,)  엄밀히 말하면 (10,1)과 다르지만 (10, 1)처럼 받아들이고 연산

# #exit()
# 훈련데이터(70%)
x_train = np.array([1,2,3,4,5,6])
y_train = np.array([1,2,3,4,5,6])
# 훈련-검증데이터
x_val = np.array([7,8])
y_val = np.array([1,2])

# 평가데이터(20%)
x_test = np.array([9,10])
y_test = np.array([9,10])

# 2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))    # input 차원이 1(x컬럼 1)이면서 출력 1개(y컬럼 1)

# 3. 컴파일, 훈련(검증 단계 추가)
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=5000, batch_size=1, 
          validation_data=(x_val, y_val)
          )
# 검증의 필요성 : 훈련이 잘되는지 확인. 과적합 지점 체크포인트 세이브포인트 등을 확인할때 필요.
# val_loss : 검증데이터(validation_data)로 계산한 오차
# loss는 계속 낮아지는데 val_loss는 정체되면 그 지점부터 학습이 제대로 안되고 있는 것이다. 따라서 val_loss로 학습진행상황을 봐야한다.
# 학습검증이 안좋으면 당연히 평가도 안좋게나온다.
# 검증데이터는 가중치갱신에 영향을 안준다.

# 4. 평가, 예측
loss =model.evaluate(x_test, y_test)
results = model.predict([11])   # 벡터를 predict하지만 케라스가 행렬 형태로 변환하여 계산한다.(2차원 출력)
print('loss :', loss)
print('[11]의 예측값 :', results)
# loss : 1.1474370956420898
# [11]의 예측값 : [[9.4482975]]