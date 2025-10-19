### <<05>>

import numpy as np 
from tensorflow.keras.models import Sequential          # import하는게 대문자로 시작하면 클래스 (꼭, 그런건아님)
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split    # import하는게 소문자로 시작하면 함수

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# [실습] : train과 test를 섞어서 sklearn으로 7:3으로 나눠라.
# train_test_split를 이용하여 전체데이터 중 테스트데이터를 랜덤으로 추출한다.
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, shuffle=True, random_state=42
)  
# test_size=0.3 대신 train_size=0.7 로 해도된다.
# 디폴트 값 : train_size=0.75 // test_size=0.25 // shffle=True // random_state=None
# shuffle: 섞을지 말지 결정. False 옵션이면 학습데이터를 앞에서 순서대로 뽑음.
# random_state: 섞을 때 "섞는 순서"를 고정해서 결과를 반복 가능하게 만듦. None이면 훈련때마다 학습(테스트)데이터 바뀜. shuffle이 False면 쓸모없는 옵션
# 데이터가 많아질수록 shuffle을 해야한다. 실제 손실계산과 훈련은 부동소수점(실수)단위로 진행되는데, 만약 훈련범위 밖 데이터가 비선형적으로 분포될경우 뒷부분은 오차가 커질 수있다. 
# shuffle이 되야 전체 데이터중 훈련데이터가 커버하는 범위가 넓어질 확률이 높고 훈련의 범위가 넓어야 넓은 비선형적인 데이터가 분포된 범위까지 계산이 가능하다.(train 데이터가 전체 데이터분포에서 추출이 되어야한다)
# random_state를 None으로 하면 그에따른 난수값을 알 수없다. 훈련결과 잘 나왔을때 해당 난수값을 찾을 수없으므로 random_state는 지정해놓고 바꿔가면서 훈련하는게 좋다.

print("x_train :", x_train)
print("y_train :", y_train)
print("x_test :", x_test)
print("y_test :", y_test)

print(x_train.shape, x_test.shape)  # (7,) (3,)
print(y_train.shape, y_test.shape)  # (7,) (3,)
exit()

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