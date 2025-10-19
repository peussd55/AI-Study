### <<05>>

import numpy as np 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
from sklearn.model_selection import train_test_split

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,7,5,7,8,6,10])

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
model.fit(x_train, y_train, epochs=100, batch_size=1)   # epochs가 일정 수치 이상 되면 loss가 횡보한다.

# 4. 평가, 예측
print('=============================================================')
loss = model.evaluate(x_test, y_test)   # 학습할때 최종 loss랑 evaluate 할때 loss가 차이가 많이 나는데 학습데이터와 테스트데이터가 분리됐기 때문이다.
results = model.predict([x])
print('loss:', loss)
print('[x]의 예측값:', results)

# 그래프 그리기
import matplotlib.pyplot as plt 
plt.scatter(x, y)   #데이터 점 찍기
plt.plot(x, results, color='red')
plt.show()
