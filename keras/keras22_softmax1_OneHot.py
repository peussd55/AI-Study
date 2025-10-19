### <<10>>

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import time

# 1. 데이터
datasets = load_iris()

x = datasets.data
y = datasets.target 
print(x.shape, y.shape) # (150, 4) (150,)

print(y)
"""
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2] # 데이터가 끼리끼리 모여있어서 train_test_split에서 shuffle 하지 않으면 2나 1이 학습데이터에 포함되지 않을 수 있다.
 """
 
"""
One-Hot Encoding : 
0, 1, 2 각각 클래스를 매트릭스화 한다.(shape가 바뀐다.)
0 : 1 0 0   -> 합 : 1
1 : 0 1 0   -> 합 : 1
2 : 0 0 1   -> 합 : 1

위치값은 서로 다르지만 가치값(총합)은 1로 모두 동일하다. = 세 클래스가 동등하다.

(중요) 다중분류에서는 반드시 One-hot encoding(y만)을 한다.

(relu, linear, mse)을 쓰면 안되는 이유 : 
- y를 매트릭스화 하지 않아서 위치값에 해당하는 정보가 없다. 
    -> 벡터를 거리계산하는데 가치값이 달라져서 동등비교가 불가하다. 
    -> 0, 1, 2는 서로 다른 동등한 클래스이인데 단순 가치비교를 하면 2클래스가 1클래스의 2배의 가치를 갖는 것처럼 되버린다.

문제점 : 클래스(라벨)가 많아지면 안그래도 벡터보다 연산이 복잡한 매트릭스가 메모리용량이 커진다.
"""

# 타겟 데이터 확인
print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([50, 50, 50], dtype=int64))
print(pd.DataFrame(y).value_counts())
print(pd.value_counts(y))
# 0    50
# 1    50
# 2    50
# dtype: int64
print(type(y))       # <class 'numpy.ndarray'>

#### OneHotEncoding (반드시 y만) 3가지 방법 ####

# 1. sklearn
from sklearn.preprocessing import OneHotEncoder
# 텐서플로우 함수(to_categorical)나 판다스 함수(get_dummies)는 벡터 (x,)를 매트릭스 (x,1)로 자동변환하지만 sklearn 함수는 자동변환이 안되므로 reshape로 행렬 변환해야한다.
# reshape할때 바뀌면 안되는 중요한 요소 : 값, 순서
# ex : reshape(1, 1) -> 1행 1열로 변환
y1 = y.reshape(-1, 1)    # -1 : 행은 자동계산 // 1 : 컬럼은 1로
ohe = OneHotEncoder(sparse=False)   # sparse=False : 데이터 타입을 scipy.sparse.csr.csr_matrix -> numpy.ndarray로 변환
# csr_matrix : sklearn에서 사용하는 압축 행렬. 0이 아닌 값만 저장
# 또는 y1 = ohe.fit_transform(y1).toarray() : numpy.ndarray // 넘파이 데이터타입을 반환하기때문에 numpy 패키지에 의존한다.
y1 = ohe.fit_transform(y1)
print("Sklearn OneHotEncoder:")
print(y1)
print(y1.shape)      # (150, 3)
print(type(y1))      # scipy.sparse.csr.csr_matrix -> numpy.ndarray
y = y1
"""
# 2. pd
y2 = pd.get_dummies(y)
print("Pandas get_dummies:")
print(y2.head())     # head는 판다스 전용 함수
print(y2.shape)      # (150, 3)
print(type(y2))      # <class 'pandas.core.frame.DataFrame'>
y = y2               # 데이터타입이 다른데 할당은 되는 이유 : pd안의 데이터의 타입은 np로 저장되있기때문에 할당자체는된다. 그러나 keras는 numpy연산이 기본이기때문에 연산중 오류를 방지하도록 되도록 np로 변환후 사용하도록한다.
# 대체방법 : y2 = pd.get_dummies(y).values -> nparray

# 3. keras
from tensorflow.keras.utils import to_categorical
y3 = to_categorical(y)
print("Keras to_categorical:")
print(y3)
print(y3.shape)      # (150, 3)
print(type(y3))      # <class 'numpy.ndarray'>
y = y3
# 이 방법의 문제점 : 컬럼값을 인덱스 그대로 사용하기때문에 y클래스가 [1,2,3]이런식으로 되어있을때 앞에 불필요한 컬럼(0)을 추가생성한다.
# 조치방법 : 
y_adjusted = y - np.min(y)
y3 = to_categorical(y_adjusted)
to_categorical은 0부터 시작하는 인덱스를 빠짐없이 계속 생성(ex : y 클래스가 2,3,4 이면 0,1에 대응하는 인덱스 컬럼 2개 생성)하기때문에
클래스에 최소값을 빼서 0부터시작하게만들고 인덱스 생성하게한다.
"""
####

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size = 0.1,
    random_state = 8282,
)

# 2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=4, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))    # y가 (150, 3)으로 reshape됐으므로 아웃풋은 3
# 다중분류모델의 출력레이어 활성화 함수는 무조건 softmax

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',  # 다중분류모델의 손실함수는 무조건 categorical_crossentropy
              optimizer='adam', 
              metrics=['acc'],  # 활성화함수가 softmax일 때 내부적으로 argmax를 써서 y와 y^의 최대값위치 존재하는 인덱스만 반환 후 정확도계산
              )
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=100,
    restore_best_weights=True,
)

start_time=time.time()
model.fit(x_train, y_train, 
          epochs=1000, 
          batch_size=8,
          verbose=1,
          validation_split=0.1,     # validation_split : 넘파이다차원배열이나 텐서 연산만 가능
          callbacks=[es],
          )
# fit : 예측값이 실수면 acc 계산할때 자동으로 반올림한 후 평가함.

end_time=time.time()

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
# 활성함수가 sigmoid일 때 :	0.5 기준 반올림 후 비교
# 활성함수가 softmax일 때 :	argmax로 인덱스 추출 후 비교
print('loss :', results[0]) # 0.41873249411582947
print('(categorical)acc :', results[1])  # 0.9333333373069763

y_pred = model.predict(x_test)
print(y_pred)
# [[1.4071658e-07 1.4876697e-04 9.9985111e-01]
#  [9.9997270e-01 2.7230915e-05 1.0799404e-07]
#   ...
#  [5.7996834e-09 1.6101172e-04 9.9983895e-01]
#  [4.0454499e-04 9.9942160e-01 1.7384252e-04]]

# argmax이용하여 최대값있는 인덱스만 벡터(시리즈)로 반환후 얼마나 일치하는지 계산
y_pred = np.argmax(y_pred, axis=1)      # axis = 1 : 행 방향
print(y_pred)                           # [2 0 1 2 1 1 2 0 0 2 0 1 0 2 0]

# 만약 y원핫인코딩할때 다른방식을 해서 y_test 데이터타입이 nparray가 아니면 numpy 함수 argmax를 바로 쓸 수없다.
# -> 판다스 데이터타입을 넘파이배열로 변환 : y_test.values : 판다스의 값(value)의 데이터타입은 넘파이배열이기 때문에 values하면 넘파이배열을 반환 / 또는 toarray() 사용
y_test = np.argmax(y_test, axis=1)
print(y_test)                           # [2 0 1 1 1 1 2 0 0 2 0 1 0 2 0] 

acc = accuracy_score(y_test, y_pred)
print("accuracy : ", acc)  