### <<27>>

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. 데이터
x_datasets = np.array([range(100), range(301,401)]).T    # (100, 2)
    # a주식 종가, b주식 종가
y1 = np.array(range(2001, 2101))
    # 한강의 화씨온도
y2 = np.array(range(13001, 13101))
    # 비트코인 가격
    
print(x_datasets.shape, y1.shape, y2.shape) # (100, 2) (100,) (100,)

x_train, x_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x_datasets, y1, y2,
    test_size=0.3,   
)   # y_train, y_test : x1_train, x1_test, x2_train, x2_test의 공통답지
# exit()
# 2-1 모델
input1 = Input(shape=(2,))
dense1 = Dense(50, activation='relu', name='ibm1')(input1)
dense2 = Dense(40, activation='relu', name='ibm2')(dense1)
dense3 = Dense(30, activation='relu', name='ibm3')(dense2)
dense4 = Dense(20, activation='relu', name='ibm4')(dense3)
output1 = Dense(10, activation='relu', name='ibm5')(dense4)     # 첫번째모델의 마지막레이어지만 앙상블의 마지막 노드는 아니기때문에 shape를 y에 맞춰줄 필요없다.
# model1 = Model(inputs=input1, outputs=output1)
# model1.summary()

# 여기서부터 y1, y2로 분기!
# y1 분기 (한강의 화씨온도)
y1_branch1 = Dense(15, activation='relu', name='y1_dense1')(output1)
y1_branch2 = Dense(10, activation='relu', name='y1_dense2')(y1_branch1)
y1_output = Dense(1, activation='linear', name='y1_output')(y1_branch2)  # 회귀이므로 linear

# y2 분기 (비트코인 가격)
y2_branch1 = Dense(15, activation='relu', name='y2_dense1')(output1)
y2_branch2 = Dense(10, activation='relu', name='y2_dense2')(y2_branch1)
y2_output = Dense(1, activation='linear', name='y2_output')(y2_branch2)  # 회귀이므로 linear

# 모델 갯수 : x1, x2, x3 레이어, merge레이어, y1레이어, y2레이어 : 6개
model = Model(inputs=input1, outputs=[y1_output, y2_output])
model.summary()

# 3. 컴파일, 훈련
import datetime
model.compile(
                loss={
                    'y1_output': 'mse',
                    'y2_output': 'mse'
                },
                metrics={
                    'y1_output': 'mae',
                    'y2_output': 'mae'
                },
              optimizer='adam',
                #   loss_weights=[0.7, 0.3]  # y1에 70%, y2에 30% 가중치
                # 이 경우: 총합_손실 = (y1_손실 × 0.7) + (y2_손실 × 0.3)
              )

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping( 
    monitor = 'val_loss',       
    mode = 'min',               
    patience=300,             
    restore_best_weights=True,  
)

hist = model.fit(x_train, [y1_train, y2_train],
                epochs=1000, 
                batch_size=32, 
                verbose=1, 
                validation_split=0.2,
                callbacks=[es],
                ) 

# 4.평가, 예측
loss = model.evaluate(x_test, [y1_test, y2_test])
print('loss:', loss)    
"""
1)
    [100.72779083251953, 1.6525429487228394, 99.07524871826172] : loss, y1_output1_loss, y2_output_loss
    loss = [전체_손실, y1_output_손실, y2_output_손실]
    첫 번째 요소 (전체 손실):
    모든 출력의 총합 손실
    y1_output_loss + y2_output_loss의 합계
    모델의 전반적인 성능을 나타냄
    두 번째 요소 (y1_output 손실):
    한강 온도 예측에 대한 개별 MSE 손실
    y1_output 레이어만의 성능
    세 번째 요소 (y2_output 손실):
    비트코인 가격 예측에 대한 개별 MSE 손실
    y2_output 레이어만의 성능
"""
"""
2)
    loss: [1003.8184204101562, 564.6488647460938, 439.1695556640625, 20.47243309020996, 7.9482421875]
    첫 번째 (1003.82): 전체 손실 (Total Loss)
    y1_output_loss + y2_output_loss의 합
    두 번째 (564.65): y1_output 손실 (한강 온도 MSE)
    한강 온도 예측의 평균제곱오차
    세 번째 (439.17): y2_output 손실 (비트코인 가격 MSE)
    비트코인 가격 예측의 평균제곱오차
    네 번째 (20.47): y1_output MAE (한강 온도 평균절대오차)
    한강 온도 예측의 평균절대오차
    다섯 번째 (7.95): y2_output MAE (비트코인 가격 평균절대오차)
    비트코인 가격 예측의 평균절대오차
"""
x_pred = np.array([range(100,106), range(400,406)]).T
print(x_pred)
# [[100 400]
#  [101 401]
#  [102 402]
#  [103 403]
#  [104 404]
#  [104 404]
#  [105 405]]

print(x_pred.shape) # (N, 2)
y_pred = model.predict(x_pred)
print(y_pred[0])
print(y_pred[1])
print('y1_pred.shape :', y_pred[0].shape)   # (N, 1)
print('y2_pred.shape :', y_pred[1].shape)   # (N, 1)
# y1_pred.shape : [[2078.3403]
# [[2060.6975]
#  [2064.2087]
#  [2064.2087]
#  [2067.72  ]
#  [2071.2314]
#  [2074.7424]
#  [2078.254 ]]
# y2_pred.shape : [[13220.346]
# [[13094.395]
#  [13116.733]
#  [13139.071]
#  [13161.409]
#  [13183.746]
#  [13206.085]]
