### <<27>>

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. 데이터
x1_datasets = np.array([range(100), range(301,401)]).T    # (100, 2)
    # a주식 종가, b주식 종가
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]).transpose() # (100, 3)
    # 원유, 환율, 금시세
y = np.array(range(2001, 2101))     # (100,) 
    # 한강의 화씨온도
print(x1_datasets.shape, x2_datasets.shape, y.shape)
    
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1_datasets, x2_datasets, y,
    test_size=0.3,   
)   # y_train, y_test : x1_train, x1_test, x2_train, x2_test의 공통답지

# 2-1 모델
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='ibm1')(input1)
dense2 = Dense(20, activation='relu', name='ibm2')(dense1)
dense3 = Dense(30, activation='relu', name='ibm3')(dense2)
dense4 = Dense(40, activation='relu', name='ibm4')(dense3)
output1 = Dense(50, activation='relu', name='ibm5')(dense4)     # 첫번째모델의 마지막레이어지만 앙상블의 마지막 노드는 아니기때문에 shape를 y에 맞춰줄 필요없다.
# model1 = Model(inputs=input1, outputs=output1)
# model1.summary()

# 2-2 모델
input2 = Input(shape=(3,))
dense21 = Dense(100, activation='relu', name='ibm21')(input2)
dense22 = Dense(50, activation='relu', name='ibm22')(dense21)
output2 = Dense(30, activation='relu', name='ibm23')(dense22)   # 두번째모델의 마지막레이어지만 앙상블의 마지막 노드는 아니기때문에 shape를 y에 맞춰줄 필요없다.
# model2 = Model(inputs=input2, outputs=output2)
# model2.summary()

# 2-3. 모델 합치기
from keras.layers.merge import concatenate, Concatenate
# 또는 from tensorflow.keras.layers import concatenate, Concatenate
# from tensorflow.keras ~ : 공식
# from tensorflow.python.keras : 비공식
merge1 = concatenate([output1, output2], name='mg1')

merge2 = Dense(40, name='mg2')(merge1)
merge3 = Dense(20, name='mg3')(merge2)
last_output = Dense(1, name='mg3')(merge2)

model = Model(inputs=[input1, input2], outputs=last_output)
model.summary()
"""
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to
==================================================================================================
 input_1 (InputLayer)           [(None, 2)]          0           []

 ibm1 (Dense)                   (None, 10)           30          ['input_1[0][0]']

 ibm2 (Dense)                   (None, 20)           220         ['ibm1[0][0]']

 input_2 (InputLayer)           [(None, 3)]          0           []

 ibm3 (Dense)                   (None, 30)           630         ['ibm2[0][0]']

 ibm21 (Dense)                  (None, 100)          400         ['input_2[0][0]']

 ibm4 (Dense)                   (None, 40)           1240        ['ibm3[0][0]']

 ibm22 (Dense)                  (None, 50)           5050        ['ibm21[0][0]']

 ibm5 (Dense)                   (None, 50)           2050        ['ibm4[0][0]']

 ibm23 (Dense)                  (None, 30)           1530        ['ibm22[0][0]']
 mg1 (Concatenate)              (None, 80)           0           ['ibm5[0][0]',
                                                                  'ibm23[0][0]']

 mg2 (Dense)                    (None, 40)           3240        ['mg1[0][0]']

 mg3 (Dense)                    (None, 1)            41          ['mg2[0][0]']

==================================================================================================
Total params: 14,431
Trainable params: 14,431
Non-trainable params: 0
__________________________________________________________________________________________________    
"""

# 3. 컴파일, 훈련
import datetime
model.compile(loss='mae', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping( 
    monitor = 'val_loss',       
    mode = 'min',               
    patience=50,             
    restore_best_weights=True,  
)

hist = model.fit([x1_train, x2_train], y_train,
                epochs=300, 
                batch_size=32, 
                verbose=1, 
                validation_split=0.2,
                callbacks=[es],
                ) 

# 4.평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)
print('loss:', loss)

x1_pred = np.array([range(100,106), range(400,406)]).T
x2_pred = np.array([range(200, 206), range(510, 516), range(249, 255)]).T

print(x1_pred.shape, x2_pred.shape) # (N, 2) (N, 3)
exit()
y_pred = model.predict([x1_pred, x2_pred])
print(y_pred)

# y_pred는 2101부터 2106까지 나오면됨