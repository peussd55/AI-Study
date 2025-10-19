### <<14>>

# 19-3 카피

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np 

#[실습]
#r2_score > 0.62

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x)
print(y)
print(x.shape, y.shape)     # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=1111
)

# 2. 모델구성
model = Sequential()
model.add(Dense(128, input_dim=10))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear')) 

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping( 
    monitor = 'val_loss',       
    mode = 'min',               
    patience=100,             
    restore_best_weights=True,  
)
####################### mcp 세이브 파일명 만들기 #######################
import datetime
date = datetime.datetime.now()
print(date)         # 2025-06-02 13:00:40.661379
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime('%m%d_%H%M')
print(date)         # 0602_1305
print(type(date))   # <class 'str'>

path = './_save/keras28_mcp/03_diabetes/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 04d : 정수 4자리, .4f : 소수점 4자리
filepath = "".join([path, 'k28_', date, '_', filename])     # 구분자를 공백("")으로 하겠다.
# ./_save/keras27_mcp2/k27_0602_1442_{epoch:04d}-{val_loss:.4f}.hdf5

print(filepath)

#exit()

mcp = ModelCheckpoint(          # 모델+가중치 저장
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only=True,
    filepath = filepath,
)

hist = model.fit(x_train, y_train, 
                 epochs=1000, 
                 batch_size=24, 
                 verbose=3, 
                 validation_split=0.2,
                 callbacks=[es, mcp],
                 )
print("=============== hist =================")
print(hist)     # <keras.callbacks.History object at 0x00000179B5A08BB0>
print("=============== hist.history =================")
print(hist.history)
# {} : 딕셔너리 // [] : 리스트
print("=============== loss =================")
print(hist.history['loss'])
print("=============== val_loss =================")
print(hist.history['val_loss'])

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))       # 9 x 6 사이즈
plt.plot(hist.history['loss'], c='red', label='loss')   # plot (x, y, color= ....) : y값만 넣으면 x는 1부터 시작하는 정수 리스트
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.title('diabetes Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper right')   # 유측 상단에 라벨표시 
plt.grid()  #격자표시
plt.show()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)

from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_test, results)
print('r2 스코어 :', r2)
# r2 스코어 : 0.5821774854770508


"""
EarlyStopping X
epochs=1000, 
batch_size=24, 
r2 스코어 :  0.5868124047871888

EarlyStopping O / restore_best_weights=True
patience=100
epochs=1000, 
batch_size=24, 
stop지점 : 303,
r2 스코어 : 0.581284979899215

EarlyStopping O / restore_best_weights=False
patience=100
epochs=1000, 
batch_size=24, 
stop지점 : 268,
r2 스코어 : r2 스코어 : 0.5810320544498263
"""
