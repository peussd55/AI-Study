### <<14>>

# 19-3 카피

from tensorflow.python.keras.models import Sequential, load_model
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

# # 2. 모델구성
# model = Sequential()
# model.add(Dense(128, input_dim=10))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='linear')) 

path = './_save/keras28_mcp/03_diabetes/'
model = load_model(path + 'k28_0604_1158_0163-2744.2815.hdf5')      # restore_best_weights=True일때 저장한 모델과 가중치 동일
# r2 스코어 : 0.5801714953342141

#model = load_model(path + 'keras27_3_save_model.h5')

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

"""
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping( 
    monitor = 'val_loss',       
    mode = 'min',               
    patience=100,             
    restore_best_weights=True,  
)

hist = model.fit(x_train, y_train, 
                 epochs=1000, 
                 batch_size=24, 
                 verbose=3, 
                 validation_split=0.2,
                 callbacks=[es],
                 )
"""

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
