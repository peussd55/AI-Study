### <<21>>

# C:\Study25\_data\kaggle\men_women

import numpy as np
import time
import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPool2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# 1. 데이터
np_path = 'c:/study25/_data/_save_npy/'

x_train_load = np.load(np_path + "keras46_07_x_train_size220.npy")
y_train_load = np.load(np_path + "keras46_07_y_train_size220.npy")

# test데이터가 따로 없으므로 train 데이터에서 분리
x_train, x_test, y_train, y_test = train_test_split(
    x_train_load, y_train_load,
    test_size = 0.3,
    # stratify=y_train_load,
    random_state=8282,
)

# 2. 모델구성 : 레이어 unit이 너무 많으면 메모리 부족으로 학습안된다.
# 학습시 데이터 이동단계 : 학습데이터 cpu메모리에서 gpu메모리로 업로드함 -> gpu메모리(전용메모리)가 부족하면 학습불가 -> 분할배치나 생성자를 사용해야함.
model = Sequential()
model.add(Conv2D(32, (3,3), strides=1, input_shape=(220, 220, 3))) # input_shape(높이, 너비, 채널) = (세로, 가로, 채널)                    
model.add(MaxPool2D(pool_size=(2, 2))) 
model.add(Dropout(0.25))       
                                            
model.add(Conv2D(64, (3,3), strides=1, activation='relu')) # input_shape(높이, 너비, 채널) = (세로, 가로, 채널)                    
model.add(MaxPool2D(pool_size=(2, 2))) 
model.add(Dropout(0.25)) 

model.add(Conv2D(128, (3,3), strides=1, activation='relu')) # input_shape(높이, 너비, 채널) = (세로, 가로, 채널)                    
model.add(MaxPool2D(pool_size=(2, 2))) 
model.add(Dropout(0.25)) 

model.add(Flatten())    
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(units=64, activation='relu'))

model.add(Dense(units=1, activation='sigmoid'))
# model.summary()

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', 
            optimizer='adam', 
            metrics=['acc'],
            )

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping( 
    monitor = 'val_acc',       
    mode = 'auto',              
    patience=100,          
    verbose=1,     
    restore_best_weights=True, 
)

date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path = './_save/kaggle/men_women/'
# filepath 가변 (갱신때마다 저장)
filename = '{epoch:04d}-{val_acc:.4f}.hdf5'    # 04d : 정수 4자리, .4f : 소수점 4자리
filepath = "".join([path, 'k46_', date, '_', filename])     # 구분자를 공백("")으로 하겠다.
# filepath 고정 (종료때만 저장)
# filepath = path + f'keras46_mcp.hdf5'

mcp = ModelCheckpoint(          # 모델+가중치 저장
    monitor = 'val_acc',
    mode = 'auto',
    save_best_only=True,
    filepath = filepath,
    verbose=1,
)

start_time = time.time()
hist = model.fit(x_train, y_train, 
                batch_size = 64,
                epochs=500,
                verbose=3, 
                validation_split=0.2,
                callbacks=[es,
                            mcp,
                           ],
                )
end_time = time.time()

# 그래프 그리기
plt.figure(figsize=(18, 5))
# 첫 번째 그래프
plt.subplot(1, 2, 1)
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.title('img loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.grid()

# 두 번째 그래프
plt.subplot(1, 2, 2)
plt.plot(hist.history['acc'], c='red', label='acc')
plt.plot(hist.history['val_acc'], c='blue', label='val_acc')
plt.title('img acc')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()
plt.grid()
plt.tight_layout()  # 간격 자동 조정

# 4. 평가, 예측
# 4.1 평가
results = model.evaluate(x_test, y_test)
print('loss :', results[0])
print('acc :', results[1])

# 4.2 예측
y_pred = model.predict(x_test)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, np.round(y_pred)) 
print('걸린시간 :', round(end_time-start_time, 2), '초')
acc2 = acc
print('acc(sklearn 지표) :', acc)

# 걸린시간 : 205.18 초
# acc(sklearn 지표) : 0.7049345417925479

# 220x220
# 걸린시간 : 390.85 초
# acc(sklearn 지표) : 0.7150050352467271

plt.show()