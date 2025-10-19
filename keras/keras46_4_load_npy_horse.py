### <<21>>

# C:\Study25\_data\tensor_cert\horse-or-human

import numpy as np
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPool2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# 1. 데이터
np_path = 'c:/study25/_data/_save_npy/'

x_train_load = np.load(np_path + "keras46_03_x_train.npy")
y_train_load = np.load(np_path + "keras46_03_y_train.npy")

# test데이터가 따로 없으므로 train 데이터에서 분리
x_train, x_test, y_train, y_test = train_test_split(
    x_train_load, y_train_load,
    test_size = 0.3,
    random_state=8282,
)

# 2. 모델구성 : 레이어 unit이 너무 많으면 메모리 부족으로 학습안된다.
# 학습시 데이터 이동단계 : 학습데이터 cpu메모리에서 gpu메모리로 업로드함 -> gpu메모리(전용메모리)가 부족하면 학습불가 -> 분할배치나 생성자를 사용해야함.
model = Sequential()
model.add(Conv2D(32, (3,3), strides=1, input_shape=(200, 200, 3))) # input_shape(높이, 너비, 채널) = (세로, 가로, 채널)                    
model.add(MaxPool2D(pool_size=(2, 2))) 
model.add(Dropout(0.25))       
                                            
model.add(Conv2D(64, (3,3), strides=1, activation='relu')) # input_shape(높이, 너비, 채널) = (세로, 가로, 채널)                    
model.add(MaxPool2D(pool_size=(2, 2))) 
model.add(Dropout(0.25)) 

model.add(Flatten())    
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(units=32, activation='relu'))

model.add(Dense(units=1, activation='sigmoid'))
# model.summary()

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', 
            optimizer='adam', 
            metrics=['acc'],
            )

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping( 
    monitor = 'val_loss',       
    mode = 'auto',              
    patience=30,          
    verbose=1,     
    restore_best_weights=True, 
)
start_time = time.time()
hist = model.fit(x_train, y_train, 
                batch_size = 32,
                epochs=200,
                verbose=3, 
                validation_split=0.2,
                callbacks=[es],
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

# 1.0

plt.show()