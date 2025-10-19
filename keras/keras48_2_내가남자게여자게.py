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
path_me = 'c:/study25/_data/image/me/'
x_test_me = np.load(path_me + "43.npy")
print(x_test_me.shape)  # (1, 220, 220, 3)
# 정규화
x_test_me = x_test_me / 255.0  # 0-255 → 0-1 범위로 정규화 [1][2]
print("정규화 후 데이터 범위:", x_test_me.min(), x_test_me.max())  # 0.0 ~ 1.0 확인

# 증폭X 모델
# path = './_save/kaggle/men_women/'
# model = load_model(path + 'k46_0616_1451_0021-0.7198.hdf5')      # restore_best_weights=True일때 저장한 모델과 가중치 동일

# 증폭O 모델
path = './_save/keras50_augment8/'
model = load_model(path + 'k50_0617_1618_0050-0.7677.hdf5')    
# model = load_model(path + 'k50_0618_1120_0048-0.6597.hdf5')      
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', 
            optimizer='adam', 
            metrics=['acc'],
            )

# 4.2 예측
from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test_me)

print(y_pred)   # [[1.]]
gender = '여자' if y_pred[0][0] > 0.5 else '남자'
print(f'내 이미지 예측: {gender} (acc: {y_pred[0][0]:.4f})')

# 증폭X 모델
# [[0.82774544]]
# 내 이미지 예측: 여자 (acc: 0.8277)

# 증폭O 모델
# [[0.0.00010069]]
# 내 이미지 예측: 남자 (acc: 0.0001)

plt.show()