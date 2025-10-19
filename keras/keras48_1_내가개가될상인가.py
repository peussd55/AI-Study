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
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image

# 1. 데이터
path_me = 'c:/study25/_data/image/me/'
x_test_me = np.load(path_me + "keras47_me_size50.npy")  # model이 50x50 사이즈로 훈련했으면 테스트데이터도 50x50 사이즈이어야한다.
print(x_test_me.shape)  # (1, 220, 220, 3)
# 정규화
x_test_me = x_test_me / 255.0  # 0-255 → 0-1 범위로 정규화 [1][2]
print("정규화 후 데이터 범위:", x_test_me.min(), x_test_me.max())  # 0.0 ~ 1.0 확인

path = './_save/kaggle/cat_dog/'
model = load_model(path + 'k43_0616_1522_0023-0.5894.hdf5')      # restore_best_weights=True일때 저장한 모델과 가중치 동일
model.summary()
# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', 
            optimizer='adam', 
            metrics=['acc'],
            )

# 4.2 예측
from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test_me)

print(y_pred)   # [[0.36585188]]
gender = '개' if y_pred[0][0] > 0.5 else '고양이'
print(f'내 이미지 예측: {gender} (acc: {y_pred[0][0]:.4f})')

plt.show()