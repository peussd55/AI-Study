### <<20>>

import numpy as np
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPool2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

np_path = 'c:/study25/_data/_save_npy/'
# np.save(np_path + 'keras44_01_x_train.npy', arr=x)
# np.save(np_path + 'keras44_01_y_train.npy', arr=y)

start = time.time()
x_train = np.load(np_path + "keras44_01_x_train.npy")
y_train = np.load(np_path + "keras44_01_y_train.npy")
end = time.time()

print(x_train)
print(y_train[:20])
print(x_train.shape, y_train.shape)
print('불러오기 걸린시간 :', round(end-start, 2), '초')     # 불러오기 걸린시간 : 66.69 초