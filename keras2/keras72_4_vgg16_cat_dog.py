### <<55>>

# VGG16 전이학습

import numpy as np 
import pandas as pd
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
import tensorflow as tf  
import random 
import time

SEED = 333
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from tensorflow.keras.applications import VGG16, VGG19 
from tensorflow.keras.applications import ResNet50, ResNet50V2, ResNet101, ResNet152, ResNet152V2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.datasets import cifar100
from sklearn.model_selection import train_test_split

# 1. 데이터
np_path = 'c:/study25/_data/_save_npy/'
path = 'c:/study25/_data/kaggle/cat_dog/'
submission_csv = pd.read_csv(path + 'sample_submission.csv')
# np.save(np_path + 'keras44_01_x_train.npy', arr=x)
# np.save(np_path + 'keras44_01_y_train.npy', arr=y)

# train데이터 로드 (.npy 불러오기)
start = time.time()
# 200하면 터짐
x_train_load = np.load(np_path + "keras44_01_x_train.npy")
y_train_load = np.load(np_path + "keras44_01_y_train.npy")
end = time.time()

# print(x_train)
print(x_train_load[:20])
print(x_train_load.shape, x_train_load.shape)     # (25000, 200, 200, 3) (25000,)
print('train 데이터 불러오기 걸린시간 :', round(end-start, 2), '초')     # 2.27 초

# 로드한 train 데이터를 학습,테스트 데이터로 분할
x_train, x_test, y_train, y_test = train_test_split(
    x_train_load, y_train_load,
    test_size = 0.2,
    random_state=8282,
)

# 테스트데이터 로드2 (.npy 불러오기)
start1 = time.time()
submission_x = np.load(np_path + 'keras44_01_x_submission.npy')
print(f"제출용 테스트 데이터 로드 완료 (from .npy). shape: {submission_x.shape}")
end1 = time.time()
print('test 데이터 불러오기 걸린시간 :', round(end1-start1, 2), '초')      # 1.09 초

# print(submission_x)
print('제출파일 shape:', submission_x.shape)    # (12500, 200, 200, 3)
print('제출파일 type:', type(submission_x))     # <class 'numpy.ndarray'>
print('x_test shape:', x_test.shape)           # (7500, 200, 200, 3)
print('x_test type:', type(x_test))            # <class 'numpy.ndarray'>

# 2. 모델
vgg16 = VGG16(
              weights="imagenet",
              include_top=False,      # False : 위(intputlayer)를 가변적으로 하고, 아래(fc layer)를 사용하지 않는다. -> 사용자가 전이학습할수있도록하는 파라미터
              input_shape=(50,50,3),
            )

vgg16.trainable = True  #가중치 동결. 전이학습용모델 불러오면

model = Sequential()
model.add(vgg16)
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# 3.컴파일, 훈련
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'],
              )

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', 
                   mode='min',
                   patience=30,
                   verbose=1,   # stop이 어느 epoch에서 걸렸는지 출력해줌(modelcheckpoint도 적용가능)
                   restore_best_weights=True,
                   )
start = time.time()
hist = model.fit(x_train, y_train, 
                 epochs=200, 
                 batch_size=128, 
                 verbose=1, 
                 validation_split=0.1,
                 callbacks=[es],
                 )
end = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1) 

y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, np.round(y_pred))
print("accuracy : ", acc)
print("걸린 시간 :", round(end-start, 2), "초")

"""
[가중치동결O, flatten]  

[가중치동결O, GAP]

[가중치동결X, flatten]

[가중치동결X, GAP]  

"""