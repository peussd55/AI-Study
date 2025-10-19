### <<55>>

# VGG16 전이학습

import numpy as np 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
import tensorflow as tf  
import random 

SEED = 333
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from tensorflow.keras.applications import VGG16, VGG19 
from tensorflow.keras.applications import ResNet50, ResNet50V2, ResNet101, ResNet152, ResNet152V2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.datasets import cifar10

vgg16 = VGG16(
              include_top=False,      # False : 위(intputlayer)를 가변적으로 하고, 아래(fc layer)를 사용하지 않는다. -> 사용자가 전이학습할수있도록하는 파라미터
              input_shape=(32,32,3),
            )

# vgg16.trainable = False #가중치 동결

model = Sequential()
model.add(vgg16)
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(10, activation='softmax'))

model.summary()
# 가중치동결X
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  vgg16 (Functional)          (None, 1, 1, 512)         14714688

#  flatten (Flatten)           (None, 512)               0
#  dense (Dense)               (None, 128)               65664
#  dense_1 (Dense)             (None, 64)                8256
#  dense_2 (Dense)             (None, 10)                650

# =================================================================
# Total params: 14,789,258
# Trainable params: 14,789,258
# Non-trainable params: 0
# _________________________________________________________________

# 가중치동결O
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  vgg16 (Functional)          (None, 1, 1, 512)         14714688  

#  flatten (Flatten)           (None, 512)               0

#  dense (Dense)               (None, 256)               131328

#  dense_1 (Dense)             (None, 128)               32896

#  dense_2 (Dense)             (None, 10)                1290

# =================================================================
# Total params: 14,880,202
# Trainable params: 165,514
# Non-trainable params: 14,714,688
# _________________________________________________________________

# 실습
# 가중치동결O
# 가중치동결X
# 시간까지비교
# Flatten과 Gap 비교

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

# 정규화
x_train = x_train/255.
x_test = x_test/255.

# 3.컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy',
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

import time
start = time.time()
hist = model.fit(x_train, y_train, 
                 epochs=200, 
                 batch_size=128, 
                 verbose=3, 
                 validation_split=0.2,
                 callbacks=[es],
                 )
end = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1) 

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1) 

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print("accuracy : ", acc)
print("걸린 시간 :", round(end-start, 2), "초")

"""
[가중치동결O, flatten]  
accuracy :  0.5867
걸린 시간 : 404.1 초

[가중치동결O, GAP]  
accuracy :  0.5866
걸린 시간 : 363.69 초

[가중치동결X, flatten]
accuracy :  0.7901
걸린 시간 : 870.03 초

[가중치동결X, GAP]  
accuracy :  0.7747
걸린 시간 : 788.98 초
"""