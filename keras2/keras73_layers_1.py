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

vgg16.trainable = False                # 가중치 동결
# vgg16.trainable = True

model = Sequential()
model.add(vgg16)
model.add(Flatten())
# model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(10, activation='softmax'))

model.summary()

print(len(model.weights))               # 30, 30
print(len(model.trainable_weights))     # 30, 4 (가중치동결 X) / (가중치동결 O)