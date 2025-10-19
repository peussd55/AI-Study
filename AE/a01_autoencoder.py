### <<58>>

import numpy as np 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], 28*28).astype('float32')/255.

# 2. 모델
input_img = Input(shape=(28*28,))

#### 인코더
# encoded = Dense(1, activation='relu')(input_img)      # 특징추출이 잘 안되고 뭉개짐

# 노드늘어날수록 선명해짐                     
# encoded = Dense(32, activation='relu')(input_img)       
# encoded = Dense(64, activation='relu')(input_img)
# encoded = Dense(128, activation='relu')(input_img)
# encoded = Dense(256, activation='relu')(input_img)
# encoded = Dense(784, activation='relu')(input_img)
encoded = Dense(1024, activation='relu')(input_img)

#### 디코더
# decoded = Dense(28*28, activation='linear')(encoded)
# decoded = Dense(28*28, activation='relu')(encoded)
# decoded = Dense(28*28, activation='linear')(encoded)
# decoded = Dense(28*28, activation='tanh')(encoded)
decoded = Dense(28*28, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)

# 3. 컴파일, 훈련
# autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=30, batch_size=128, validation_split=0.2)

# 4. 평가, 예측
decoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()