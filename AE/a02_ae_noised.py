### <<58>>

import numpy as np 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], 28*28).astype('float32')/255.

# 평균 0, 표편 0.1인 정규분포 형태의 랜덤값을 노이즈로 추가
x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)
print(x_train_noised.shape, x_test_noised.shape)        # (60000, 784) (10000, 784)
print(np.max(x_train), np.min(x_test))                  # 1.0 0.0
print(np.max(x_train_noised), np.min(x_test_noised))    # 1.5245708987979847 -0.5063763773721566

# 최대, 최소를 0~1로 제한
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, 0, 1)
print(np.max(x_train_noised), np.min(x_test_noised))    # 1.0 0.0

# 2. 모델
input_img = Input(shape=(28*28,))

#### 인코더
# encoded = Dense(1, activation='relu')(input_img)      # 특징추출이 잘 안되고 뭉개짐

# 노드늘어날수록 선명해짐                     
# encoded = Dense(32, activation='relu')(input_img)       
encoded = Dense(64, activation='relu')(input_img)
# encoded = Dense(128, activation='relu')(input_img)
# encoded = Dense(256, activation='relu')(input_img)
# encoded = Dense(784, activation='relu')(input_img)
# encoded = Dense(1024, activation='relu')(input_img)

# encoded = Dense(64, activation='tanh')(input_img)     # 괜춘

#### 디코더
# decoded = Dense(28*28, activation='linear')(encoded)
# decoded = Dense(28*28, activation='relu')(encoded)
# decoded = Dense(28*28, activation='linear')(encoded)
# decoded = Dense(28*28, activation='tanh')(encoded)    # 구림
decoded = Dense(28*28, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)

# 3. 컴파일, 훈련
# autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train_noised, x_train, epochs=50, batch_size=128, validation_split=0.2)   # 노이즈가 들어간 것을 원본으로 학습 : 복원기술
# [생성형모델에서의 loss]
# 학습이 진행이 잘되는지에 대한 판단으로 loss는 의미가있으나
# 최종결과를 loss로 판단을 할 수는 없다. 사람이 판단을 하기때문이다.

# 4. 평가, 예측
decoded_imgs = autoencoder.predict(x_test_noised)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test_noised[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

