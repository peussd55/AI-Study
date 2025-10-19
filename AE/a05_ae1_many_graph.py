### <<58>>

# 인코더는 특징을 추출한다는 점에서 차원축소로 볼 수도있으므로 임의로 feature갯수를 정하기보다 pca를 적용할 수있다.

import numpy as np 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.decomposition import PCA

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

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(28*28,)))
    model.add(Dense(784, activation='sigmoid'))
    return model

# 0.95 이상 : n_comp_100
# 0.99 이상 : n_comp_999
# 0.999 이상 : n_comp_99
# 1.0 일때 : n_comp_95

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train_pca, _), (x_test_pca, _) = mnist.load_data()   # y_train, y_test는 받지 않겠다
print(x_train_pca.shape, x_test_pca.shape)  # (60000, 28, 28) (10000, 28, 28)

x = np.concatenate([x_train_pca, x_test_pca], axis=0)
print(x.shape)  # (70000, 28, 28)

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
print(x.shape)  # (70000, 784)

pca = PCA(n_components=x.shape[1])
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr) # 누적합
print(evr_cumsum)
print(len(evr_cumsum))  # 784

n_comp_100 = np.argmax(evr_cumsum >= 1.0) +1
print(n_comp_100)   # 713

n_comp_999 = np.argmax(evr_cumsum >= 0.999) +1
print(n_comp_999)   # 486

n_comp_99 = np.argmax(evr_cumsum >= 0.99) +1
print(n_comp_99)   # 331

n_comp_95 = np.argmax(evr_cumsum >= 0.95) +1
print(n_comp_95)   # 154

# hidden_size = n_comp_95

model_01 = autoencoder(hidden_layer_size=1)
model_08 = autoencoder(hidden_layer_size=8)
model_32 = autoencoder(hidden_layer_size=32)
model_64 = autoencoder(hidden_layer_size=64)
model_154 = autoencoder(hidden_layer_size=n_comp_95)
model_331 = autoencoder(hidden_layer_size=n_comp_99)
model_486 = autoencoder(hidden_layer_size=n_comp_999)
model_713 = autoencoder(hidden_layer_size=n_comp_100)

# 3. 컴파일, 훈련
# model.compile(optimizer='adam', loss='mse')
print("=============================== node 1개 시작 ===============================")
model_01.compile(optimizer='adam', loss='binary_crossentropy')
model_01.fit(x_train_noised, x_train, epochs=20, batch_size=128, validation_split=0.2, verbose=0)

print("=============================== node 8개 시작 ===============================")
model_08.compile(optimizer='adam', loss='binary_crossentropy')
model_08.fit(x_train_noised, x_train, epochs=20, batch_size=128, validation_split=0.2, verbose=0)

print("=============================== node 32개 시작 ===============================")
model_32.compile(optimizer='adam', loss='binary_crossentropy')
model_32.fit(x_train_noised, x_train, epochs=20, batch_size=128, validation_split=0.2, verbose=0)

print("=============================== node 64개 시작 ===============================")
model_64.compile(optimizer='adam', loss='binary_crossentropy')
model_64.fit(x_train_noised, x_train, epochs=20, batch_size=128, validation_split=0.2, verbose=0)

print("=============================== node 154개 시작 ===============================")
model_154.compile(optimizer='adam', loss='binary_crossentropy')
model_154.fit(x_train_noised, x_train, epochs=20, batch_size=128, validation_split=0.2, verbose=0)

print("=============================== node 331개 시작 ===============================")
model_331.compile(optimizer='adam', loss='binary_crossentropy')
model_331.fit(x_train_noised, x_train, epochs=20, batch_size=128, validation_split=0.2, verbose=0)

print("=============================== node 486개 시작 ===============================")
model_486.compile(optimizer='adam', loss='binary_crossentropy')
model_486.fit(x_train_noised, x_train, epochs=20, batch_size=128, validation_split=0.2, verbose=0)

print("=============================== node 713개 시작 ===============================")
model_713.compile(optimizer='adam', loss='binary_crossentropy')
model_713.fit(x_train_noised, x_train, epochs=20, batch_size=128, validation_split=0.2, verbose=0)

# 4. 평가, 예측
decoded_imgs_01 = model_01.predict(x_test_noised)
decoded_imgs_08 = model_08.predict(x_test_noised)
decoded_imgs_32 = model_32.predict(x_test_noised)
decoded_imgs_64 = model_64.predict(x_test_noised)
decoded_imgs_154 = model_154.predict(x_test_noised)
decoded_imgs_331 = model_331.predict(x_test_noised)
decoded_imgs_486 = model_486.predict(x_test_noised)
decoded_imgs_713 = model_713.predict(x_test_noised)

import matplotlib.pyplot as plt
import random
fig, axes = plt.subplots(9, 5, figsize=(15, 15))

random_images = random.sample(range(decoded_imgs_01.shape[0]), 5)   # 랜덤이미지 5개 인덱스추출
outputs = [x_test, decoded_imgs_01, decoded_imgs_08, decoded_imgs_32, decoded_imgs_64,
           decoded_imgs_154, decoded_imgs_331, decoded_imgs_486, decoded_imgs_713]

for row_num, row in enumerate(axes):
    for col_num, ax in enumerate(row):
        ax.imshow(outputs[row_num][random_images[col_num]].reshape(28,28), cmap='gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
plt.show()