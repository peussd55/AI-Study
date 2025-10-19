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

hidden_size = n_comp_95

model = autoencoder(hidden_layer_size=hidden_size)

# 3. 컴파일, 훈련
# model.compile(optimizer='adam', loss='mse')
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(x_train_noised, x_train, epochs=50, batch_size=128, validation_split=0.2)   # 노이즈가 들어간 것을 원본으로 학습 : 복원기술
# [생성형모델에서의 loss]
# 학습이 진행이 잘되는지에 대한 판단으로 loss는 의미가있으나
# 최종결과를 loss로 판단을 할 수는 없다. 사람이 판단을 하기때문이다.

# 4. 평가, 예측
decoded_imgs = model.predict(x_test_noised)

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

