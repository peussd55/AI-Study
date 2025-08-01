### <<26>>
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout # Conv1D, Flatten 추가
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train_flat = x_train.reshape(60000, 28 * 28)
x_test_flat = x_test.reshape(10000, 28 * 28)
num_features = x_train_flat.shape[1] # 784

ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train.reshape(-1, 1))
y_test = ohe.transform(y_test.reshape(-1, 1))

# 요청사항 1: Conv1D 입력을 위한 데이터 Reshape
x_train_reshaped = x_train_flat.reshape(x_train_flat.shape[0], num_features, 1)
x_test_reshaped = x_test_flat.reshape(x_test_flat.shape[0], num_features, 1)

# 요청사항 2: 모델의 첫 번째 레이어를 Conv1D로 변경
model = Sequential()
model.add(Conv1D(filters=128, kernel_size=5, input_shape=(num_features, 1), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_acc', mode='max', patience=50, restore_best_weights=True)

start = time.time()
model.fit(
    x_train_reshaped, y_train, epochs=200, batch_size=64, verbose=1,
    validation_split=0.2, callbacks=[es]
)
end = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test_reshaped, y_test, verbose=1)
y_pred = np.argmax(model.predict(x_test_reshaped), axis=1)
y_test_argmax = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test_argmax, y_pred)

print("걸린 시간 :", round(end-start, 2), "초")
print(f'loss: {loss[0]}, acc: {loss[1]}')
print(f"accuracy: {acc}")
