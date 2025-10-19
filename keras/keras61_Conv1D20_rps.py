### <<26>>
import numpy as np
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPool1D # Conv1D, Flatten, MaxPool1D 추가
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

# 1. 데이터
np_path = 'c:/study25/_data/_save_npy/'
x_train_load = np.load(np_path + "keras46_05_x_train.npy")
y_train_load = np.load(np_path + "keras46_05_y_train.npy")

x_train, x_test, y_train, y_test = train_test_split(
    x_train_load, y_train_load, test_size=0.3, random_state=8282
)

datagen = ImageDataGenerator(
    horizontal_flip=True, vertical_flip=True, width_shift_range=0.1, rotation_range=15
)
augment_size = 700
randidx = np.random.randint(x_train.shape[0], size=augment_size)
x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
x_augmented = datagen.flow(x_augmented, y_augmented, batch_size=augment_size, shuffle=False).next()[0]
x_train = np.concatenate([x_train, x_augmented])
y_train = np.concatenate([y_train, y_augmented])

# Conv1D 입력을 위해 데이터 Flatten
num_features = x_train.shape[1] * x_train.shape[2] * x_train.shape[3]
x_train_flat = x_train.reshape(x_train.shape[0], num_features)
x_test_flat = x_test.reshape(x_test.shape[0], num_features)

# 요청사항 1: Conv1D 입력을 위한 데이터 Reshape
x_train_reshaped = x_train_flat.reshape(x_train_flat.shape[0], num_features, 1)
x_test_reshaped = x_test_flat.reshape(x_test_flat.shape[0], num_features, 1)

# 요청사항 2: 모델을 Conv1D 기반으로 변경
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=7, input_shape=(num_features, 1), activation='relu'))
model.add(MaxPool1D(pool_size=4))
model.add(Dropout(0.25))
model.add(Conv1D(filters=64, kernel_size=7, activation='relu'))
model.add(MaxPool1D(pool_size=4))
model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=3, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_acc', mode='max', patience=30, restore_best_weights=True)

start_time = time.time()
model.fit(
    x_train_reshaped, y_train, batch_size=32, epochs=200, verbose=1,
    validation_split=0.2, callbacks=[es]
)
end_time = time.time()

# 4. 평가, 예측
print("걸린 시간 :", round(end_time - start_time, 2), "초")
results = model.evaluate(x_test_reshaped, y_test)
y_pred = np.argmax(model.predict(x_test_reshaped), axis=1)
y_test_argmax = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test_argmax, y_pred)

print(f'loss: {results[0]}, acc: {results[1]}')
print(f"Accuracy Score: {acc}")
