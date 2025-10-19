### <<59>>

"""
keras46, 47을 참고하여, 남자 여자 사진에 노이즈를 주고,
내 사진에도 노이즈를 추가,
오토 인코더(CAE)로 피부 미백 훈련 가중치를 만든다.
그 가중치로 내 사진을 예측해서 피부 미백
"""

# C:\Study25\_data\kaggle\men_women

import numpy as np
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPool2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, UpSampling2D, MaxPooling2D

start1 = time.time()
train_datagen = ImageDataGenerator(
    rescale=1./255,      # 0 ~ 255 스케일링, 정규화
)

path_train = 'c:/Study25/_data/kaggle/men_women/'  # 경로를 로컬 환경에 맞게 수정

xy_train = train_datagen.flow_from_directory( 
    path_train,
    target_size=[200, 200],
    batch_size=100,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
    seed=333,
)
# Found 3309 images belonging to 2 classes.

end1 = time.time()
print('배치업로드 준비 완료시간 :', round(end1-start1, 2), '초')

############ 모든 수치화된 batch데이터를 하나로 합치기 ############ 
print("데이터 배치를 하나로 합치는 중...")
all_x = []
all_y = []
for i in range(len(xy_train)):
    x_batch, y_batch = xy_train[i]
    all_x.append(x_batch)
    all_y.append(y_batch)
    print(f"{i+1}/{len(xy_train)} 배치 처리 완료")

############ 리스트를 하나의 numpy 배열로 합친다. ############
x = np.concatenate(all_x, axis=0)
y = np.concatenate(all_y, axis=0)
print('x.shape : ', x.shape)  # (3309, 200, 200, 3)
print('y.shape : ', y.shape)  # (3309,)


# ==================== 노이즈 추가 (수정된 부분) ====================
print("노이즈 추가 작업 시작...")
# 평균 0, 표준편차 0.1인 정규분포 형태의 랜덤값을 노이즈로 추가
# x는 이미 0~1 사이로 정규화되어 있음
x_noised = x + np.random.normal(0, 0.1, size=x.shape)

# 최대, 최소를 0~1로 제한
x_noised = np.clip(x_noised, a_min=0, a_max=1)
print("노이즈 추가 작업 완료.")
print(f"노이즈 추가 후 최대/최소값: {np.max(x_noised):.4f}, {np.min(x_noised):.4f}")
# =================================================================

############ 합친 numpy를 저장한다. ############
start4 = time.time()
np_path = 'c:/study25/_data/_save_npy/'

# 원본(깨끗한) 데이터와 노이즈 추가된 데이터를 각각 저장
np.save(np_path + 'keras59_men_women_x_original.npy', arr=x)
np.save(np_path + 'keras59_men_women_x_noised.npy', arr=x_noised)
np.save(np_path + 'keras59_men_women_y.npy', arr=y)

end4 = time.time()
print('npy 저장 완료!')
print('npy 저장시간 :', round(end4-start4, 2), '초')

# # (선택사항) 노이즈 추가 결과 시각화
# plt.figure(figsize=(10, 5))

# # 원본 이미지
# plt.subplot(1, 2, 1)
# plt.imshow(x[0])
# plt.title('Original Image')
# plt.axis('off')

# # 노이즈 추가된 이미지
# plt.subplot(1, 2, 2)
# plt.imshow(x_noised[0])
# plt.title('Noised Image')
# plt.axis('off')

# plt.show()

# 데이터 로드
x_train_load = np.load(np_path + "keras59_men_women_x_noised.npy")
x_test_laod = np.load(np_path + "keras59_men_women_x_original.npy")

# 2. 모델
def autoencoder(hidden_layer_size):
    # 인코더
    input_img = Input(shape=(220, 220, 3))
    # (28, 28, 1) -> (28, 28, hidden_layer_size)
    x = Conv2D(hidden_layer_size, (3, 3), activation='relu', padding='same')(input_img)
    # (28, 28, hidden_layer_size) -> (14, 14, hidden_layer_size)
    x = MaxPooling2D((2, 2), padding='same')(x)
    # (14, 14, hidden_layer_size) -> (14, 14, hidden_layer_size/2)
    x = Conv2D(hidden_layer_size/2, (3, 3), activation='relu', padding='same')(x)
    # (14, 14, hidden_layer_size/2) -> (7, 7, hidden_layer_size/2)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # 디코더
    # (7, 7, hidden_layer_size/2) -> (7, 7, hidden_layer_size/2)
    x = Conv2D(hidden_layer_size/2, (3, 3), activation='relu', padding='same')(encoded)
    # (7, 7, hidden_layer_size/2) -> (14, 14, hidden_layer_size/2)
    x = UpSampling2D((2, 2))(x)
    # (14, 14, hidden_layer_size/2) -> (14, 14, hidden_layer_size/2/2)
    x = Conv2D(hidden_layer_size/2/2, (3, 3), activation='relu', padding='same')(x)
    # (14, 14, hidden_layer_size/2/2) -> (28, 28, hidden_layer_size/2/2)
    x = UpSampling2D((2, 2))(x)
    # (28, 28, hidden_layer_size/2/2) -> (28, 28, 1)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    model = Model(input_img, decoded)
    return model