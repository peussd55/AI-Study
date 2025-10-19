### <<55>>

import numpy as np 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
import tensorflow as tf  
import random 
import pandas as pd
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

SEED = 333
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from tensorflow.keras.applications import (
    VGG16, VGG19,
    ResNet50, ResNet50V2, ResNet101, ResNet152, ResNet152V2,
    DenseNet121, DenseNet169, DenseNet201,
    InceptionV3, InceptionResNetV2,
    MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large,
    NASNetMobile, NASNetLarge,
    EfficientNetB0, EfficientNetB1, EfficientNetB3,
    Xception
)

model_list = [
    VGG16(include_top=False, input_shape=(32, 32, 3)),
    VGG19(include_top=False, input_shape=(32, 32, 3)),
    ResNet50(include_top=False, input_shape=(32, 32, 3)),
    ResNet50V2(include_top=False, input_shape=(32, 32, 3)),
    ResNet101(include_top=False, input_shape=(32, 32, 3)),
    ResNet152(include_top=False, input_shape=(32, 32, 3)),
    ResNet152V2(include_top=False, input_shape=(32, 32, 3)),
    DenseNet121(include_top=False, input_shape=(32, 32, 3)),
    DenseNet169(include_top=False, input_shape=(32, 32, 3)),
    DenseNet201(include_top=False, input_shape=(32, 32, 3)),
    # InceptionV3(include_top=False, input_shape=(32, 32, 3)),
    # InceptionResNetV2(include_top=False, input_shape=(32, 32, 3)),
    MobileNet(include_top=False, input_shape=(32, 32, 3)),
    MobileNetV2(include_top=False, input_shape=(32, 32, 3)),
    MobileNetV3Small(include_top=False, input_shape=(32, 32, 3)),
    MobileNetV3Large(include_top=False, input_shape=(32, 32, 3)),
    # NASNetMobile(include_top=False, input_shape=(32, 32, 3)),
    # NASNetLarge(include_top=False, input_shape=(32, 32, 3)),
    EfficientNetB0(include_top=False, input_shape=(32, 32, 3)),
    EfficientNetB1(include_top=False, input_shape=(32, 32, 3)),
    EfficientNetB3(include_top=False, input_shape=(32, 32, 3)),
    # Xception(include_top=False, input_shape=(32, 32, 3)),
]

# 1. 데이터 전처리
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.
x_test = x_test / 255.

# 2. 반복할 backbone 모델 리스트
# (include_top, input_shape 설정된 model_list가 이미 만들어져 있어야 함!)
# 예: model_list = [VGG16(...), VGG19(...), ...]
# name_list = ["VGG16", "VGG19", ...]  # 모델명도 따로 만들어두면 분석에 유리

for i, backbone in enumerate(model_list):
    for freeze in [True]:   # 가중치 동결: True/False
        for use_gap in [True]:  # GAP, Flatten 비교

            # 기본 이름 생성
            model_name = backbone.name
            freeze_name = "FT" if freeze else "NoFT"
            gap_name = "GAP" if use_gap else "Flatten"
            print(f"=== {model_name} | Freeze: {freeze} | GAP: {use_gap} ===")

            # 가중치 동결 설정
            backbone.trainable = not freeze  # True면 Fine-tuning, False면 동결

            # 모델 구성
            model = Sequential()
            model.add(backbone)
            if use_gap:
                model.add(GlobalAveragePooling2D())
            else:
                model.add(Flatten())
            model.add(Dense(256))
            model.add(Dense(128))
            model.add(Dense(10, activation='softmax'))

            # 컴파일 및 콜백
            model.compile(
                loss='sparse_categorical_crossentropy',
                optimizer='adam',
                metrics=['acc']
            )
            es = EarlyStopping(
                monitor='val_loss',
                mode='min',
                patience=30,
                verbose=1,
                restore_best_weights=True,
            )

            # 학습
            start = time.time()
            hist = model.fit(
                x_train, y_train,
                epochs=200,
                batch_size=128,
                validation_split=0.2,
                verbose=0,
                callbacks=[es]
            )
            end = time.time()

            # 평가
            loss, acc_metric = model.evaluate(x_test, y_test, verbose=0)
            y_pred = model.predict(x_test, verbose=0)
            y_pred_argmax = np.argmax(y_pred, axis=1)
            acc = accuracy_score(y_test, y_pred_argmax)

            print(f"{model_name} / Freeze:{freeze} / GAP:{use_gap} --- accuracy: {acc:.4f}, elapsed: {round(end - start, 2)} 초\n")

            # 필요시 로그 저장 등 추가
            # === Functional | Freeze: True | GAP: True ===
            # Functional / Freeze:True / GAP:True --- accuracy: 0.5865, elapsed: 368.38 초
            
    """
    전이학습모델 성능이 잘 안나오는 이유 : 대부분 가져온 모델의 학습된 shape size와 내가 돌릴 데이터의 shape size가 차이가 커서 떨어지는 요인이 가장 큼. 
    """