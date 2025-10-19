### <<30>>

# https://www.kaggle.com/competitions/nlp-getting-started/overview

import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Embedding, GRU, SimpleRNN, Bidirectional, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer

import tensorflow as tf
from tensorflow.keras import backend as K
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.f1score = self.add_weight(name='f1_score', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred > 0.5, tf.bool)

        true_positives = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        true_positives = tf.cast(true_positives, self.dtype)
        count_true_positives = tf.reduce_sum(true_positives)

        possible_positives = tf.cast(y_true, self.dtype)
        count_possible_positives = tf.reduce_sum(possible_positives)

        predicted_positives = tf.cast(y_pred, self.dtype)
        count_predicted_positives = tf.reduce_sum(predicted_positives)

        precision = count_true_positives / (count_predicted_positives + K.epsilon())
        recall = count_true_positives / (count_possible_positives + K.epsilon())
        f1_cal = 2*(precision*recall)/(precision + recall + K.epsilon())

        self.count.assign_add(1)
        a = 1.0 / self.count
        b = 1.0 - a
        self.f1score.assign(a*f1_cal+b*self.f1score)

    def result(self):
        return self.f1score

    def reset_state(self):
        self.f1score.assign(0)
        self.count.assign(0)

# 1. 데이터 (앙상블로 하기)
path = './_data/kaggle/nlp/'

# 맨 앞, 공통 컬럼 datetime을 인덱스 컬럼으로 지정
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

print(train_csv.shape, test_csv.shape, submission_csv.shape)  # (7613, 4) (3263, 3) (3263, 2)
print(train_csv.head())
print(test_csv.head())

print(train_csv.columns)
print(test_csv.columns)

# 결측치 확인
print(train_csv.info())
print(train_csv.isna().sum()) 
#  0   keyword   7552 non-null   object : 결측치 있음
#  1   location  5080 non-null   object : 결측치 있음
#  2   text      7613 non-null   object
#  3   target    7613 non-null   int64
print(test_csv.info())
print(test_csv.isna().sum())     
#  0   keyword   3237 non-null   object : 결측치 있음
#  1   location  2158 non-null   object : 결측치 있음
#  2   text      3263 non-null   object

print(train_csv['target'].value_counts())
# 0    4342
# 1    3271
x = train_csv.drop('target', axis=1)
y = train_csv['target']

# keyword, location 컬럼 원핫인코딩
# 순서가 없는 범주형 데이터(예: 색상, 지역명 등)는 One-Hot Encoding을,
# 순서가 있는 데이터(예: 등급, 크기 등)는 Label Encoding을 사용하는 것이 일반적입니다.

from sklearn.preprocessing import OneHotEncoder
# 결측치 처리
train_csv['keyword'] = train_csv['keyword'].fillna('unknown').str.lower()
# One-Hot Encoding
keyword_ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')  # sparse=False : 밀집행렬(nparray)로 반환  handle_unknown='ignore' 훈련데이터에없는 값이 테스트데이터에있으면 0으로 처리함.
x_train_keyword = keyword_ohe.fit_transform(train_csv[['keyword']])
# test data 결측치처리, 원핫인코딩
test_csv['keyword'] = test_csv['keyword'].fillna('unknown').str.lower()
x_test_keyword = keyword_ohe.transform(test_csv[['keyword']])

# 결과 shape: (샘플 수, 고유 키워드 개수)
print(x_train_keyword)
print(x_train_keyword.shape)  # (7613, 222)
print(x_test_keyword)
print(x_test_keyword.shape)  # (3263, 222)

# 결측치 처리
train_csv['location'] = train_csv['location'].fillna('unknown').str.lower()
# One-Hot Encoding
location_ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
x_train_location = location_ohe.fit_transform(train_csv[['location']])
# test data 결측치처리, 원핫인코딩
test_csv['location'] = test_csv['location'].fillna('unknown').str.lower()
x_test_location = location_ohe.transform(test_csv[['location']])

# 결과 shape: (샘플 수, 고유 location 개수)
print(x_train_location)
print(x_train_location.shape)
print(x_test_location)
print(x_test_location.shape)    # (3263, 3233)

# text 컬럼 토크나이징
x_text_tokenizing = train_csv['text']
token = Tokenizer()
token.fit_on_texts(x_text_tokenizing)
print(token.word_index)
# {'location': 1, 'keword': 2, 'co': 3, 't': 4, 'http': 5
#  ...
#  'guy': 982, 'jobs': 983, "the
vocab_size = len(token.word_index)
print(f"단어 사전의 크기: {vocab_size}")    # 22700

x_tokenized = token.texts_to_sequences(x_text_tokenizing)
# print(x)
# 1. maxlen 결정을 위한 길이 계산
token_lengths = [len(seq) for seq in x_tokenized]

# 2. maxlen을 95%지점 또는 최대길이로 선택
maxlen = int(np.percentile(token_lengths, 95))      # 25
# maxlen = max(token_lengths)                       # 33
print(f"선택된 maxlen: {maxlen}")

# 패딩
from tensorflow.keras.preprocessing.sequence import pad_sequences
padding_x = pad_sequences(x_tokenized,
              padding='pre',        # <-> 'post' : 패딩을 앞 또는 뒤에 넣는다. default : pre
              maxlen=maxlen,
              truncating='pre',     # <-> 'post : maxlen을 넘어가는 단어 자를때 어디서부터 자를지. default : pre
              )
print(padding_x.shape)  # (7613, 31)
print(padding_x)
x_train_text = padding_x
# 테스트데이터 토크나이징, 패딩
test_text = token.texts_to_sequences(test_csv['text'])
test_text = pad_sequences(test_text, maxlen=maxlen, padding='pre', truncating='pre')
x_test_text = test_text

# 결과 shape: (샘플 수, 고유 location 개수)
print(x_train_text)
print(x_train_text.shape)
print(x_test_text)
print(x_test_text.shape)    # (3263, 25)

(x_train_text, x_val_text, 
 x_train_keyword, x_val_keyword, 
 x_train_location, x_val_location, 
 y_train, y_val) = train_test_split(
    x_train_text, x_train_keyword, x_train_location, y,
    test_size=0.2
)
  

print("x_train_text shape:", x_train_text.shape)            # (6090, 25)
print("x_val_text shape:", x_val_text.shape)              # (1523, 25)
print("x_train_keyword shape:", x_train_keyword.shape)      # (6090, 222)
print("x_val_keyword shape:", x_val_keyword.shape)        # (1523, 222)
print("x_train_location shape:", x_train_location.shape)    # (6090, 3233)
print("x_val_location shape:", x_val_location.shape)      # (1523, 3233)
print("y_train shape:", y_train.shape)                      # (6090,)
print("y_val shape:", y_val.shape)                        # (1523,)
                                                             
# 2. 모델
# 2.1. 입력 레이어 정의
input_text = Input(shape=(maxlen,), name='text_input')  # 텍스트 입력 (maxlen=25)
input_keyword = Input(shape=(222,), name='keyword_input')  # 키워드 입력 (222차원)
input_location = Input(shape=(3233,), name='location_input')  # 위치 입력 (3233차원)

# 2.2. 텍스트 처리 브랜치 (LSTM 사용)
x_text = Embedding(input_dim=vocab_size+1, output_dim=50)(input_text)
x_text = LSTM(64)(x_text)
x_text = Dense(32, activation='relu', name='text_dense')(x_text)

# 2.3. 키워드 처리 브랜치
x_keyword = Dense(64, activation='relu', name='keyword_dense1')(input_keyword)
x_keyword = Dense(32, activation='relu', name='keyword_dense2')(x_keyword)

# 2.4. 위치 처리 브랜치 (고차원 데이터 처리)
x_location = Dense(128, activation='relu', name='location_dense1')(input_location)  # 차원 축소
x_location = Dense(64, activation='relu', name='location_dense2')(x_location)
x_location = Dense(32, activation='relu', name='location_dense3')(x_location)

# 2.5. 모델 앙상블
concat = Concatenate(name='concat_layer')([x_text, x_keyword, x_location])
x = Dense(64, activation='relu', name='ensemble_dense1')(concat)
x = Dense(32, activation='relu', name='ensemble_dense2')(x)
output = Dense(1, activation='sigmoid', name='output')(x)

# 2.6. 모델 생성
model = Model(inputs=[input_text, input_keyword, input_location], outputs=output)
model.summary()

model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['accuracy', F1Score()],
              )

es = EarlyStopping( 
    monitor = 'val_f1_score',       
    mode = 'max',               
    patience=50,             
    restore_best_weights=True,  
    verbose=1,
)

hist = model.fit(
        [x_train_text, x_train_keyword, x_train_location],
        y_train,
          epochs=100,
          batch_size=64,
            validation_data=(
                [x_val_text, x_val_keyword, x_val_location],
                y_val
            ),
          callbacks=[es],
          )

import matplotlib.pyplot as plt
## 그래프 그리기
plt.figure(figsize=(18, 5))
# 첫 번째 그래프
plt.subplot(1, 2, 1)  # (행, 열, 위치)
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.grid()

# 두 번째 그래프
plt.subplot(1, 2, 2)
plt.plot(hist.history['f1_score'], c='green', label='f1_score')
plt.plot(hist.history['val_f1_score'], c='orange', label='val_f1_score')
plt.title('F1_score')
plt.xlabel('epochs')
plt.ylabel('f1_score')
plt.legend()
plt.grid()

plt.tight_layout()  # 간격 자동 조정

# 4. 평가, 예측
results = model.evaluate(
    [x_val_text, x_val_keyword, x_val_location],  # 검증 데이터 사용
    y_val,
    verbose=0
)
print('검증 손실 및 메트릭:', results)  # [0.4271385371685028, 0.818122148513794, 0.7627373933792114]

# 검증 데이터 예측
y_pred_prob = model.predict([x_val_text, x_val_keyword, x_val_location])
from sklearn.metrics import f1_score
# 최적 임계값 탐색
best_f1 = 0
best_threshold = 0.5
for threshold in np.arange(0.1, 0.9, 0.01):
    preds = (y_pred_prob > threshold).astype(int)
    f1 = f1_score(y_val, preds)  # y_val 사용
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"최적 임계값: {best_threshold:.4f}, 최대 F1 점수: {best_f1:.4f}")

# 최종 F1 점수 계산
y_pred = (y_pred_prob > best_threshold).astype(int)
final_f1 = f1_score(y_val, y_pred)  # y_val 사용
print('최종 검증 F1 점수:', final_f1)
# 0.7820512820512822
# 0.7680608365019012
# 0.7722448979591836
# 0.7782945736434108
# 0.7452907452907453
# -> 로컬 점수는 일반 방식보다 더 높으나 public score는 비슷하거나 좀 더 떨어짐

##### 제출 파일 생성 #####
# 이미 전처리된 테스트 데이터 사용
y_submit_prob = model.predict([x_test_text, x_test_keyword, x_test_location])
y_submit = (y_submit_prob > best_threshold).astype(int)

# 제출 파일 저장
submission_csv['target'] = y_submit
submission_csv.to_csv(f'{path}submission_{final_f1:.4f}.csv', index=False)

plt.show()