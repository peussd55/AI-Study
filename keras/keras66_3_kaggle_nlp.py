### <<30>>

import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Embedding, GRU, SimpleRNN, Bidirectional
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

# 1. 데이터(컬럼 한개로 병합해서하기)
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

# text 컬럼에 keyword + location 합치기
def merge_text(df):
    df['keyword'] = df['keyword'].fillna('unknown')
    df['location'] = df['location'].fillna('unknown').str.lower()
    df['text'] = df['text'] + ' keword: ' + df['keyword'] + \
        ' location: ' + df['location']
    return df

train = merge_text(train_csv)
test = merge_text(test_csv)

# 데이터 분리
x = train['text']
y = train['target']

## 토크나이징
token = Tokenizer()
token.fit_on_texts(x)
print(token.word_index)
# {'location': 1, 'keword': 2, 'co': 3, 't': 4, 'http': 5
#  ...
#  'guy': 982, 'jobs': 983, "the
vocab_size = len(token.word_index)
print(f"단어 사전의 크기: {vocab_size}")    # 24591

x_tokenized = token.texts_to_sequences(x)
# print(x)
# 1. maxlen 결정을 위한 길이 계산
token_lengths = [len(seq) for seq in x_tokenized]

# 2. maxlen을 95%지점 또는 최대길이로 선택
maxlen = int(np.percentile(token_lengths, 95))      # 31
# maxlen = max(token_lengths)                       # 40
print(f"선택된 maxlen: {maxlen}")

## 패딩
from tensorflow.keras.preprocessing.sequence import pad_sequences
padding_x = pad_sequences(x_tokenized,
              padding='pre',        # <-> 'post' : 패딩을 앞 또는 뒤에 넣는다. default : pre
              maxlen=maxlen,
              truncating='pre',     # <-> 'post : maxlen을 넘어가는 단어 자를때 어디서부터 자를지. default : pre
              )
print(padding_x.shape)  # (7613, 31)
print(padding_x)

x = padding_x

x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y,
    train_size=0.7,
    random_state=8282,
    shuffle=True,
    stratify=y,
)

print("x_train shape:", x_train.shape)  # (6090, 31)
print("x_test shape:", x_test.shape)    # (1523, 31)
print("y_train shape:", y_train.shape)  # (6090,)
print("y_test shape:", y_test.shape)    # (1523,)

# 2. 모델
model = Sequential()
model.add(Embedding(input_dim=vocab_size + 1,   # 단어 사전 크기 + 1 (패딩)
                    output_dim=256,              # 출력 차원 (하이퍼파라미터)
                    input_length=maxlen))       # 입력 시퀀스 길이 (패딩 길이)
model.add(Bidirectional(LSTM(128, return_sequences=True))) # return_sequences=True는 다음 LSTM을 위해
model.add(Bidirectional(LSTM(64)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[F1Score()])

es = EarlyStopping( 
    monitor = 'val_f1_score',       
    mode = 'max',               
    patience=50,             
    restore_best_weights=True,  
    verbose=1,
)

hist = model.fit(x_train, y_train,
          epochs=100,
          batch_size=64,
          validation_data=(x_test, y_test),
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
results = model.evaluate(x_train, y_train)
print('loss/f1_score :', results)    #

from sklearn.metrics import f1_score
y_pred_prob = model.predict(x_test)

# 0.1부터 0.9까지 다양한 임계값으로 F1 Score 계산
best_f1 = 0
best_threshold = 0.5
for threshold in np.arange(0.1, 0.9, 0.01):
    preds = (y_pred_prob > threshold).astype(int)
    f1 = f1_score(y_test, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
        
print(f"최적 임계값: {best_threshold}, 최대 F1 Score: {best_f1}")


y_pred = (y_pred_prob > best_threshold).astype(int)
f1 = f1_score(y_test, y_pred)
print('f1_score :', f1) # 0.7394136807817591

##### csv 파일 만들기 #####
# 1. 제출용 테스트 데이터(test_csv)에서 텍스트 추출
x_submission_text = test['text']

# 2. 훈련 때 사용했던 'token'으로 토큰화
submission_tokenized = token.texts_to_sequences(x_submission_text)

# 3. 훈련 때 사용했던 'maxlen'으로 패딩
submission_padded = pad_sequences(submission_tokenized,
                                  padding='pre',
                                  maxlen=maxlen,
                                  truncating='pre')
                                  
# 4. 전처리가 완료된 숫자 배열로 예측 수행
y_submit = model.predict(submission_padded)

# best_threshold로 저장
y_submit = (y_submit > 0.5).astype(int)
submission_csv['target'] = y_submit
submission_csv.to_csv(f'{path}submission_{f1}.csv', index=False)  # 인덱스 생성옵션 끄면 첫번째 컬럼이 인덱스로 지정됨.(안끄면 인덱스 자동생성)

plt.show()