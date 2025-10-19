### <<28>>

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import time
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# 1. 데이터
docs = [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화에요',
    '추천하고 싶은 영화입니다.', '한 번 더 보고 싶어요.', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다', '참 재밌네요.',
    '석준이 바보', '준희 잘생겼다', '이삭이 또 구라친다',
]

labels = np.array([1,1,1, 1,1,0, 0,0,0, 0,0,1, 0,1,0])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5, '영화에요': 6, '추천하고': 7, '싶은': 8, 
# '영화입니다': 9, '한': 10, '번': 11, '더': 12, '보고': 13, '싶어요': 14, '글쎄': 15, '별로에요': 16, 
# '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, '재미없어요': 21, '재미없다': 22, '재밌네요': 23, 
# '석준이': 24, '바보': 25, '준희': 26, '잘생겼다': 27, '이삭이': 28, '또': 29, '구라친다': 30}

x = token.texts_to_sequences(docs)
print(x)
# [[2, 3], [1, 4], [1, 5, 6], [7, 8, 9], [10, 11, 12, 13, 14], 
# [15], [16], [17, 18], [19, 20], [21], [2, 22], [1, 23], 
# [24, 25], [26, 27], [28, 29, 30]]
# -> 이 리스트안에 있는 리스트는 길이가 제각각이라 nparray로 바로 변환 못한다. 길이를 최대값에 맞추기위해 0으로 padding해야한다.

### 패딩 ###
from tensorflow.keras.preprocessing.sequence import pad_sequences
padding_x = pad_sequences(x,
              padding='pre',        # <-> 'post' : 패딩을 앞 또는 뒤에 넣는다. default : pre
              maxlen=5,
              truncating='pre',     # <-> 'post : maxlen을 넘어가는 단어 자를때 어디서부터 자를지. default : pre
              )
print(padding_x.shape)
print(padding_x)
"""
(15, 5)                             
[[ 0  0  0  2  3]
 [ 0  0  0  1  4]
 [ 0  0  1  5  6]
 [ 0  0  7  8  9]
 [10 11 12 13 14]
 [ 0  0  0  0 15]
 [ 0  0  0  0 16]
 [ 0  0  0 17 18]
 [ 0  0  0 19 20]
 [ 0  0  0  0 21]
 [ 0  0  0  2 22]
 [ 0  0  0  1 23]
 [ 0  0  0 24 25]
 [ 0  0  0 26 27]
 [ 0  0 28 29 30]]

(15, 4)
[[ 0  0  2  3]
 [ 0  0  1  4]
 [ 0  1  5  6]
 [ 0  7  8  9]
 [11 12 13 14]
 [ 0  0  0 15]
 [ 0  0  0 16]
 [ 0  0 17 18]
 [ 0  0 19 20]
 [ 0  0  0 21]
 [ 0  0  2 22]
 [ 0  0  1 23]
 [ 0  0 24 25]
 [ 0  0 26 27]
 [ 0 28 29 30]]
 
 (15, 4), pre, post
[[ 0  0  2  3]
 [ 0  0  1  4]
 [ 0  1  5  6]
 [ 0  7  8  9]
 [10 11 12 13]
 [ 0  0  0 15]
 [ 0  0  0 16]
 [ 0  0 17 18]
 [ 0  0 19 20]
 [ 0  0  0 21]
 [ 0  0  2 22]
 [ 0  0  1 23]
 [ 0  0 24 25]
 [ 0  0 26 27]
 [ 0 28 29 30]]
 
 (15, 4), post, post
[[ 2  3  0  0]
 [ 1  4  0  0]
 [ 1  5  6  0]
 [ 7  8  9  0]
 [10 11 12 13]
 [15  0  0  0]
 [16  0  0  0]
 [17 18  0  0]
 [19 20  0  0]
 [21  0  0  0]
 [ 2 22  0  0]
 [ 1 23  0  0]
 [24 25  0  0]
 [26 27  0  0]
 [28 29 30  0]]
 """
 
# (15,5) dnn모델구성 (토크나이징O, 패딩O, 원핫X, 임베딩X)
# [이삭이 잘 생겼다] -> 긍정 또는 부정 (이진분류) 예측

# 1. 데이터
x = padding_x
y = labels
print(x)    
# [[ 0  0  0  2  3]
#  [ 0  0  0  1  4]
#  [ 0  0  1  5  6]
#  [ 0  0  7  8  9]
#  [10 11 12 13 14]
#  [ 0  0  0  0 15]
#  [ 0  0  0  0 16]
#  [ 0  0  0 17 18]
#  [ 0  0  0 19 20]
#  [ 0  0  0  0 21]
#  [ 0  0  0  2 22]
#  [ 0  0  0  1 23]
#  [ 0  0  0 24 25]
#  [ 0  0  0 26 27]
#  [ 0  0 28 29 30]]
print(y)    # [1 1 1 1 1 0 0 0 0 0 0 1 0 1 0]
print(x.shape, y.shape) # (15, 5), (15,)
text = '이삭이 잘 생겼다'
token2 = Tokenizer()
token2.fit_on_texts([text])
print(token2.word_index)    # {'이삭이': 1, '잘': 2, '생겼다': 3}
print(token2.word_counts)   # OrderedDict([('이삭이', 1), ('잘', 1), ('생겼다', 1)])
x_pred = token2.texts_to_sequences([text])
print(x_pred)   # [[1, 2, 3]]

padding_x_pred = pad_sequences(x_pred,
              padding='pre',        # <-> 'post' : 패딩을 앞 또는 뒤에 넣는다. default : pre
              maxlen=5,
              truncating='pre',     # <-> 'post : maxlen을 넘어가는 단어 자를때 어디서부터 자를지. default : pre
              )
print(padding_x_pred.shape) # (1, 5)
print(padding_x_pred)   # [[0 0 1 2 3]]


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,   
)

# 2. 모델구성
model = Sequential()
input1 = Input(shape=(5,))
dense1 = Dense(50, activation='relu',)(input1)
dense2 = Dense(40, activation='relu',)(dense1)
dense3 = Dense(30, activation='relu',)(dense2)
dense4 = Dense(20, activation='relu',)(dense3)
output1 = Dense(1, activation='sigmoid',)(dense4)
model = Model(inputs=input1, outputs=output1)
model.summary()

# 3. 컴파일 , 훈련
model.compile(loss='binary_crossentropy', optimizer='adam')

hist = model.fit(x_train, y_train,
                epochs=100, 
                batch_size=32, 
                verbose=1, 
                validation_data=(x_test, y_test),
                ) 

# 4.평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_pred = model.predict(padding_x_pred)
print(y_pred)
y_pred = np.round(y_pred)
print(y_pred)   # [[1.]]
