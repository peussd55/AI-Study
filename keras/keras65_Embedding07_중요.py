### <<29>>

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import time
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
docs = [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화에요',
    '추천하고 싶은 영화입니다.', '한 번 더 보고 싶어요.', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다', '참 재밌네요.',
    '석준이 바보', '준희 잘생겼다', '이상이 또 구라친다',
]

labels = np.array([1,1,1, 1,1,0, 0,0,0, 0,0,1, 0,1,0])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5, '영화에요': 6, '추천하고': 7, '싶은': 8, 
# '영화입니다': 9, '한': 10, '번': 11, '더': 12, '보고': 13, '싶어요': 14, '글쎄': 15, '별로에요': 16, 
# '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, '재미없어요': 21, '재미없다': 22, '재밌네요': 23, 
# '석준이': 24, '바보': 25, '준희': 26, '잘생겼다': 27, '이상이': 28, '또': 29, '구라친다': 30}

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
            #   truncating='pre',     # <-> 'post : maxlen을 넘어가는 단어 자를때 어디서부터 자를지. default : pre
              )
print(padding_x.shape)  # (15, 5)
print(padding_x)
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
#################################################################################################################
# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

model = Sequential()
# 임베딩은 3차원을 출력으로 가진다.
# 임베딩 1.
# model.add(Embedding(input_dim=31, output_dim=100, input_length=5, ))    
    # input_dim : 단어사전의 갯수(말뭉치의 갯수), 0인덱스부터 생성하기때문에 word_index 크기의 + 1 만큼 잡아줘야한다.
    # 임베딩은 0 인덱스부터 차원을 생성하는데 word_index는 1부터 시작하기때문에 30으로 지정하면 word_index 마지막 사전단어가 학습데이터에서 삭제된다. 따라서 31을 해야한다.
    # output_dim : 다음 레이어로 전달하는 노드의 갯수(조절가능), 단어 하나를 표현하는 밀집 벡터(dense vector)의 차원(크기) 
    # input_length : (N, 5), 컬럼의 갯수, 문장의 시퀀스의 갯수
    
# 임베딩 2.
# model.add(Embedding(input_dim=31, output_dim=100, ))    
    # 임베딩에서는 input_length를 명시안해도 알아서 조절해줌. 그러나 정확하지않은 값을 넣으면 에러가 난다.
    
# 임베딩 3.
# model.add(Embedding(input_dim=133, output_dim=100, ))      # 단어사전 갯수를 크게 한다. 모든 단어를 학습하지만 메모리 낭비가 생긴다.
model.add(Embedding(input_dim=31, output_dim=100, ))         # 오류는 안나나 학습가능한 사전 단어 갯수가 줄어들어서 성능 저하(단, cpu연산시에는 엄격하기때문에 오류가 난다.)

# 임베딩 4.
# model.add(Embedding(31, 100, ))                   # 에러 X
# model.add(Embedding(31, 100, 5))                  # 에러 O
# model.add(Embedding(31, 100, input_length=5,))    # 에러 X
# model.add(Embedding(31, 100, input_length=6))     # 에러 O
# model.add(Embedding(31, 100, input_length=1,))    # 에러 X (1은 먹힌다)

model.add(LSTM(16))
model.add(Dense(1, activation='sigmoid'))

model.summary()
"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 embedding (Embedding)       (None, 5, 100)            3100 = 31*100

 lstm (LSTM)                 (None, 16)                7488

 dense (Dense)               (None, 1)                 17

=================================================================
Total params: 10,605
Trainable params: 10,605
Non-trainable params: 0
_________________________________________________________________
"""

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(padding_x, labels, epochs=100)

# 4. 평가, 예측
results = model.evaluate(padding_x, labels)
print('loss/acc :', results)

x_pred = ['이상이 참 잘생겼다']
x_pred = token.texts_to_sequences(x_pred)
x_pred = pad_sequences(x_pred, maxlen=5)
y_pred = model.predict(x_pred)
print('이상이 참 잘생겼다:', y_pred)
print('이상이 참 잘생겼다의 결과:', np.around(y_pred))