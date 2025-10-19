### <<30>>

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd

# 1. 데이터
(x_train, y_train), (x_test, y_test)  = imdb.load_data(
    num_words=10000,
    
)

print(x_train)
print(y_train)  # [1 0 0 ... 0 1 0]
print(x_train.shape, y_train.shape) # (25000,) (25000,)
print(np.unique(y_train, return_counts=True))   # (array([0, 1], dtype=int64), array([12500, 12500], dtype=int64)) : 이진분류, 데이터균형 -> accuracy 지표 써도됨
print(pd.value_counts(y_train))
# 1    12500
# 0    12500
# dtype: int64

# x_train은 이미 tokenizing이 되었기때문에 이 코드로 토큰 갯수를 반환한다.
print('영화평의 최대길이 : ', max(len(i) for i in x_train) )            # 2494
print('영화평의 최소길이 : ', min(len(i) for i in x_train) )            # 11
print('영화평의 평균길이 : ', sum(map(len, x_train))/len(x_train))      # 238.71364

# 패딩
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train,
              padding='pre',        # <-> 'post' : 패딩을 앞 또는 뒤에 넣는다. default : pre
              maxlen=2494,          # 시퀀스(문장)의 길이. 이게 무조건 길다고 좋은게아니다. 0으로 채워진 긴 시퀀스를 가진 x_train를 학습하면 메모리낭비가 심하고 성능이 저하될수있다.
              truncating='pre',     # <-> 'post : maxlen을 넘어가는 단어 자를때 어디서부터 자를지. default : pre
              )
x_test = pad_sequences(x_test,
              padding='pre',        # <-> 'post' : 패딩을 앞 또는 뒤에 넣는다. default : pre
              maxlen=2494,          # 시퀀스(문장)의 길이. 0으로 채워진 긴 시퀀스를 가진 x_train를 학습하면 메모리낭비가 심하고 성능이 저하될수있다.
              truncating='pre',     # <-> 'post : maxlen을 넘어가는 단어 자를때 어디서부터 자를지. default : pre
              )
print(x_train.shape)  # (25000, 2494)
print(x_train)

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

model = Sequential()
# 임베딩은 3차원을 출력으로 가진다.
# 임베딩 1.
# model.add(Embedding(input_dim=31, output_dim=100, input_length=5, ))    
    # input_dim : 단어사전의 갯수(말뭉치의 갯수), 0인덱스부터 생성하기때문에 word_index 크기의 + 1 만큼 잡아줘야한다.
    # 임베딩은 0 인덱스부터 차원을 생성하는데 word_index는 1부터 시작하기때문에 30으로 지정하면 word_index 마지막 사전단어가 학습데이터에서 삭제된다. 따라서 31을 해야한다.
    # output_dim : 다음 레이어로 전달하는 노드의 갯수(조절가능) 
    # input_length : (N, 5), 컬럼의 갯수, 문장의 시퀀스의 갯수
    
# 임베딩 2.
# model.add(Embedding(input_dim=31, output_dim=100, ))    
    # 임베딩에서는 input_length를 명시안해도 알아서 조절해줌. 그러나 정확하지않은 값을 넣으면 에러가 난다.
    
# 임베딩 3.
# model.add(Embedding(input_dim=13300, output_dim=100, ))      #  단어사전 갯수를 크게 한다. 모든 단어를 학습하지만 메모리 낭비가 생긴다.
model.add(Embedding(input_dim=10000, output_dim=100, ))         # 오류는 안나나 학습가능한 사전 단어 갯수가 줄어들어서 성능 저하(단, cpu연산시에는 엄격하기때문에 오류가 난다.)
# output_dim 은 통상적으로 input_dim의 ^0.25 승 : 10000 -> 10인데 너무 작으면 더 늘린다.
# input_length은 Embedding이 디폴트로 길이를 알아서 잡아주므로 잘모르겠어면 냅둘것(1 또는 정확한 값만 받음)

# 임베딩 4.
# model.add(Embedding(31, 100, ))                   # 에러 X
# model.add(Embedding(31, 100, 5))                  # 에러 O
# model.add(Embedding(31, 100, input_length=5,))    # 에러 X
# model.add(Embedding(31, 100, input_length=6))     # 에러 O
# model.add(Embedding(31, 100, input_length=1,))    # 에러 X (1은 먹힌다)

model.add(LSTM(16))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train,
            validation_data=(x_test, y_test),
          epochs=10,
          batch_size=128,
          )

# 4. 평가, 예측
results = model.evaluate(x_train, y_train)
print('loss/acc :', results)    # loss/acc : [0.03724878281354904, 0.990559995174408]

from sklearn.metrics import accuracy_score

y_pred = model.predict(x_test)
y_pred = np.round(y_pred)
acc = accuracy_score(y_test, y_pred)
print('acc :', acc)     # 0.85704