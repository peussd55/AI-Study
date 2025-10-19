### <<29>>

from tensorflow.keras.datasets import reuters
import numpy as np

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=1000,  # 단어사전의 개수, 빈도수가 높은 단어 순으로 1000개 뽑는다.
    test_split=0.2,
    # maxlen=200,     # 최대 단어 길이가 100개까지 있는 문장
)
print(x_train[0])
print(x_train)
print(x_train.shape, x_test.shape)  # (8982,) (2246,)
print(y_train.shape, y_test.shape)  # (8982,) (2246,)
print(y_train[0])   # 3
print(y_train)      # [ 3  4  3 ... 25  3 25]
print(np.unique(y_train))
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]

# 엥그러바 (8982, 100)로 변환 / acc > 0.7
print(type(x_train))    # nparray
print(type(x_train[0])) # list

print('뉴스기사의 최대길이 : ', max(len(i) for i in x_train) )  # 2376
print('뉴스기사의 최소길이 : ', min(len(i) for i in x_train) )  # 13
print('뉴스기사의 평균길이 : ', sum(map(len, x_train))/len(x_train))    # 145.33

# 패딩
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train,
              padding='pre',        # <-> 'post' : 패딩을 앞 또는 뒤에 넣는다. default : pre
              maxlen=250,          # 시퀀스(문장)의 길이. 이게 무조건 길다고 좋은게아니다. 0으로 채워진 긴 시퀀스를 가진 x_train를 학습하면 메모리낭비가 심하고 성능이 저하될수있다.
              truncating='pre',     # <-> 'post : maxlen을 넘어가는 단어 자를때 어디서부터 자를지. default : pre
              )
x_test = pad_sequences(x_test,
              padding='pre',        # <-> 'post' : 패딩을 앞 또는 뒤에 넣는다. default : pre
              maxlen=250,          # 시퀀스(문장)의 길이. 이게 무조건 길다고 좋은게아니다. 0으로 채워진 긴 시퀀스를 가진 x_train를 학습하면 메모리낭비가 심하고 성능이 저하될수있다.
              truncating='pre',     # <-> 'post : maxlen을 넘어가는 단어 자를때 어디서부터 자를지. default : pre
              )
print(x_train.shape)  # (8982, 2376)
print(x_train)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

# (중요) 다중분류모델에서는 y레이블이 정수형태([0 1 2 3 ... 45])이면 
# 1. 손실함수를 sparse_categorical_crossentropy 쓰던가 
# 2. y를 원핫인코딩한 후, 손실함수 categorical_crossentropy를 써야한다.

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
# model.add(Embedding(input_dim=13300, output_dim=100, ))      #  단어사전 갯수를 크게 한다. 모든 단어를 학습하지만 메모리 낭비가 생긴다.
model.add(Embedding(input_dim=1001, output_dim=100, ))         # 오류는 안나나 학습가능한 사전 단어 갯수가 줄어들어서 성능 저하(단, cpu연산시에는 엄격하기때문에 오류가 난다.)
# output_dim 은 통상적으로 input_dim의 ^0.25 승 : 10000 -> 10인데 너무 작으면 더 늘린다.
# input_length은 Embedding이 디폴트로 길이를 알아서 잡아주므로 잘모르겠어면 냅둘것(1 또는 정확한 값만 받음)

# 임베딩 4.
# model.add(Embedding(31, 100, ))                   # 에러 X
# model.add(Embedding(31, 100, 5))                  # 에러 O
# model.add(Embedding(31, 100, input_length=5,))    # 에러 X
# model.add(Embedding(31, 100, input_length=6))     # 에러 O
# model.add(Embedding(31, 100, input_length=1,))    # 에러 X (1은 먹힌다)

model.add(LSTM(16))
model.add(Dense(46, activation='softmax'))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train,
            validation_data=(x_test, y_test),
          epochs=30,
          batch_size=64,
          )

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss:', results[0])              # loss: 1.2571403980255127
print('(categorical)acc:', results[1])  # (categorical)acc: 0.703027606010437

from sklearn.metrics import accuracy_score

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)  # y_pred가 2차원으로 출력되어서 변환해줘야함.
print(y_pred)   # [ 3  1 16 ...  3  3 11]
print(y_test)   # [ 3 10  1 ...  3  3 24]

# acc
acc = accuracy_score(y_test, y_pred)
print('acc :', acc) # acc : 0.7030276046304541
