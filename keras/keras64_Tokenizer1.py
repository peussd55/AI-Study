### <<28>>

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

text = '오늘도 못생기고 영어를 디게 디게 디게 못 하는 일삭이는 재미없는 개그를 \
        마구 마구 마구 마구하면서 딴짓을 한다.'

token = Tokenizer()
token.fit_on_texts([text])
# token : 텍스트를 구성하는 가장 작은 단위(=한국어 문장의 어절)

print(token.word_index)     # 1부터시작함(0부터 시작X)
# {'디게': 1, '마구': 2, '오늘도': 3, '못생기고': 4, '영어를': 5, '못': 6, '하는': 7, '일삭이는': 8, '재미없는': 9, '개그를': 10, '마구하면서': 11, '딴짓을': 12, '한다': 13}
# 어절 단위로 라벨링
# 순위(라벨링) 기준 : 빈도 -> 순서

print(token.word_counts)
# OrderedDict([('오늘도', 1), ('못생기고', 1), ('영어를', 1), ('디게', 3), ('못', 1), ('하는', 1), ('일삭이는', 1), ('재미없는', 1), ('개그를', 1), ('마구', 3), ('마구하면서', 1), ('딴짓을', 1), ('한다', 1)])
# 반복 횟수출력

x = token.texts_to_sequences([text])
print(x)
# [[3, 4, 5, 1, 1, 1, 6, 7, 8, 9, 10, 2, 2, 2, 11, 12, 13]]
# index로 치환해서 문장구성

########### 원핫인코딩 3가지 만들기 ###########
print(len(x[0]))    # 17, list
# 1. 판다스
import pandas as pd
x_sequence1 = x[0]
x_sequence1 = pd.Series(x_sequence1)   # <class 'pandas.core.series.Series'>
x_sequence1 = pd.get_dummies(x_sequence1)
print("Pandas get_dummies:")
print(x_sequence1)
"""
    1   2   3   4   5   6   7   8   9   10  11  12  13
0    0   0   1   0   0   0   0   0   0   0   0   0   0
1    0   0   0   1   0   0   0   0   0   0   0   0   0
2    0   0   0   0   1   0   0   0   0   0   0   0   0
3    1   0   0   0   0   0   0   0   0   0   0   0   0
4    1   0   0   0   0   0   0   0   0   0   0   0   0
5    1   0   0   0   0   0   0   0   0   0   0   0   0
6    0   0   0   0   0   1   0   0   0   0   0   0   0
7    0   0   0   0   0   0   1   0   0   0   0   0   0
8    0   0   0   0   0   0   0   1   0   0   0   0   0
9    0   0   0   0   0   0   0   0   1   0   0   0   0
10   0   0   0   0   0   0   0   0   0   1   0   0   0
11   0   1   0   0   0   0   0   0   0   0   0   0   0
12   0   1   0   0   0   0   0   0   0   0   0   0   0
13   0   1   0   0   0   0   0   0   0   0   0   0   0
14   0   0   0   0   0   0   0   0   0   0   1   0   0
15   0   0   0   0   0   0   0   0   0   0   0   1   0
16   0   0   0   0   0   0   0   0   0   0   0   0   1
"""
print(x_sequence1.shape)      # (17, 13)
print(type(x_sequence1))      # <class 'pandas.core.frame.DataFrame'>

# 2. sklearn
from sklearn.preprocessing import OneHotEncoder
x_sequence2 = x[0]
x_sequence2 = np.array(x_sequence2)         # <class 'numpy.ndarray'>
x_sequence2 = x_sequence2.reshape(-1, 1)
ohe = OneHotEncoder(sparse_output=False)    # sparse_output : sklearn 1.3이상 문법. ohe가 인자로 받은 값을 nparray로 반환하게하기
x_sequence2 = ohe.fit_transform(x_sequence2)
print("Sklearn OneHotEncoder:")
print(x_sequence2)
"""
[[0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
"""
print(x_sequence2.shape)      # (17, 13)
print(type(x_sequence2))      # <class 'numpy.ndarray'>

# 3. keras
from tensorflow.keras.utils import to_categorical
x_sequence3 = x[0]
x_sequence3 = x_sequence3 - np.min(x_sequence3) # 조치 : 컬럼값을 인덱스 그대로 사용하기때문에 y클래스가 [1,2,3]이런식으로 되어있을때 앞에 불필요한 컬럼(0)을 추가생성한다.
x_sequence3 = to_categorical(x_sequence3)   # <class 'numpy.ndarray'>
print("Keras to_categorical:")
print(x_sequence3)
"""
[[0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
"""
print(x_sequence3.shape)      # (17, 13)
print(type(x_sequence3))      # <class 'numpy.ndarray'>

# 행 : 어절의 갯수 / 열 : 토큰 각각의라벨