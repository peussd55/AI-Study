### <<28>>

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

text1 = '오늘도 못생기고 영어를 디게 디게 디게 못 하는 일삭이는 재미없는 개그를 \
        마구 마구 마구 마구 하면서 딴짓을 한다.'

text2 = '오늘도 박석사가 자아를 디게 디게 찾아냈다. 상진이는 마구 마구 딴짓을 한다. \
        재현은 무생기고 재미없는 딴짓을 한다.'
        
token = Tokenizer()
token.fit_on_texts([text1, text2])

print(token.word_index)     # 인덱스 1부터시작함(0부터 시작X)
# {'마구': 1, '디게': 2, '딴짓을': 3, '한다': 4, '오늘도': 5, '재미없는': 6, 
# '못생기고': 7, '영어를': 8, '못': 9, '하는': 10, '일삭이는': 11, '개그를': 12, 
# '하면서': 13, '박석사가': 14, '자아를': 15, '찾아냈다': 16, '상진
# 이는': 17, '재현은': 18, '무생기고': 19}

print(token.word_counts)
# OrderedDict([('오늘도', 2), ('못생기고', 1), ('영어를', 1), ('디게', 5), ('못', 1), 
# ('하는', 1), ('일삭이는', 1), ('재미없는', 2), ('개그를', 1), ('마구', 6), ('하면서', 1), 
# ('딴짓을', 3), ('한다', 3), ('박석사가', 1), ('자아를', 1), ('찾아냈다', 1), ('상진이는', 1), 
# ('재현은', 1), ('무생기고', 1)])

x = token.texts_to_sequences([text1, text2])
print(x)        # <class 'list'>
# [[5, 7, 8, 2, 2, 2, 9, 10, 11, 6, 12, 1, 1, 1, 1, 13, 3, 4], [5, 14, 15, 2, 2, 16, 17, 1, 1, 3, 4, 18, 19, 6, 3, 4]]
x = np.array(x) # <class 'numpy.ndarray'>
print(x)
# [list([5, 7, 8, 2, 2, 2, 9, 10, 11, 6, 12, 1, 1, 1, 1, 13, 3, 4])
#  list([5, 14, 15, 2, 2, 16, 17, 1, 1, 3, 4, 18, 19, 6, 3, 4])]

x = np.concatenate(x)
print(x)
print(len(x))   # 34
print(type(x))  # <class 'numpy.ndarray'>
# [ 5  7  8  2  2  2  9 10 11  6 12  1  1  1  1 13  3  4  5 14 15  2  2 16
#  17  1  1  3  4 18 19  6  3  4]

########### 원핫인코딩 3가지 만들기 ###########
print(len(x))   # 34, nparray
# 1. 판다스
import pandas as pd
x_sequence1 = x
x_sequence1 = pd.Series(x_sequence1)   # <class 'pandas.core.series.Series'>
x_sequence1 = pd.get_dummies(x_sequence1)
print("Pandas get_dummies:")
print(x_sequence1)
print(x_sequence1.shape)      # (34, 19)
print(type(x_sequence1))      # <class 'pandas.core.frame.DataFrame'>

# 2. sklearn
from sklearn.preprocessing import OneHotEncoder
x_sequence2 = x
x_sequence2 = np.array(x_sequence2)         # <class 'numpy.ndarray'>
x_sequence2 = x_sequence2.reshape(-1, 1)
ohe = OneHotEncoder(sparse_output=False)    # sparse_output : sklearn 1.3이상 문법. ohe가 인자로 받은 값을 nparray로 반환하게하기
x_sequence2 = ohe.fit_transform(x_sequence2)
print("Sklearn OneHotEncoder:")
print(x_sequence2)
print(x_sequence2.shape)      # (34, 19)
print(type(x_sequence2))      # <class 'numpy.ndarray'>

# 3. keras
from tensorflow.keras.utils import to_categorical
x_sequence3 = x
# x_sequence3 = x_sequence3 - np.min(x_sequence3)
x_sequence3 = to_categorical(x_sequence3)   # <class 'numpy.ndarray'>
print("Keras to_categorical:")
print(x_sequence3)
print(x_sequence3.shape)            # (34, 20)
x_sequence3 = x_sequence3[:, 1:]    # 불필요한 열 제거(패딩된 열)
print(x_sequence3)                  # (34, 19)
print(x_sequence3.shape)
print(type(x_sequence3))      # <class 'numpy.ndarray'>

# 행 : 어절의 갯수 / 열 : 토큰 각각의라벨
# https://wikidocs.net/33520 참고
# 이 방식(원핫인코딩)으로만 하는 자연어처리의 문제점 : 토큰이 많아지면 많아질수록(차원이 길어질수록) 연산 공간낭비가 심하다.
# 희소 표현(sparse representation) : 값이 대부분 0으로 표현
# 희소 벡터

# -> 임베딩방식으로 개선 : 토큰 각각의 좌표값을 생성. 인코딩값보다 좌표값이 훨씬 짧음. 임베딩사전구축으로 토큰간 관계에 따른 위치에 좌표값 구축필요
# 밀집 벡터

