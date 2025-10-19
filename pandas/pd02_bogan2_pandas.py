### <<40>>

import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan],    
                    ])
data = data.transpose()
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)
#       0    1     2    3
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

# 0. 결측치 확인
print(data.isnull())        # Nan인걸 True로 반환
#        0      1      2      3
# 0  False  False  False   True
# 1   True  False  False  False
# 2  False   True  False   True
# 3  False  False  False  False
# 4  False   True  False   True
print("\n")
print(data.isnull().sum())  # 열 기준으로 NaN의 갯수반환
# 0    1
# 1    2
# 2    0
# 3    3
print("\n")
print(data.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 5 entries, 0 to 4
# Data columns (total 4 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   0       4 non-null      float64
#  1   1       3 non-null      float64
#  2   2       5 non-null      float64
#  3   3       2 non-null      float64
# dtypes: float64(4)
# memory usage: 288.0 bytes
# None

# 1. 결측치 삭제
print(data.dropna())    # 디폴트가 행이다. axis=0
#      0    1    2    3
# 3  8.0  8.0  8.0  8.0
print(data.dropna(axis=1))
#       2
# 0   2.0
# 1   4.0
# 2   6.0
# 3   8.0
# 4  10.0

# 2-1. 특정값 - 평균
means = data.mean()     # 열 기준으로 
print(means)
# 0    6.500000
# 1    4.666667
# 2    6.000000
# 3    6.000000
# dtype: float64
data2 = data.fillna(means)
print(data2)
#       0         1     2    3
# 0   2.0  2.000000   2.0  6.0
# 1   6.5  4.000000   4.0  4.0
# 2   6.0  4.666667   6.0  6.0
# 3   8.0  8.000000   8.0  8.0
# 4  10.0  4.666667  10.0  6.0

# 2-2. 특정값 - 중위값
med = data.median()
print(med)
# 0    7.0
# 1    4.0
# 2    6.0
# 3    6.0
# dtype: float64
data3 = data.fillna(med)
print(data3)
#       0    1     2    3
# 0   2.0  2.0   2.0  6.0
# 1   7.0  4.0   4.0  4.0
# 2   6.0  4.0   6.0  6.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  4.0  10.0  6.0

# 2-3. 특정값 - 0
data4 = data.fillna(0)
print(data4)
#       0    1     2    3
# 0   2.0  2.0   2.0  0.0
# 1   0.0  4.0   4.0  4.0
# 2   6.0  0.0   6.0  0.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  0.0  10.0  0.0

data4_2 = data.fillna(777)
print(data4_2)
#        0      1     2      3
# 0    2.0    2.0   2.0  777.0
# 1  777.0    4.0   4.0    4.0
# 2    6.0  777.0   6.0  777.0
# 3    8.0    8.0   8.0    8.0
# 4   10.0  777.0  10.0  777.0

# 2-4. 특정값 - ffill (통상 마지막값, 시계열에서 주로 씀)
data5 = data.ffill()    # 가장 첫번째 행은 채울값이 없어서  Nan
print(data5)    
#       0    1     2    3
# 0   2.0  2.0   2.0  NaN
# 1   2.0  4.0   4.0  4.0
# 2   6.0  4.0   6.0  4.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  8.0  10.0  8.0

# 2-5. 특정ㄱ값 - bfill (통상 첫번째, 시계열에서 주로 씀)
data6 = data.bfill()    # 가장  마지막 행은 채울값이 없어서  Nan
print(data6)
#       0    1     2    3
# 0   2.0  2.0   2.0  4.0
# 1   6.0  4.0   4.0  4.0
# 2   6.0  8.0   6.0  8.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

######################## 특정 컬럼만 ########################
means = data['x1'].mean()
print(means)    # 6.5

med = data['x4'].median()
print(med)      # 6.0

data['x1'] = data['x1'].fillna(means)
data['x2'] = data['x2'].ffill()
data['x4'] = data['x4'].fillna(med)

print(data)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  6.0
# 1   6.5  4.0   4.0  4.0
# 2   6.0  4.0   6.0  6.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  8.0  10.0  6.0