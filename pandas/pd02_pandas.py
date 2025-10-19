import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]
                     ])

data = data.transpose()
data.columns = ['x1','x2','x3','x4']
print(data)

# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN



#0. 결측치 확인
print(data.isnull())
print(data.isna())
print(data.isna().sum())

print(data.info())
exit()
#1. 결측치 삭제 
# print(data.dropna()) # 디폴트가 행. (axis = 0)
# print(data.dropna(axis = 0)) # 디폴트가 행. (axis = 0)
print(data.dropna(axis = 1)) # 디폴트가 행. (axis = 0)

# 행 
#      0    1    2    3
# 3   8.0  8.0  8.0  8.0

# 열 
#      2
# 0   2.0
# 1   4.0
# 2   6.0
# 3   8.0
# 4  10.0

# 2-1. 특정값 - 평균 
mean = data.mean()
print(mean)


# 0    6.500000
# 1    4.666667
# 2    6.000000
# 3    6.000000
# dtype: float64

data2 = data.fillna(mean)
print(data2)


#       0      1        2    3
# 0   2.0  2.000000   2.0  6.0
# 1   6.5  4.000000   4.0  4.0
# 2   6.0  4.666667   6.0  6.0
# 3   8.0  8.000000   8.0  8.0
# 4  10.0  4.666667  10.0  6.0


# 2-2. 특정값 - 중위값

med = data.median()
print(med)

data3 = data.fillna(med)
print(data3)



# 2-3.특정값 - 0
data4 = data.fillna(0)
print(data4)



data4_2 = data.fillna(777)
print(data4_2)



# 2-4.특정값 - ffill (통상 마지막 값), (시계열 )
data5 = data.ffill()
print(data5) # 가장 첫번째 행은 채울 값이 없어서 Nan


# 2-5. 특정값 - bfill(통상 첫번째), (시계열)

data6= data.bfill()
print(data6) # 가장 마지막 행은 채울 값이 없어서 Nan


##################특정 컬럼만####################
means = data['x1'].mean()
print(means)

# 6.5

med = data['x4'].median()
print(med)

# 6.0

data['x1'] = data['x1'].fillna(means)
data['x2'] = data['x2'].ffill()
data['x4'] = data['x4'].fillna(med)

print(data)
