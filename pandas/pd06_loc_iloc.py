### <<40>>

import pandas as pd
print(pd.__version__)

data = [
    ['삼성', '1000', '2000'],
    ['현대', '1100', '3000'],
    ['LG', '2000', '500'],
    ['아모레', '3500', '6000'],
    ['네이버', '100', '1500'],
]

index = ['031', '050', '033', '045', '023']
columns = ['종목명', '시가', '종가']

df = pd.DataFrame(data=data, index=index, columns=columns)
print(df)
#      종목명    시가    종가
# 031   삼성  1000  2000
# 050   현대  1100  3000
# 033   LG  2000   500
# 045  아모레  3500  6000
# 023  네이버   100  1500

# print(df[0])        # key error
# print(df['031'])    # key error : 판다스는 행을 입력하면 에러
print(df['종목명'])     # ★★★"판다스 열행"★★★ : 요소를 하나만입력하면 열값을 반환하고 열,행순서로 입력해야 열과 행값 모두 반환한다. 넘파이는 반대(넘파이는 행열)
# 031     삼성
# 050     현대
# 033     LG
# 045    아모레
# 023    네이버

#### 아모레 시가 출력 ####
## 에러나는 방식 ##
# print(df[3,1])
# print(df['045', '종목명'])
# print(df[['045']['종목명']])
# print(df['045']['종목명'])
# print(df[3][1])

## 올바른 방식 ##
print(df['종목명']['045'])  # ★★★"판다스 열행"★★★, # 아모레

"""
[열,행기준인 판다스를 행,열기준으로 추출하기]

loc : 인덱스 기준으로 행,열 데이터 추출 (사용자가 명시한 인덱스. 여기선 031, 050, 033, ...)
iloc : 행번호를 기준으로 행,열 데이터 추출 = int loc (판다스가 부여하는 자동행번호(위치값)임(0, 1, 2, ...))
"""
print("============ 아모레(행) 뽑기 ===========")

print(df.iloc[3])
# 종목명     아모레
# 시가     3500
# 종가     6000
# print(df.iloc['045'])   # TypeError: Cannot index by location index with a non-integer keys

print(df.loc['045'])
# 종목명     아모레
# 시가     3500
# 종가     6000
# print(df.loc[3])          # KeyError: 3

print("============ 네이버(행) 뽑기 ===========")
print(df.iloc[4])
# print(df.iloc['023'])
# print(df.loc[4])
print(df.loc['023'])
# 종목명     네이버
# 시가      100
# 종가     1500
# Name: 023, dtype: object

print("============ 아모레 종가(행,열) 뽑기 ===========")
print(df.iloc[3][2])        # 6000
print(df.iloc[3]['종가'])   # 6000
print(df.iloc[3,2])         # 6000
# print(df.iloc[3, '종가'])   # 데이터 타입 달라서 error

print(df.loc['045'][2])         # 6000
print(df.loc['045']['종가'])    # 6000
print(df.loc['045', '종가'])    # 6000
# print(df.loc['045', 2])      # 데이터 타입 달라서 error

print(df.iloc[3].iloc[2])       # 6000
print(df.iloc[3].loc['종가'])   # 6000

print(df.loc['045'].loc['종가'])    # 6000
print(df.loc['045'].iloc[2])        # 6000