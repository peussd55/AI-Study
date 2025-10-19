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

print("================ 아모레와 네이버의 시가 ============")
print(df.loc['045', '시가'], df.loc['023', '시가'])     # 3500 100
print(df.iloc[3, 1], df.iloc[4, 1])                    # 3500 100

# 원하는 종목
targets = ['네이버', '아모레']

# df를 **한 번만** 호출해 원하는 두 종목의 ‘시가’ 열 추출
open_prices = df.loc[df['종목명'].isin(targets), '시가']

print(open_prices)