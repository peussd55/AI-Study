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

print("=====================================")
print("시가가 1000원 이상인 행을 출력")
print(df['시가'].astype(int) >= 1000)   # astype(int)을 안해도 아스키코드순으로 비교해서 출력해준다.
# 031     True
# 050     True
# 033     True
# 045     True
# 023    False

print(df[ df['시가'].astype(int) >= 1000 ])             # T/F 로 뽑았으면 굳이 loc 안써도 된다.
print(df.loc[ df['시가'].astype(int) >= 1000 ])
#      종목명    시가    종가
# 031   삼성  1000  2000
# 050   현대  1100  3000
# 033   LG  2000   500
# 045  아모레  3500  6000

df3 = df[df['시가'].astype(int) >= 1000]['종가']        # T/F 로 뽑았으면 굳이 loc 안써도 된다.
df3 = df.loc[df['시가'].astype(int) >= 1000]['종가']
print(df3)