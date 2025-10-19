import pandas as pd
import numpy as np

data = [['삼성','1000','2000'],
         ['현대','1100','3000'],
         ['LG','2000','500'],
         ['아모레','3500','6000'],
         ['네이버','100','1500'],
         ]
index=['031','059','033','045','023']
columns =['종목명','시가','종가']
df = pd.DataFrame(data=data, index=index, columns=columns)

print(df)

#      종목명    시가    종가
# 031   삼성  1000  2000
# 059   현대  1100  3000
# 033   LG  2000   500
# 045  아모레  3500  6000
# 023  네이버   100  1500

print("===============아모레와 네이버 시가 ")

print(df.iloc[3][1],df.iloc[4][1])
print(df.loc['045']['시가'],df.loc['023']['시가'])