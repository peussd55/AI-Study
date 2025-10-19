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

# print(df)

#print(df[0]) #KeyError(key) from err
# print(df['031'])
# print(df['종목명'])     #판다스 열 행 / 컬럼이 기준

#############아모레 출력 
# print(df[3,1]) # 에러
print(df['종목명']['045'])

# loc:인덱스 기준으로 행 데이터 추출
# iloc: 행번호를 기준으로 행 데이터 추출 int location

print("-------------아모레 뽑기-----------")

print(df.iloc[4])
print(df.loc['023'])

print("-------------아모레 종가 뽑기-----------")

print(df.iloc[3][2])    #6000
print(df.iloc[3]['종가']) #6000
print(df.iloc[3,2]) #6000

print(df.loc['045'][2]) #6000
print(df.loc['045']['종가'])    #6000
print(df.loc['045','종가'])  #6000

print(df.iloc[3].iloc[2])
print(df.iloc[3].loc['종가'])
print(df.loc['045'].loc['종가'])