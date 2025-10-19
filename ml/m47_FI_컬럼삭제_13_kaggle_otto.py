### <<37>>

# feature_importances : 트리기반모델에서만 제공

from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder

seed = 123
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
path = './_data/kaggle/otto/'
# 맨 앞, 공통 컬럼 datetime을 인덱스 컬럼으로 지정
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')
# # target컬럼 레이블 인코딩(원핫 인코딩 사전작업)###
# # 정수형을 직접 원핫인코딩할경우 keras, pandas, sklearn 방식 모두 가능하지만 문자형태로 되어있을 경우에는 pandas방식만 문자열에서 직접 원핫인코딩이 가능하다.
# le = LabelEncoder() # 인스턴스화
# train_csv['target'] = le.fit_transform(train_csv['target'])   # fit 함수 + transform 함친 합친 함수 : 변환해서 적용
# train data x로 y 분리
x = train_csv.drop(['target'], axis=1)
print(x)
print('x type:',type(x))
y = train_csv['target']
print('y type:',type(y))
print(y)
le = LabelEncoder()     # xgboost 쓰려면 y 정수형으로 라벨링(0~8)
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    stratify=y,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
model = XGBClassifier(random_state=seed)

model.fit(x_train, y_train)
print("=========", model.__class__.__name__, "========")
print('acc :', model.score(x_test, y_test))     # acc : 0.8116515837104072
print(model.feature_importances_)
    
print("25%지점 :",np.percentile(model.feature_importances_, 25))   # 25% 지점확인
# 25%지점 : 0.0035112767945975065

percentile = np.percentile(model.feature_importances_, 25)
print(type(percentile))     # <class 'numpy.float64'>

col_name = []
# 삭제할 컬럼(25%이하인놈)을 찾아내기
for i, fi in enumerate(model.feature_importances_):
    # print(i, fi)
    if fi <= percentile:
        col_name.append(x.columns[i])
    else:
        continue
print(col_name)
# ['feat_5', 'feat_6', 'feat_10', 'feat_12', 'feat_18', 'feat_21', 'feat_22', 'feat_28', 'feat_31', 'feat_44', 'feat_46', 'feat_49', 'feat_54', 'feat_61', 'feat_63', 'feat_65', 'feat_66', 'feat_70', 'feat_73', 'feat_74', 'feat_81', 'feat_82', 'feat_87', 'feat_89']

# dataframe 생성한 후 삭제할 컬럼 drop하기
x = x.drop(columns=col_name)

print(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    stratify=y,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model.fit(x_train, y_train)
print('acc2 :', model.score(x_test, y_test))     # acc2 : 0.8106819650937298