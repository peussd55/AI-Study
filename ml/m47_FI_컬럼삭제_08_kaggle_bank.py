### <<37>>

# feature_importances : 트리기반모델에서만 제공

from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
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
path = './_data/kaggle/bank/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

# 인코딩
le_geo = LabelEncoder()
le_gen = LabelEncoder()
train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])
train_csv['Gender'] = le_gen.fit_transform(train_csv['Gender'])
test_csv['Geography'] = le_geo.transform(test_csv['Geography'])
test_csv['Gender'] = le_gen.transform(test_csv['Gender'])

# 불필요 컬럼 제거
train_csv = train_csv.drop(["CustomerId", "Surname"], axis=1)
test_csv = test_csv.drop(["CustomerId", "Surname"], axis=1)

x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8, 
    stratify=y,   # 분류데이터에서만 사용가능
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
model = XGBClassifier(random_state=seed)

model.fit(x_train, y_train)
print("=========", model.__class__.__name__, "========")
print('acc :', model.score(x_test, y_test))     # acc : 0.8635743933105099
print(model.feature_importances_)
    
# [0.01000721 0.05612099 0.05218606 0.12075434 0.00962212 0.02436673
#  0.51608354 0.01322166 0.18730022 0.01033714]
print("25%지점 :",np.percentile(model.feature_importances_, 25))   # 25% 지점확인
# 25%지점 : 0.011058272095397115

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
print(col_name)     # ['CreditScore', 'Tenure', 'EstimatedSalary']

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
print('acc :', model.score(x_test, y_test))     # acc :  0.8627866816129912