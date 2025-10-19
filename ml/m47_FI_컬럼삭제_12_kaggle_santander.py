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

seed = 123
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
path = './_data/kaggle/santander/'           
# 맨 앞, 공통 컬럼 datetime을 인덱스 컬럼으로 지정
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')
print(train_csv.shape, test_csv.shape, submission_csv.shape)  # (200000, 202) (200000, 201) (200000, 2)
print(train_csv['target'].value_counts())
# 이진분류 (불균형)
# 0    179902
# 1     20098
# train data x로 y 분리
x = train_csv.drop(['target'], axis=1)
print(x)
y = train_csv['target']


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
print('acc :', model.score(x_test, y_test))     # acc : 0.91405
print(model.feature_importances_)
    
print("25%지점 :",np.percentile(model.feature_importances_, 25))   # 25% 지점확인
# 25%지점 : 0.0029357103630900383

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
# ['var_3', 'var_7', 'var_10', 'var_14', 'var_16', 'var_17', 'var_19', 'var_25', 'var_27', 'var_29', 'var_30', 'var_38', 'var_39', 'var_42', 'var_46', 'var_47', 'var_60', 'var_61', 'var_62', 'var_64', 'var_65', 'var_69', 'var_72', 'var_73', 'var_79', 'var_84', 'var_96', 'var_97', 'var_98', 'var_100', 'var_101', 'var_103', 'var_113', 'var_117', 'var_120', 'var_124', 'var_126', 'var_129', 'var_136', 'var_142', 'var_143', 'var_153', 'var_158', 'var_159', 'var_160', 'var_161', 'var_176', 'var_181', 'var_185', 'var_189']


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
print('acc2 :', model.score(x_test, y_test))     # acc2 : 0.91385