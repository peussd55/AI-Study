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
datasets = load_digits()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # ((1797, 64) (1797,)


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
print('acc :', model.score(x_test, y_test))     # acc : 0.975
print(model.feature_importances_)
    
# [0.         0.06415373 0.01059751 0.00873501 0.00434788 0.03380749
#  0.00921624 0.01642124 0.         0.01852774 0.01413111 0.01034479
#  0.00841835 0.01131467 0.00558309 0.00333757 0.         0.00507369
#  0.00538672 0.04030624 0.01079499 0.04513224 0.00413447 0.        
#  0.         0.00495149 0.02641704 0.00841728 0.02751666 0.01865148
#  0.01018624 0.         0.         0.07354813 0.00440697 0.00674871
#  0.04887262 0.01661007 0.02459871 0.         0.         0.00697014
#  0.03422127 0.04050089 0.0116045  0.01881086 0.02428929 0.        
#  0.         0.00862246 0.005208   0.00565973 0.01122131 0.0095808 
#  0.02874239 0.         0.         0.04148491 0.00863673 0.00476112
#  0.06507792 0.01276453 0.04295827 0.01819468]
print("25%지점 :",np.percentile(model.feature_importances_, 25))   # 25% 지점확인
# 25%지점 : 0.004392194678075612

percentile = np.percentile(model.feature_importances_, 25)
print(type(percentile))     # <class 'numpy.float64'>

col_name = []
# 삭제할 컬럼(25%이하인놈)을 찾아내기
for i, fi in enumerate(model.feature_importances_):
    # print(i, fi)
    if fi <= percentile:
        col_name.append(datasets.feature_names[i])
    else:
        continue
print(col_name)
# ['pixel_0_0', 'pixel_0_4', 'pixel_1_0', 'pixel_1_7', 'pixel_2_0', 'pixel_2_6', 'pixel_2_7', 'pixel_3_0', 'pixel_3_7', 'pixel_4_0', 'pixel_4_7', 'pixel_5_0', 'pixel_5_7', 'pixel_6_0', 'pixel_6_7', 'pixel_7_0']

# dataframe 생성한 후 삭제할 컬럼 drop하기
x = pd.DataFrame(x, columns=datasets.feature_names)
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
print('acc2 :', model.score(x_test, y_test))     # acc2 : 0.9722222222222222