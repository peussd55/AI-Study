### <<37>>

# feature_importances : 트리기반모델에서만 제공

from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
import random
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

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

# 2. 모델
model = XGBClassifier(random_state=seed)

model.fit(x_train, y_train)
print("=========", model.__class__.__name__, "========")
print('r2 :', model.score(x_test, y_test))
r2 = model.score(x_test, y_test)
print(model.feature_importances_)
    
print("25%지점 :",np.percentile(model.feature_importances_, 25))   # 25% 지점확인

percentile = np.percentile(model.feature_importances_, 25)
print(type(percentile))     # <class 'numpy.float64'>

col_name = []
# 삭제할 컬럼(25%이하인놈)을 찾아내기
for i, fi in enumerate(model.feature_importances_):
    # print(i, fi)
    if fi <= percentile:
        # col_name.append(datasets.feature_names[i])
        col_name.append(x.columns[i])
    else:
        continue
print(col_name)     # 

# dataframe 생성한 후 삭제할 컬럼 drop하기
# x_f = pd.DataFrame(x, columns=datasets.feature_names)
x_f = x
x1 = x_f.drop(columns=col_name)
x2 = x_f[col_name]
com_len = len(col_name) # 삭제할 컬럼 갯수

x1_train, x1_test, x2_train, x2_test = train_test_split(
    x1, x2, train_size=0.8, random_state=seed,
    stratify=y,
)
print(x1_train.shape, x1_test.shape)    # 
print(x2_train.shape, x2_test.shape)    # 
print(y_train.shape, y_test.shape)      # 

print('그냥 한거 r2 :', r2)

# PCA로 삭제된 컬럼합치기
for i in range(1, com_len):
    print(f'============ n_component가 {i} 일때 ================')
    pca = PCA(n_components=i)
    x2_train_pca  = pca.fit_transform(x2_train)
    x2_test_pca = pca.transform(x2_test)
    print(x2_train.shape, x2_test.shape)

    x_train = np.concatenate([x1_train, x2_train_pca], axis=1)
    x_test = np.concatenate([x1_test, x2_test_pca], axis=1)
    print(x_train.shape, x_test.shape)

    model.fit(x_train, y_train)
    
    print('FI_Drop + PCA :', model.score(x_test, y_test))       # (날린거 PCA로 축소하고 합쳤을 때)
    
    # 그냥 한거 r2 : 0.91405
    # ============ n_component가 1 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 151) (40000, 151)
    # FI_Drop + PCA : 0.91365
    # ============ n_component가 2 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 152) (40000, 152)
    # FI_Drop + PCA : 0.913525
    # ============ n_component가 3 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 153) (40000, 153)
    # FI_Drop + PCA : 0.91285
    # ============ n_component가 4 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 154) (40000, 154)
    # FI_Drop + PCA : 0.91405
    # ============ n_component가 5 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 155) (40000, 155)
    # FI_Drop + PCA : 0.91455
    # ============ n_component가 6 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 156) (40000, 156)
    # FI_Drop + PCA : 0.914075
    # ============ n_component가 7 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 157) (40000, 157)
    # FI_Drop + PCA : 0.913475
    # ============ n_component가 8 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 158) (40000, 158)
    # FI_Drop + PCA : 0.913425
    # ============ n_component가 9 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 159) (40000, 159)
    # FI_Drop + PCA : 0.914
    # ============ n_component가 10 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 160) (40000, 160)
    # FI_Drop + PCA : 0.91495
    # ============ n_component가 11 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 161) (40000, 161)
    # FI_Drop + PCA : 0.91375
    # ============ n_component가 12 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 162) (40000, 162)
    # FI_Drop + PCA : 0.91435
    # ============ n_component가 13 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 163) (40000, 163)
    # FI_Drop + PCA : 0.914675
    # ============ n_component가 14 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 164) (40000, 164)
    # FI_Drop + PCA : 0.914075
    # ============ n_component가 15 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 165) (40000, 165)
    # FI_Drop + PCA : 0.912825
    # ============ n_component가 16 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 166) (40000, 166)
    # FI_Drop + PCA : 0.914
    # ============ n_component가 17 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 167) (40000, 167)
    # FI_Drop + PCA : 0.91395
    # ============ n_component가 18 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 168) (40000, 168)
    # FI_Drop + PCA : 0.914125
    # ============ n_component가 19 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 169) (40000, 169)
    # FI_Drop + PCA : 0.9132
    # ============ n_component가 20 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 170) (40000, 170)
    # FI_Drop + PCA : 0.913325
    # ============ n_component가 21 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 171) (40000, 171)
    # FI_Drop + PCA : 0.9141
    # ============ n_component가 22 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 172) (40000, 172)
    # FI_Drop + PCA : 0.9143
    # ============ n_component가 23 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 173) (40000, 173)
    # FI_Drop + PCA : 0.91345
    # ============ n_component가 24 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 174) (40000, 174)
    # FI_Drop + PCA : 0.91405
    # ============ n_component가 25 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 175) (40000, 175)
    # FI_Drop + PCA : 0.914075
    # ============ n_component가 26 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 176) (40000, 176)
    # FI_Drop + PCA : 0.913375
    # ============ n_component가 27 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 177) (40000, 177)
    # FI_Drop + PCA : 0.914075
    # ============ n_component가 28 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 178) (40000, 178)
    # FI_Drop + PCA : 0.913675
    # ============ n_component가 29 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 179) (40000, 179)
    # FI_Drop + PCA : 0.913275
    # ============ n_component가 30 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 180) (40000, 180)
    # FI_Drop + PCA : 0.913925
    # ============ n_component가 31 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 181) (40000, 181)
    # FI_Drop + PCA : 0.91345
    # ============ n_component가 32 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 182) (40000, 182)
    # FI_Drop + PCA : 0.9137
    # ============ n_component가 33 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 183) (40000, 183)
    # FI_Drop + PCA : 0.91275
    # ============ n_component가 34 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 184) (40000, 184)
    # FI_Drop + PCA : 0.91505
    # ============ n_component가 35 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 185) (40000, 185)
    # FI_Drop + PCA : 0.91425
    # ============ n_component가 36 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 186) (40000, 186)
    # FI_Drop + PCA : 0.911875
    # ============ n_component가 37 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 187) (40000, 187)
    # FI_Drop + PCA : 0.91335
    # ============ n_component가 38 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 188) (40000, 188)
    # FI_Drop + PCA : 0.914225
    # ============ n_component가 39 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 189) (40000, 189)
    # FI_Drop + PCA : 0.91385
    # ============ n_component가 40 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 190) (40000, 190)
    # FI_Drop + PCA : 0.9133
    # ============ n_component가 41 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 191) (40000, 191)
    # FI_Drop + PCA : 0.9139
    # ============ n_component가 42 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 192) (40000, 192)
    # FI_Drop + PCA : 0.914425
    # ============ n_component가 43 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 193) (40000, 193)
    # FI_Drop + PCA : 0.91405
    # ============ n_component가 44 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 194) (40000, 194)
    # FI_Drop + PCA : 0.914325
    # ============ n_component가 45 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 195) (40000, 195)
    # FI_Drop + PCA : 0.912975
    # ============ n_component가 46 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 196) (40000, 196)
    # FI_Drop + PCA : 0.912425
    # ============ n_component가 47 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 197) (40000, 197)
    # FI_Drop + PCA : 0.913075
    # ============ n_component가 48 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 198) (40000, 198)
    # FI_Drop + PCA : 0.91285
    # ============ n_component가 49 일때 ================
    # (160000, 50) (40000, 50)
    # (160000, 199) (40000, 199)
    # FI_Drop + PCA : 0.913325