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

    # 그냥 한거 r2 : 0.8116515837104072
    # ============ n_component가 1 일때 ================
    # (49502, 24) (12376, 24)
    # (49502, 70) (12376, 70)
    # FI_Drop + PCA : 0.8112475759534583
    # ============ n_component가 2 일때 ================
    # (49502, 24) (12376, 24)
    # (49502, 71) (12376, 71)
    # FI_Drop + PCA : 0.8071266968325792
    # ============ n_component가 3 일때 ================
    # (49502, 24) (12376, 24)
    # (49502, 72) (12376, 72)
    # FI_Drop + PCA : 0.8124595992243051
    # ============ n_component가 4 일때 ================
    # (49502, 24) (12376, 24)
    # (49502, 73) (12376, 73)
    # FI_Drop + PCA : 0.8126212023270847
    # ============ n_component가 5 일때 ================
    # (49502, 24) (12376, 24)
    # (49502, 74) (12376, 74)
    # FI_Drop + PCA : 0.8108435681965094
    # ============ n_component가 6 일때 ================
    # (49502, 24) (12376, 24)
    # (49502, 75) (12376, 75)
    # FI_Drop + PCA : 0.8110859728506787
    # ============ n_component가 7 일때 ================
    # (49502, 24) (12376, 24)
    # (49502, 76) (12376, 76)
    # FI_Drop + PCA : 0.8101971557853911
    # ============ n_component가 8 일때 ================
    # (49502, 24) (12376, 24)
    # (49502, 77) (12376, 77)
    # FI_Drop + PCA : 0.8110859728506787
    # ============ n_component가 9 일때 ================
    # (49502, 24) (12376, 24)
    # (49502, 78) (12376, 78)
    # FI_Drop + PCA : 0.8101971557853911
    # ============ n_component가 10 일때 ================
    # (49502, 24) (12376, 24)
    # (49502, 79) (12376, 79)
    # FI_Drop + PCA : 0.812136393018746
    # ============ n_component가 11 일때 ================
    # (49502, 24) (12376, 24)
    # (49502, 80) (12376, 80)
    # FI_Drop + PCA : 0.8115707821590175
    # ============ n_component가 12 일때 ================
    # (49502, 24) (12376, 24)
    # (49502, 81) (12376, 81)
    # FI_Drop + PCA : 0.8111667744020685
    # ============ n_component가 13 일때 ================
    # (49502, 24) (12376, 24)
    # (49502, 82) (12376, 82)
    # FI_Drop + PCA : 0.8108435681965094
    # ============ n_component가 14 일때 ================
    # (49502, 24) (12376, 24)
    # (49502, 83) (12376, 83)
    # FI_Drop + PCA : 0.8114899806076277
    # ============ n_component가 15 일때 ================
    # (49502, 24) (12376, 24)
    # (49502, 84) (12376, 84)
    # FI_Drop + PCA : 0.8141564318034906
    # ============ n_component가 16 일때 ================
    # (49502, 24) (12376, 24)
    # (49502, 85) (12376, 85)
    # FI_Drop + PCA : 0.8097931480284422
    # ============ n_component가 17 일때 ================
    # (49502, 24) (12376, 24)
    # (49502, 86) (12376, 86)
    # FI_Drop + PCA : 0.8116515837104072
    # ============ n_component가 18 일때 ================
    # (49502, 24) (12376, 24)
    # (49502, 87) (12376, 87)
    # FI_Drop + PCA : 0.8130252100840336
    # ============ n_component가 19 일때 ================
    # (49502, 24) (12376, 24)
    # (49502, 88) (12376, 88)
    # FI_Drop + PCA : 0.8108435681965094
    # ============ n_component가 20 일때 ================
    # (49502, 24) (12376, 24)
    # (49502, 89) (12376, 89)
    # FI_Drop + PCA : 0.8114899806076277
    # ============ n_component가 21 일때 ================
    # (49502, 24) (12376, 24)
    # (49502, 90) (12376, 90)
    # FI_Drop + PCA : 0.8112475759534583
    # ============ n_component가 22 일때 ================
    # (49502, 24) (12376, 24)
    # (49502, 91) (12376, 91)
    # FI_Drop + PCA : 0.813994828700711
    # ============ n_component가 23 일때 ================
    # (49502, 24) (12376, 24)
    # (49502, 92) (12376, 92)
    # FI_Drop + PCA : 0.8105203619909502