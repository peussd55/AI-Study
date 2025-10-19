### <<36>>

# 수치형데이터에 LDA 적용 : 수치형 데이터인 y를 범주화하여 분류형처럼 변형하고 LDA를 적용하는 코드

import numpy as np
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import random

seed = 999
random.seed(seed)
np.array(seed)

# 1. 데이터
datasets = load_diabetes()
x = datasets['data']
y = datasets.target
print(x.shape, y.shape)     # (442, 10) (442,)
# print(y)
y_original = y.copy()       # 회귀모델에 적용할때는 LDA를 적용하기위해 y를 범주화하긴하지만 정답지는 원래의 y를 써야한다.
y = np.rint(y).astype(int)    # 반올림 한 후 int형으로 변환 -> y 범주화
# print(y)
print(np.unique(y, return_counts=True))

# train-test data 분리
x_train, x_test, y_train, y_test, y_train_0, y_test_0 = train_test_split(
   x, y, y_original, test_size=0.2, 
   random_state=seed,
    # stratify=y,    # y가 범주화되긴했으나 y의 각 클래스마다 최소2개이상의 샘플이 존재해야 오류가 안난다.(훈련, 테스트데이터 각각 1개씩 적용하기위해서)
)

# 스케일링
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

################################# PCA ##################################
# pca = PCA(n_components=10)  # n_components 디폴트값 : 가능한 max값(10)
# x_train = pca.fit_transform(x_train)
# x_test - pca.transform(x_test)
# pca_EVR = pca.explained_variance_ratio_
# print(np.cumsum(pca_EVR))
# # [0.39159198 0.5455036  0.67158621 0.7694678  0.83505705 0.89527312
# #  0.95070422 0.99125971 0.99905512 1.        ]

################################# LDA #################################
# 본래 수치형 데이터인 y_train을 LDA적용하기위해서 범주화하고 여기서 적용한다.
lda = LinearDiscriminantAnalysis(n_components=10)   # LDA의 n_components는 최대 클래스(y)의 수-1개 까지만 적용가능. 컬럼갯수가 y클래스 수보다 작으면 컬럼갯수를 따라간다.
x_train = lda.fit_transform(x_train, y_train)       # train데이터 fit_trainsform할땐 y도 들어가야한다.
x_test = lda.transform(x_test)
lda_EVR = lda.explained_variance_ratio_
print(np.cumsum(lda_EVR))
# [0.26203828 0.38566865 0.49547118 0.59373071 0.68353735 0.76579111
#  0.83948868 0.90639955 0.9545517  1.        ]

# 2. 모델
# models = [
model = RandomForestRegressor(random_state=seed)

# 3. 훈련
# for i, model in enumerate(models):
print(f"Processing {model.__class__.__name__}...")
model.fit(x_train, y_train_0)
results = model.score(x_test, y_test_0)
#  print(x.shape)
print(x_train.shape, '의 score :', results)

# 원래
# (353, 10) 의 score : 0.38559383155810856  0.40811600614249166 0.4915102089348713

# PCA 적용  -> 성능 제일 처참
# (353, 10) 의 score : -0.6023247919271411

# LDA 적용
# (353, 10) 의 score : 0.35581293264311853  0.4427609129709369  0.5524264313839826


    
    
    