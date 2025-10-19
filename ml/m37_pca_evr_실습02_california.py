### <<32>>

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
import warnings     # 경고무시
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.decomposition import PCA

# 1. 데이터
x = datasets = fetch_california_housing()
x = datasets.data
y = datasets['target']
print(x.shape)  # (20640, 8)
print(y.shape)  # (20640,)

# 주성분분석 임계값에 따른 필요한 n_components 구하기
pca = PCA(n_components=8)   # PCA에서 최대 주성분 개수는 샘플 수와 특성 수 중 작은 값으로 제한됩니다. 즉, x.shape가 (100,200)이면 n_components는 최대 100
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr) # 누적합
print(evr_cumsum)
# [0.99978933 0.99990261 0.99998589 0.99999233 0.99999746 0.99999978
#  0.99999998 1.        ]
print(len(evr_cumsum))  # 8

# 부동소수점 오차와  argmax 조합의 위험성 : argmax는 조건을 만족하는 원소가 없을때도 0을 반환한다. 즉, 0번째 원소부터 만족하는 결과랑 같게나와서 혼동을 줄 수있다.
# -> 왜 만족을 못하는지? np.argmax(evr_cumsum >= 1.0)에서의 1.0과 [0.99978933 0.99990261 0.99998589 0.99999233 0.99999746 0.99999978 0.99999998 1.        ]의 1.을 동일하게 취급하지못하고
# 마지막 인덱스를 1.0보다 낮게 취급해서(0.999999xxxx로 인식) 만족하는게 없어서 0을 반환하는 문제발생.
# 다시예를들면 누적합 리스트가[0.1, 0.2, 0.3] 인데 np.argmax(list >= 0.5)를 하면 만족하는 원소가 없음에도 0을 반환하는 경우와 같은 문제이다.
# 해결방법) evr_cumsum가 1.0을 인식하지못하는 경우 1.0에 오차허용 tolerance 을 빼줘서 1. 을 조건에 포함시킬수 있도록한다.
# => 이 문제는 x_pca = x.copy()처럼 값을 복사해올때 발생한다. copy해올때 값이 바뀌는 것이 아니라 pca객체가 연산할때 미묘한 차이로인해 발상한다.

tolerance = 1e-10
n_comp_100 = np.argmax(evr_cumsum >= 1.0)
print(n_comp_100)   # 7

n_comp_999 = np.argmax(evr_cumsum >= 0.999)
print(n_comp_999)   # 0

n_comp_99 = np.argmax(evr_cumsum >= 0.99)
print(n_comp_99)   # 0

n_comp_95 = np.argmax(evr_cumsum >= 0.95)
print(n_comp_95)   # 0

n_comp_list = [n_comp_100+1, n_comp_999+1, n_comp_99+1, n_comp_95+1]
print(n_comp_list)  # [8, 1, 1, 1]

# n_components를 구했으면 x를 원래대로 되돌려준다.
x = datasets.data

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8, 
    # stratify=y,   # 분류데이터에서만 사용가능
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333, )

# 2. 모델
model = HistGradientBoostingRegressor()

# 3. 훈련
for i in n_comp_list:
    scores = cross_val_score(model, x_train, y_train, cv=kfold)
    print("================================================================================")
    print(i, '일때의 r2_score :', scores, '\n평균 r2_score :', round(np.mean(scores), 4))
    # [kfold_train_test_split]
    # r2_score : [0.83978557 0.83866319 0.83032923 0.83098388 0.82708287] 
    # 평균 r2_score : 0.8334

    # [kfold]
    # r2_score : [0.82844582 0.84063748 0.82240123 0.84105292 0.84572463] 
    # 평균 r2_score : 0.8357

    y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)

    r2 = r2_score(y_test, y_pred)
    print(i, '일때의 cross_val_predict r2_score :', round(r2, 4))     # 0.8061
    
    # ================================================================================
    # 8 일때의 r2_score : [0.84074984 0.83658908 0.8284458  0.83028639 0.82813789]
    # 평균 r2_score : 0.8328
    # 8 일때의 cross_val_predict r2_score : 0.8061
    # ================================================================================
    # 1 일때의 r2_score : [0.84011271 0.83590014 0.83198616 0.82750995 0.82844141]
    # 평균 r2_score : 0.8328
    # 1 일때의 cross_val_predict r2_score : 0.8061
    # ================================================================================
    # 1 일때의 r2_score : [0.83957538 0.8385313  0.82768376 0.82953131 0.82653445]
    # 평균 r2_score : 0.8324
    # 1 일때의 cross_val_predict r2_score : 0.8061
    # ================================================================================
    # 1 일때의 r2_score : [0.83880226 0.83707324 0.82807659 0.82681578 0.82314014]
    # 평균 r2_score : 0.8308
    # 1 일때의 cross_val_predict r2_score : 0.8061