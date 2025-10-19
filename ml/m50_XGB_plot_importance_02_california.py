### <<37>>

# feature_importances : 트리기반모델에서만 제공
# 44카피

from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
import random
import numpy as np

seed = 123
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    # stratify=y,
)

# 2. 모델
model1 = DecisionTreeRegressor(random_state=seed)
model2 = RandomForestRegressor(random_state=seed)
model3 = GradientBoostingRegressor(random_state=seed)
model4 = XGBRegressor(random_state=seed)

models = [model1, model2, model3, model4]

for model in models:
    model.fit(x_train, y_train)
    print("=========", model.__class__.__name__, "========")
    print('r2 :', model.score(x_test, y_test))
    print(model.feature_importances_)   # 부스팅모델(사이킷런기반)에서 중요도를 선정하는 기준 : 빈도수(Frequency)
"""
========= DecisionTreeRegressor ========
r2 : 0.6000320873754088
[0.51933732 0.04854233 0.04799864 0.02724695 0.03268622 0.13136699
 0.0991418  0.09367973]
========= RandomForestRegressor ========
r2 : 0.8121690217687418
[0.52251872 0.05198119 0.04717624 0.02923746 0.03156002 0.13387452
 0.0920824  0.09156945]
========= GradientBoostingRegressor ========
r2 : 0.7978378408140232
[0.59864938 0.03019079 0.02141916 0.00492639 0.00427861 0.12193384
 0.10819286 0.11040897]
========= XGBRegressor ========
r2 : 0.83707103301617
[0.47826383 0.07366086 0.0509511  0.02446287 0.02366972 0.14824368
 0.0921493  0.10859864]
"""
    
import matplotlib.pyplot as plt

# def plot_feature_importance_datasets(model):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center')    # 수평 막대 그래프, 4개의 열의 feature_importance 그래프, 값 위치 센터
#     plt.yticks(np.arange(n_features), model.feature_importances_)  # 눈금, 숫자 레이블 표시
#     plt.xlabel("feature importance")
#     plt.ylabel("Feature")
#     plt.ylim(-1, n_features)    # 축 범위 설정
#     plt.title(model.__class__.__name__)

from xgboost.plotting import plot_importance
plot_importance(model)
# F Score : Frequency Score : Tree에서 맨 밑 값을 찾기위해 방문(split)한 총 횟수. 성능이 높으면 F Score가 높을 확률이있다(dnn에서 레이어의 깊이와 비슷)
"""
[importance_type]
 'weight' : default. 얼마나 자주 split했나. 통상 빈도수(Frequency)
 'gain' : split이 모델의 성능을 얼마나 개선했나 // 통상적으로 많이 씀
 'cover' : split하기 위한 sample 수 // 별로 안 씀
"""
    
# plot_feature_importance_datasets(model)
plt.show()