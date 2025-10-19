### <<37>>

# feature_importances : 트리기반모델에서만 제공
# 44카피

from sklearn.datasets import fetch_covtype
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
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (581012, 54) (581012,)
y = y-1

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    stratify=y,
)

# 2. 모델
model1 = DecisionTreeClassifier(random_state=seed)
model2 = RandomForestClassifier(random_state=seed)
model3 = GradientBoostingClassifier(random_state=seed)
model4 = XGBClassifier(random_state=seed)

models = [model1, model2, model3, model4]

for model in models:
    model.fit(x_train, y_train)
    print("=========", model.__class__.__name__, "========")
    print('acc :', model.score(x_test, y_test))
    print(model.feature_importances_)   # 부스팅모델(사이킷런기반)에서 중요도를 선정하는 기준 : 빈도수(Frequency)
"""

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