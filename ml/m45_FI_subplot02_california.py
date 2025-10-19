### <<36>>

# feature_importances : 트리기반모델에서만 제공

from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
import random
import numpy as np
import matplotlib.pyplot as plt

seed = 123
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (150, 4) (150,)

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

def plot_feature_importance_datasets(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')    # 수평 막대 그래프, 4개의 열의 feature_importance 그래프, 값 위치 센터
    plt.yticks(np.arange(n_features), model.feature_importances_)  # 눈금, 숫자 레이블 표시
    plt.xlabel("feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)    # 축 범위 설정
    plt.title(model.__class__.__name__)
    
# 기존 plot_feature_importance_datasets 함수를 subplot용으로 수정
def plot_feature_importance_datasets(model, ax):
    n_features = datasets.data.shape[1]
    ax.barh(np.arange(n_features), model.feature_importances_, align='center')
    ax.set_yticks(np.arange(n_features))
    ax.set_yticklabels(datasets.feature_names)  # 실제 feature 이름 표시
    ax.set_xlabel("Feature Importance")
    ax.set_ylabel("Feature")
    ax.set_ylim(-1, n_features)
    ax.set_title(model.__class__.__name__)

# subplot으로 4개의 모델 feature importance 시각화
fig, axes = plt.subplots(2, 2, figsize=(15, 10))  # 2행 2열 subplot 생성
axes = axes.flatten()  # 1차원 배열로 변환


for i, model in enumerate(models):
    model.fit(x_train, y_train)
    print("=========", model.__class__.__name__, "========")
    print('acc :', model.score(x_test, y_test))
    print(model.feature_importances_)
    
    # ========= DecisionTreeRegressor ========
    # acc : 0.6043630467808063
    # [0.51931246 0.04981492 0.04834159 0.02632489 0.03346774 0.13169898
    # 0.09904432 0.09199511]
    # ========= RandomForestRegressor ========
    # acc : 0.8114334347018757
    # [0.52255998 0.05206861 0.04703019 0.0292777  0.03162071 0.13398205
    # 0.09197568 0.09148508]
    # ========= GradientBoostingRegressor ========
    # acc : 0.7978378408140232
    # [0.59864938 0.03019079 0.02141916 0.00492639 0.00427861 0.12193384
    # 0.10819286 0.11040897]
    # ========= XGBRegressor ========
    # acc : 0.83707103301617
    # [0.47826383 0.07366086 0.0509511  0.02446287 0.02366972 0.14824368
    # 0.0921493  0.10859864]
    
    # plot_feature_importance_datasets 함수 이용하여 각 subplot에 그래프 그리기
    plot_feature_importance_datasets(model, axes[i])

plt.tight_layout()  # subplot 간격 자동 조정
plt.show()