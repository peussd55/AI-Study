### <<37>>

# feature_importances : 트리기반모델에서만 제공
# 44카피

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np

seed = 123
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (150, 4) (150,)

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
========= DecisionTreeClassifier ========
acc : 0.8333333333333334
[0.0125     0.03       0.92133357 0.03616643]
========= RandomForestClassifier ========
acc : 0.9333333333333333
[0.08860798 0.0214775  0.48716455 0.40274997]
========= GradientBoostingClassifier ========
acc : 0.9666666666666667
[0.00157033 0.02147603 0.82483538 0.15211825]
========= XGBClassifier ========
acc : 0.9333333333333333
[0.02430454 0.02472077 0.7376847  0.21328996]
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
plot_importance(model, importance_type = 'gain', title = 'feature importance [gain]')  
# F Score : Frequency Score : Tree에서 맨 밑 값을 찾기위해 방문(split)한 총 횟수. 성능이 높으면 F Score가 높을 확률이있다(dnn에서 레이어의 깊이와 비슷)
"""
[importance_type]
 'weight' : default. 얼마나 자주 split했나. 통상 빈도수(Frequency)
 'gain' : split이 모델의 성능을 얼마나 개선했나 // 통상적으로 많이 씀
 'cover' : split하기 위한 sample 수 // 별로 안 씀
"""

# plot_feature_importance_datasets(model)
plt.show()

