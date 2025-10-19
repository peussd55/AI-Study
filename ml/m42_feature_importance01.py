### <<36>>

# feature_importances : 트리기반모델에서만 제공

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
x, y = load_iris(return_X_y=True)
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
    print(model.feature_importances_)
    
    # ========= DecisionTreeClassifier ========
    # acc : 0.8333333333333334
    # [0.0125     0.03       0.92133357 0.03616643]
    # ========= RandomForestClassifier ========
    # acc : 0.9333333333333333
    # [0.08860798 0.0214775  0.48716455 0.40274997]
    # ========= GradientBoostingClassifier ========
    # acc : 0.9666666666666667
    # [0.00155689 0.02151625 0.82488114 0.15204571]
    # ========= XGBClassifier ========
    # acc : 0.9333333333333333
    # [0.02430454 0.02472077 0.7376847  0.21328996]
    
    # 컬럼별 기여도 (비율, 총합1)