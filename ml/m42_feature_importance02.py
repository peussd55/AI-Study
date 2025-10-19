### <<36>>

# feature_importances : 트리기반모델에서만 제공

from sklearn.datasets import load_iris, load_diabetes
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
x, y = load_diabetes(return_X_y=True)
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

print("=======(시드값):", seed ,"==========")
for model in models:
    model.fit(x_train, y_train)
    print("=========", model.__class__.__name__, "========")
    print('r2 :', model.score(x_test, y_test))
    print(model.feature_importances_)
    
    # =======(시드값): 123 ==========
    # ========= DecisionTreeRegressor ========
    # r2 : 0.15795709914946876
    # [0.09559417 0.01904038 0.23114463 0.0534315  0.03604905 0.05879742
    # 0.04902482 0.01682605 0.36525519 0.07483678]
    # ========= RandomForestRegressor ========
    # r2 : 0.5260875642282989
    # [0.05770917 0.01047587 0.28528549 0.09846103 0.04390962 0.05190847
    # 0.05713042 0.02626033 0.28720491 0.08165469]
    # ========= GradientBoostingRegressor ========
    # r2 : 0.5583569356618018
    # [0.04935014 0.01077655 0.30278452 0.11174122 0.02686628 0.05718503
    # 0.04058792 0.01773638 0.33840513 0.04456684]
    # ========= XGBRegressor ========
    # r2 : 0.39065385219018145
    # [0.04159961 0.07224615 0.17835377 0.06647415 0.04094251 0.04973729
    # 0.03822911 0.10475955 0.3368922  0.07076568]
    
    # =======(시드값): 321 ==========
    # ========= DecisionTreeRegressor ========
    # r2 : -0.10392921469846073
    # [0.11389836 0.0023501  0.23792837 0.05579891 0.04020933 0.05453858
    # 0.0333886  0.02271428 0.37044018 0.06873329]
    # ========= RandomForestRegressor ========
    # r2 : 0.41101230261083677
    # [0.05758182 0.01023597 0.25338776 0.09164269 0.04148326 0.04832844
    # 0.06387366 0.02422506 0.33957909 0.06966223]
    # ========= GradientBoostingRegressor ========
    # r2 : 0.4074557970911611
    # [0.047488   0.01409435 0.25070525 0.08840658 0.02387153 0.04549464
    # 0.06747747 0.01836681 0.38418562 0.05990976]
    # ========= XGBRegressor ========
    # r2 : 0.24783479111186757
    # [0.03337811 0.05178599 0.180124   0.05708085 0.03160705 0.0595936
    # 0.07715637 0.09329977 0.33657563 0.07939863]
    
    # 컬럼별 기여도 (비율, 총합1)