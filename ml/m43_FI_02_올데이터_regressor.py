### <<36>>

# feature_importances : 트리기반모델에서만 제공

from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
import random
import numpy as np
from sklearn.preprocessing import StandardScaler

seed = 123
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
data1 = fetch_california_housing()
data2 = load_diabetes()

datasets = [data1, data2]
dataset_name = ['fetch_california_housing', 'diabetes']

model1 = DecisionTreeRegressor(random_state=seed)
model2 = RandomForestRegressor(random_state=seed)
model3 = GradientBoostingRegressor(random_state=seed)
model4 = XGBRegressor(random_state=seed)
models = [model1, model2, model3, model4]

for i, data in enumerate(datasets):
    x = data.data
    y = data.target
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, random_state=seed,
        # stratify=y,
    )
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    print("================================", dataset_name[i], "====================================")

    # 2. 모델
    for model in models:
        model.fit(x_train, y_train)
        print("=========", model.__class__.__name__, "========")
        print('acc :', model.score(x_test, y_test))
        print(model.feature_importances_)

"""
================================ fetch_california_housing ====================================
========= DecisionTreeRegressor ========
acc : 0.6038812893164018
[0.51931246 0.04981492 0.04834159 0.02632489 0.03346774 0.13169898
 0.09904432 0.09199511]
========= RandomForestRegressor ========
acc : 0.8115747285676449
[0.52255998 0.05206861 0.04703019 0.0292777  0.03162071 0.13398205
 0.09197568 0.09148508]
========= GradientBoostingRegressor ========
acc : 0.7978474110349009
[0.59864938 0.03019079 0.02141916 0.00492639 0.00427861 0.12193384
 0.10819286 0.11040897]
========= XGBRegressor ========
acc : 0.83707103301617
[0.47826383 0.07366086 0.0509511  0.02446287 0.02366972 0.14824368
 0.0921493  0.10859864]
================================ diabetes ====================================
========= DecisionTreeRegressor ========
acc : 0.15795709914946876
[0.09559417 0.01904038 0.23114463 0.0534315  0.03604905 0.05879742
 0.04902482 0.01682605 0.36525519 0.07483678]
========= RandomForestRegressor ========
acc : 0.5242680611224524
[0.05770917 0.01047587 0.28528549 0.09846103 0.04390962 0.05190847
 0.05713042 0.02626033 0.28720491 0.08165469]
========= GradientBoostingRegressor ========
acc : 0.558445212046043
[0.04935014 0.01077655 0.30278452 0.11174122 0.02686628 0.05718503
 0.04058792 0.01773638 0.33840513 0.04456684]
========= XGBRegressor ========
acc : 0.39065385219018145
[0.04159961 0.07224615 0.17835377 0.06647415 0.04094251 0.04973729
 0.03822911 0.10475955 0.3368922  0.07076568]
 
"""