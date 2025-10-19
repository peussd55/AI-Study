### <<38>>

from sklearn.datasets import load_diabetes

from sklearn.model_selection import train_test_split
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score

seed = 123
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    # stratify=y,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
model = KNeighborsRegressor(n_neighbors=5)

model.fit(x_train, y_train)

print("=========", model.__class__.__name__, "========")
print('r2 :', model.score(x_test, y_test))

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2 :', r2)

# ========= KNeighborsRegressor ========
# r2 : 0.45373900245244714
# r2 : 0.45373900245244714
