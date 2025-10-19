### <<38>>

from sklearn.datasets import fetch_covtype

from sklearn.model_selection import train_test_split
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.preprocessing import LabelEncoder

seed = 123
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape)     
print(np.unique(y, return_counts=True))  

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    stratify=y,
)

# 2. 모델
model = KNeighborsClassifier(n_neighbors=5)

model.fit(x_train, y_train)

print("=========", model.__class__.__name__, "========")
print('acc :', model.score(x_test, y_test)) 

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('accuracy_score :', acc)
f1 = f1_score(y_test, y_pred, average='macro')
print('f1_score :', f1)

# ========= KNeighborsClassifier ========
# acc : 0.9689680989303202
# accuracy_score : 0.9689680989303202
# f1_score : 0.9410571064110705
