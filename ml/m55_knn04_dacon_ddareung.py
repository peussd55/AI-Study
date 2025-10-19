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
path = './_data/dacon/따릉이/'          # 시스템 경로에서 시작.

train_csv =  pd.read_csv(path + 'train.csv', index_col=0)     # 0번컬럼을 인덱스컬럼으로 지정
test_csv = pd.read_csv(path + 'test.csv', index_col=0)  
submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)

# 결측치 처리
train_csv = train_csv.fillna(train_csv.mean())
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['count'], axis=1)   # count 컬럼 제거
y = train_csv['count']                  # 타겟 변수

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    # stratify=y,
)

# 2. 모델
model = KNeighborsRegressor(n_neighbors=5)

model.fit(x_train, y_train)

print("=========", model.__class__.__name__, "========")
print('r2 :', model.score(x_test, y_test))

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2 :', r2)

# ========= KNeighborsRegressor ========
# r2 : 0.2881846807954016
# r2 : 0.2881846807954016
