### <<38>>

from sklearn.datasets import load_digits

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
path = './_data/kaggle/santander/'           

# 맨 앞, 공통 컬럼 datetime을 인덱스 컬럼으로 지정
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

print(train_csv.shape, test_csv.shape, submission_csv.shape)  # (200000, 202) (200000, 201) (200000, 2)

print(train_csv['target'].value_counts())
# 이진분류 (불균형)
# 0    179902
# 1     20098

# train data x로 y 분리
x = train_csv.drop(['target'], axis=1)
print(x)
y = train_csv['target']

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
# acc : 0.898925
# accuracy_score : 0.898925
# f1_score : 0.47730142152713667
