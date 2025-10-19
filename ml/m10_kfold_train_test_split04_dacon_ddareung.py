### <<32>>

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
import warnings     # 경고무시
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor

# 1. 데이터
path = './_data/dacon/따릉이/'          
train_csv =  pd.read_csv(path + 'train.csv', index_col=0)    
test_csv = pd.read_csv(path + 'test.csv', index_col=0)  
submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)
x = train_csv.drop(['count'], axis=1) 
y = train_csv['count'] 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8, 
    # stratify=y,   # 분류데이터에서만 사용가능
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333, )

# 2. 모델
model = HistGradientBoostingRegressor()

# 3. 훈련
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('r2_score :', scores, '\n평균 r2_score :', round(np.mean(scores), 4))
# [kfold_train_test_split]
# r2_score : [0.77666226 0.75268977 0.79743558 0.83193943 0.77363227] 
# 평균 r2_score : 0.7865

# [kfold]
# r2_score : [0.79346634 0.74631699 0.7724165  0.80025196 0.82694093] 
# 평균 r2_score : 0.7879

y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)

r2 = r2_score(y_test, y_pred)
print('cross_val_predict r2_score :', round(r2, 4))     # 0.6957