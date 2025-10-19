
"""
https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016
"""

import numpy as np
import pandas as pd 
from tensorflow.keras.models import Sequential, load_model 
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Dropout, BatchNormalization
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터
# 데이터 load
path = 'c:/Study25/_data/Kaggle/jena/'        
# Date Time을 인덱스로 지정
train_csv = pd.read_csv(path + 'jena_climate_2009_2016.csv')

wv =  train_csv['wv (m/s)']
bad_wv = wv == -9999.0
wv[bad_wv] = 0.0

max_wv = train_csv['max. wv (m/s)']
bad_max_wv = max_wv == -9999.0
max_wv[bad_max_wv] = 0.0

np.random.seed(42)
data = {
    'wd (deg)': np.random.rand(1000) * 360,
    'wv (m/s)': np.random.rand(1000) * 20,
    'max. wv (m/s)': np.random.rand(1000) * 30,
    'T (degC)': np.random.rand(1000) * 40 - 10,
    'p (mbar)': np.random.rand(1000) * 50 + 950,
    'rh (%)': np.random.rand(1000) * 100
}
df = pd.DataFrame(train_csv)

# 상관계수 행렬 계산
corr_matrix = df.corr()

# 히트맵 생성
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 상삼각 마스크
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
            square=True, fmt='.2f', mask=mask, 
            cbar_kws={"shrink": .8})
plt.title('특성 간 상관계수 히트맵', fontsize=16)
plt.tight_layout()
plt.show()