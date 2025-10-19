### <<25>>
import pandas as pd 
import numpy as np
from sklearn.metrics import mean_squared_error

학생csv = "jena_박은수_submit.csv"

path1 = "C:/study25/_data/kaggle/jena/"
path2 = "C:/study25/_save/keras56/"

datasets = pd.read_csv(path1 + 'jena_climate_2009_2016.csv', index_col=0)

y_정답 = datasets.iloc[-144:,-1]    # wd컬럼
print(y_정답)
print(y_정답.shape)

학생꺼 = pd.read_csv(path2 + 학생csv, index_col=0)
print(학생꺼)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_정답, 학생꺼)
print('RMSE :', rmse)
