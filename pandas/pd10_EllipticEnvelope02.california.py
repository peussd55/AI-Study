import numpy as np
import pandas as pd
import random
seed=2497
random.seed(seed)
np.random.seed(seed)

###### prepare data
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
x,y = fetch_california_housing(return_X_y=True)
columns = data.feature_names
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
# 'Population', 'AveOccup', 'Latitude', 'Longitude']

###### treat outliers 
df = pd.DataFrame(data=x, columns=columns)
from sklearn.covariance import EllipticEnvelope

outliers = EllipticEnvelope(contamination=0.1)
results = outliers.fit_predict(df[['MedInc','HouseAge']])
df_1 = df[['MedInc','HouseAge']]
df_1['EE']=results
df_1.loc[df_1['EE']==1, 'EE']='in'
df_1.loc[df_1['EE']==-1, 'EE']='out'

df_1_1=df_1.drop(columns='HouseAge')
df_1_1['name']='MedInc'
df_1_1.rename(columns={'MedInc':'value'}, inplace=True)
df_1_2=df_1.drop(columns='MedInc')
df_1_2['name']='HouseAge'
df_1_2.rename(columns={'HouseAge':'value'}, inplace=True)

df_2 = pd.concat([df_1_1,df_1_2], axis=0)
print(df_2.head())

###### box plot
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(data=df_2, x='name', y='value', hue='EE')
plt.show()

exit()

    ### train test split
from sklearn.model_selection import train_test_split
x_tr,x_ts,y_tr,y_ts = train_test_split(x,y, train_size=0.8,
                                       shuffle=True, random_state=seed)

    ### scale
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
x_tr = sc.fit_transform(x_tr)
x_ts = sc.transform(x_ts)

###### leran
    ### model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(x_tr,y_tr)

y_pred = lr.predict(x_ts)

###### eval
from sklearn.metrics import r2_score
r2 = r2_score(y_ts, y_pred)
print(r2)
