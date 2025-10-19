### <<35>>

# PCA : 주성분분석, 차원축소, x변수가 원핫인코딩됐을때 사용되면 좋다.

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

# 1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets.target
print(x.shape, y.shape)     # (150, 4) (150,)


x_train, x_test, y_train, y_test = train_test_split(
   x, y, test_size=0.2, 
   stratify=y, 
)

#### 스케일링 적용되는 순서 : train/test 분리 뒤,  pca 앞
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

pca = PCA(n_components=3)   # n_componets : 몇 개의 컬럼으로 압축할 것인지
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)     # test도 pca할땐 transform만 한다.
print(x_train)                    
print(x_train.shape)              # # (150, 3)
print(x_test)                    
print(x_test.shape)              # # (150, 3)

# 2. 모델
model = RandomForestClassifier()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가
results = model.score(x_test, y_test)
print(x.shape)
print(x.shape, '의 score :', results)
# (150, 4) 의 score : 1.0
# (150, 3) 의 score : 0.8333333333333334
# (150, 2) 의 score : 1.0
# iris는 데이터가 너무 적어서 성능 뒤죽박죽