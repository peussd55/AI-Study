### <<35>>

# PCA : 주성분분석

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA

# 1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets.target
print(x.shape, y.shape)     # (150, 4) (150,)

#### pca가 들어갈때는 보통 pca전에 스케일러함
scaler = StandardScaler()
x = scaler.fit_transform(x)

x_copied = x.copy()

pca = PCA(n_components=3)   # n_componets : 몇 개의 컬럼으로 압축할 것인지
x_copied = pca.fit_transform(x)
print(x)                    
print(x.shape)              # # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(
   x_copied, y, test_size=0.2, 
   stratify=y, 
)

# 2. 모델
models = [
    XGBClassifier(random_state=42),
    RandomForestClassifier(random_state=42),
    CatBoostClassifier(verbose=0, random_state=42),
    LGBMClassifier(verbose=0, verbosity=-1, random_state=42)
]

# 3. 훈련
for i, model in enumerate(models):
    print(f"Processing {model.__class__.__name__}...")
    model.fit(x_train, y_train)
    results = model.score(x_test, y_test)
    #  print(x.shape)
    print(x.shape, '의 score :', results)
    
   #  Processing XGBClassifier...
   # (150, 3) 의 score : 0.9666666666666667
   # Processing RandomForestClassifier...
   # (150, 3) 의 score : 0.9666666666666667
   # Processing CatBoostClassifier...
   # (150, 3) 의 score : 0.9666666666666667
   # Processing LGBMClassifier...
   # (150, 3) 의 score : 0.9666666666666667

# 4. 평가
# (150, 4) 의 score : 1.0
# (150, 3) 의 score : 0.8333333333333334
# (150, 2) 의 score : 1.0