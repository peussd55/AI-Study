### <<35>>

# LDA : 분류에서 강력하지만 회귀에서도 사용가능

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets.target
print(x.shape, y.shape)     # (150, 4) (150,)

# train-test data 분리
x_train, x_test, y_train, y_test = train_test_split(
   x, y, test_size=0.2, 
   stratify=y, 
)

# 스케일링
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# LDA 적용
lda = LinearDiscriminantAnalysis(n_components=2)   # LDA의 n_components는 최대 클래스(y)의 수-1개 까지만 적용가능. 컬럼갯수가 y클래스 수보자 작으면 컬럼갯수를 따라간다.
x_train = lda.fit_transform(x_train, y_train)       # train데이터 fit_trainsform할땐 y도 들어가야한다.
x_test = lda.transform(x_test)

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
    print(x_train.shape, '의 score :', results)
    
    # Processing XGBClassifier...
    # (120, 2) 의 score : 0.9666666666666667
    # Processing RandomForestClassifier...
    # (120, 2) 의 score : 0.9333333333333333
    # Processing CatBoostClassifier...
    # (120, 2) 의 score : 0.9333333333333333
    # Processing LGBMClassifier...
    # (120, 2) 의 score : 0.9333333333333333

# 4. 평가
# (150, 4) 의 score : 1.0
# (150, 3) 의 score : 0.8333333333333334
# (150, 2) 의 score : 1.0