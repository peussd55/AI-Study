### <<35>>

from tensorflow.keras.datasets import mnist
import  numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, _), (x_test, _) = mnist.load_data()   # y_train, y_test는 받지 않겠다
print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)

x = np.concatenate([x_train, x_test], axis=0)
print(x.shape)  # (70000, 28, 28)

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
print(x.shape)  # (70000, 784)

pca = PCA(n_components=x.shape[1])
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr) # 누적합
print(evr_cumsum)
print(len(evr_cumsum))  # 784

n_comp_100 = np.argmax(evr_cumsum >= 1.0) +1
print(n_comp_100)   # 713

n_comp_999 = np.argmax(evr_cumsum >= 0.999) +1
print(n_comp_999)   # 486

n_comp_99 = np.argmax(evr_cumsum >= 0.99) +1
print(n_comp_99)   # 331

n_comp_95 = np.argmax(evr_cumsum >= 0.95) +1
print(n_comp_95)   # 154

# 조건별 필요한 주성분 개수 찾기 함수
def find_n_components(evr_cumsum, threshold, tolerance=1e-10):
    """특정 임계값을 처음 넘는 주성분 개수를 반환"""
    indices = np.where(evr_cumsum >= (threshold - tolerance))[0]
    if len(indices) > 0:
        return indices[0] + 1  # 인덱스는 0부터 시작하므로 +1
    else:
        return len(evr_cumsum)  # 모든 주성분이 필요

# 1. 1.0일 때 몇개?
tolerance = 1e-10
n_comp_100 = find_n_components(evr_cumsum, 1.0 ,tolerance)
print(784-n_comp_100, '개')   # 71 개
# 2. 0.999 이상 몇개?
n_comp_999 = find_n_components(evr_cumsum, 0.999)
print(784-n_comp_999, '개')   # 298 개
# 3. 0.99 이상 몇개?
evr_cumsum_99 = find_n_components(evr_cumsum, 0.99)
print(784-evr_cumsum_99, '개')  # 453 개
# 4. 0.95 이상 몇개?
evr_cumsum_95 = find_n_components(evr_cumsum, 0.95)
print(784-evr_cumsum_95, '개')  # 630 개