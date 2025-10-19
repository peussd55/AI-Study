### <<08>>

# 18-1 카피

from sklearn.datasets import load_boston
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from sklearn.model_selection import train_test_split

# 1. 데이터
dataset = load_boston()
print('보스톤데이터셋 :', dataset)
print('보스톤데이터셋의 설명 :', dataset.DESCR)
print('보스톤데이터셋 특성이름:', dataset.feature_names)
#exit()

x = dataset.data
y = dataset.target

print(x)
print(y)
print(x.shape, y.shape)  #(506, 13) (506,)

#### 앵그러봐(r2 기준 0.75이상) ####

# 1. 데이터
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.3,
    random_state=1111
)
print('x_test :', x_test)
print('y_test :', y_test)

# 2. 모델구성
model = Sequential()
model.add(Dense(128, input_dim=13))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear')) 

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
# val_loss를 기준으로 최소값을 찾을건데 patience를 10으로 한다. 10번 내에 최소값을 찾으면 loss를 갱신하고 10번을 더 훈련시킨다. 10번을 훈련을 더 하는동안 최소값이 안 나타나면 훈련을 종료한다.
es = EarlyStopping( 
    monitor = 'val_loss',       # 기준을 val_loss로
    mode = 'min',               # 최대값 : max, 알아서 찾아줘 : auto
    patience=1000,                # 참는 횟수는 10번
    restore_best_weights=True,  # 가장 최소지점을 save할것인지. default = False. False가 성능이 더 잘나오면 모델이 과적합됐을 수 있음. False 마지막 종료시점의 가중치를 저장한다.
    verbose=1,                  # stop이 어느 epoch에서 걸렸는지 출력해줌
)

hist = model.fit(x_train, y_train, 
                 epochs=10000, 
                 batch_size=24, 
                 verbose=3, 
                 validation_split=0.2,
                 callbacks=[es],
                 )

print("=============== hist =================")
print(hist)     # <keras.callbacks.History object at 0x00000179B5A08BB0>
print("=============== hist.history =================")
print(hist.history)
# {'loss': [147.9894561767578, 65.87415313720703, 69.58604431152344, 66.62543487548828, 63.503456115722656, 57.571075439453125, 63.87793731689453, 58.730751037597656, 54.63190460205078, 47.30581283569336], 'val_loss': [74.74795532226562, 97.66914367675781, 81.14277648925781, 94.65666198730469, 64.31427764892578, 60.60000991821289, 64.17593383789062, 63.86719512939453, 56.42604064941406, 68.24020385742188]}
# model.fit 의 history로 loss, val_loss를 뽑아서 그래프 시각화 할 수 있다.
# {} : 딕셔너리 // [] : 리스트
print("=============== loss =================")
print(hist.history['loss'])
print("=============== val_loss =================")
print(hist.history['val_loss'])

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))       # 9 x 6 사이즈
plt.plot(hist.history['loss'], c='red', label='loss')   # plot (x, y, color= ....) : y값만 넣으면 x는 1부터 시작하는 정수 리스트
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
#plt.rcParams['font.family'] = 'Malgun Gothic'
#plt.rc('font', family='Malgun Gothic')
plt.title('Boston Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper right')   # 유측 상단에 라벨표시 
plt.grid()  #격자표시
plt.show()


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)   # 훈련이 끝난 모델의 loss를 한번 계산해서  반환
results = model.predict(x_test)

print('loss :', loss)
print('x_test :', x_test)
#print('x_test의 예측값 :', results)

from sklearn.metrics import r2_score, mean_squared_error

# def RMSE(y_test, y_predict):
#     # mean_squared_error : mse를 계산해주는 함수
#     return np.sqrt(mean_squared_error(y_test, y_predict))

# rmse = RMSE(y_test, results)
# print('RMSE :', rmse)

r2 = r2_score(y_test, results)
print('r2 스코어 :', r2)
# r2 스코어 : 0.8059879794485231

"""
EarlyStopping X
epochs=1000, 
batch_size=24, 
r2 스코어 : 0.8136751901361008

EarlyStopping O / restore_best_weights=True
patience=1000
epochs=10000, 
batch_size=24, 
stop지점 : 1229,
r2 스코어 : 0.8099889181016016

EarlyStopping O / restore_best_weights=False
patience=1000
epochs=10000, 
batch_size=24, 
stop지점 : 3061,
r2 스코어 : 0.7946116029715309
"""