### <<06>>
# 11_3의 하이퍼파라미터튜닝 테스트케이스코드

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np 

# 하이퍼파라미터 변수 (사용자가 쉽게 조정 가능하도록 가장 위에 위치)
epochs_list = [100]
batch_size_list = [1]
hidden_layers_list = [
    # 피라미드(넓은 시작, 좁은 끝, 일반적인) 구조(4개)
    [32, 16, 8, 4, 2],
    [64, 32, 16, 8, 4],  
    [128, 64, 32, 16, 8], 
    [256, 128, 64, 32, 16],
    [512, 256, 128, 64, 32],
    [1024, 512, 256, 128, 64],

    # 균일 구조
    [8, 8, 8, 8,8],
    [16, 16, 16, 16, 16],
    [32, 32, 32, 32, 32],
    [64, 64, 64, 64, 64],
    [128, 128, 128, 128, 128],
    [256, 256, 256, 256, 256],
    
    # 병목(중간이 좁아지는 구조), 역피라미드구조는 성능이 그다지 좋지 않다.
]

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (442, 10) (442,)

# train_test_split 고정
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.1,
    random_state=999
)

best_r2 = -np.inf
best_params = {}
best_model = None

# 하이퍼파라미터 튜닝 시작
for epochs in epochs_list:
    for batch_size in batch_size_list:
        for i, hidden_layers in enumerate(hidden_layers_list):
            # 모델 구성
            model = Sequential()
            model.add(Dense(hidden_layers[0], input_dim=10))
            for units in hidden_layers[1:]:
                model.add(Dense(units))
            model.add(Dense(1))
            
            # 컴파일, 훈련
            model.compile(loss='mse', optimizer='adam')
            model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            
            # 평가, 예측
            results = model.predict(x_test)
            r2 = r2_score(y_test, results)
            
            print(f"epochs: {epochs}, batch_size: {batch_size}, hidden_layers: {i+1}, r2: {r2:.4f}")
            
            # 최고 성능 모델 저장
            if r2 > best_r2:
                best_r2 = r2
                best_params = {
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'hidden_layers': hidden_layers
                }
                best_model = model
            
            # 목표 달성 확인
            if r2 >= 0.62:
                print(f"\n목표 R2 스코어 달성! R2: {r2:.4f}")
                print(f"최적 파라미터: epochs={epochs}, batch_size={batch_size}, hidden_layers={hidden_layers}")
                
                # 최종 모델 평가
                loss = best_model.evaluate(x_test, y_test, verbose=0)
                print(f"최종 Loss: {loss:.4f}")
                print(f"최종 R2 스코어: {best_r2:.4f}")
                
                # 조기 종료
                break
        
        # 목표 달성 시 조기 종료
        if best_r2 >= 0.62:
            break
    
    # 목표 달성 시 조기 종료
    if best_r2 >= 0.62:
        break

# 목표 달성 여부 확인
if best_r2 < 0.62:
    print("\n목표 R2 스코어에 도달하지 못했습니다.")
    print(f"최고 R2 스코어: {best_r2:.4f}")
    print(f"최적 파라미터: {best_params}")
    
    # 최고 성능 모델로 최종 평가
    loss = best_model.evaluate(x_test, y_test, verbose=0)
    print(f"최종 Loss: {loss:.4f}")
