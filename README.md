# AI-Study

딥러닝과 머신러닝을 기초부터 직접 구현하며 공부한 코드와 노트입니다(2025년).

같은 개념을 TensorFlow 1의 저수준 API, Keras, PyTorch로 각각 다시 구현했습니다. 라이브러리가 대신 해 주는 부분(경사하강, 스태킹, 은닉 상태 전달)은 손으로 한 번씩 풀어 썼습니다.

- Python 파일 810개 · 약 9만 4천 줄 · 공부 노트 67개
- 실습 코드라 데이터 경로가 로컬 기준입니다. 실행하려면 경로를 바꿔야 합니다.

## 먼저 볼 파일

| 주제 | 파일 | 내용 |
|---|---|---|
| 경사하강법을 손으로 | [`tf114/tf11_gradientDescent1.py`](tf114/tf11_gradientDescent1.py) | 옵티마이저를 쓰지 않고, 손실을 가중치로 미분한 식으로 가중치를 직접 갱신 |
| 스태킹 앙상블 직접 구현 | [`ml/m29_Stacking_생코딩02_california.py`](ml/m29_Stacking_생코딩02_california.py) | 모델별 예측을 열로 쌓아 다시 학습하는 과정을 라이브러리의 스태킹 없이 작성 |
| RNN 은닉 상태 | [`torch/torch22_RNN_hidden_state.py`](torch/torch22_RNN_hidden_state.py) | 초기 은닉 상태를 직접 만들어 넘기고, 마지막 시점만 써서 과적합을 줄임 |
| 전이 학습 비교 | [`keras2/keras74_ALL_TransferLearnig.py`](keras2/keras74_ALL_TransferLearnig.py) | VGG · ResNet · DenseNet · MobileNet · EfficientNet 등 사전 학습 모델 20여 종을 같은 틀로 비교 |
| 잡음 제거 합성곱 오토인코더 | [`AE/a06_CAE.py`](AE/a06_CAE.py) | 잡음을 넣은 MNIST를 합성곱 · 업샘플링 구조로 복원 |
| 입력이 여럿인 텍스트 분류 | [`keras/keras66_4_kaggle_nlp_enssemble.py`](keras/keras66_4_kaggle_nlp_enssemble.py) | 본문 · 키워드 · 위치를 각각 입력으로 받아 합치는 함수형 모델 |

## 폴더별 내용

| 폴더 | 파일 수 | 다룬 것(공부한 순서) |
|---|---|---|
| `keras/` | 296 | 회귀 · 분류 기초 → 검증과 과적합 → 모델 저장과 체크포인트 → 스케일러 → Dropout → 함수형 API → CNN → 데이터 증강 → RNN · LSTM → Conv1D → 앙상블 → 토크나이저 · 임베딩 · 텍스트 분류 |
| `ml/` | 266 | 교차 검증 → 하이퍼파라미터 탐색(Grid · Random · Halving · Bayesian) → Bagging · Voting · Stacking → PCA · LDA → 피처 중요도와 선택 → SMOTE → 파이프라인 |
| `torch/` | 80 | 텐서와 자동 미분 → MLP(회귀 · 이진 · 다중 분류) → DataLoader · CustomDataset → CIFAR DNN · CNN → RNN(다층 · 양방향 · 은닉 상태) · LSTM → 모델 저장과 불러오기 |
| `tf114/` | 44 | TensorFlow 1 저수준 API: placeholder · Variable · 세션 → 선형 회귀 → 경사하강 직접 계산 → sigmoid · softmax → XOR과 MLP |
| `keras2/` | 35 | 옵티마이저와 학습률 조정 → 하이퍼파라미터 탐색 → 전이 학습(VGG · 부분 동결) → 활성화 함수 비교(SiLU · Mish 등) |
| `AE/` | 9 | 오토인코더 · 잡음 제거 · 합성곱 오토인코더 |
| `llm/` · `fastapi/` | 7 | OpenAI API 챗봇(멀티턴) · FastAPI 기초 |
| `pandas/` · `seaborn/` · `python/` · `python_import/` | 74 | 데이터 처리와 시각화 · 파이썬 기초 |
| `html/` · `js/` | 15 | 웹 기초 |
| `메모/` | 67 | 공부 노트: 손실 함수 · 활성화 함수 · 평가 지표 · 스케일링 · CNN · 양방향 LSTM · PyTorch와 Keras의 차이 · 개발 환경 설정 |

## 쓴 것

Python · TensorFlow(1.14 · 2.x) · Keras · PyTorch · scikit-learn · XGBoost · LightGBM · CatBoost · pandas · seaborn
