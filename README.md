# AI-Study

## 1. 저장소 개요
이 저장소는 인공지능 전반을 학습하기 위해 진행한 실습 코드와 예제를 모아둔 포트폴리오입니다. 기본 파이썬 문법부터 데이터 전처리, 고전 머신러닝, 딥러닝 프레임워크 실험, 대규모 언어 모델 연동, 웹 프론트엔드 및 백엔드 튜토리얼까지 폭넓은 학습 기록을 체계적으로 정리했습니다.

## 2. 핵심 기술 스택
| 분류 | 사용 기술 | 활용 맥락 |
| --- | --- | --- |
| 프로그래밍 언어 | Python, JavaScript | 알고리즘 실습, 웹 프론트엔드 샘플 제작 |
| 데이터 처리 | NumPy, pandas, Scikit-learn | 데이터 전처리, 피처 엔지니어링, 분류/회귀 모델링 |
| 시각화 | Matplotlib, Seaborn | 모델 결과 및 데이터 탐색 시각화 |
| 딥러닝 프레임워크 | TensorFlow/Keras, PyTorch | 오토인코더, MLP, CNN, 최적화 실험 |
| 클라우드 & LLM | OpenAI API, Streamlit | 대화형 챗봇 및 LLM 기반 애플리케이션 |
| 웹 프레임워크 | FastAPI, HTML/CSS/JS | REST API 및 동적인 UI 구성 |

## 3. 모듈별 프로젝트 요약
### 3.1 오토인코더 실험 (AE/)
- **목적**: MNIST 데이터를 활용해 오토인코더 구조를 설계하고, 인코더/디코더 노드 수, 손실 함수에 따른 재구성 품질 변화를 비교합니다.
- **핵심 기술**: TensorFlow Keras의 `Model`/`Dense` API, `binary_crossentropy` 손실, Matplotlib 시각화.

### 3.2 TensorFlow Keras 기초 (keras/)
- **목적**: 심층 신경망, 배치 처리, 다층 퍼셉트론 등 Keras 기본기를 다집니다.
- **핵심 기술**: `Sequential` 모델 구성, 다양한 활성화 함수/최적화 전략, 다차원 데이터 처리.

### 3.3 고급 Keras 최적화 실험 (keras2/)
- **목적**: ReduceLROnPlateau, Bayesian HyperParameter Search 등 고급 튜닝 기법으로 캘리포니아 주택, 따릉이 대여 등 실제 데이터셋 성능을 개선합니다.
- **핵심 기술**: 학습률 스케줄링, 하이퍼파라미터 탐색, Kaggle/Dacon 데이터셋 전처리.

### 3.4 TensorFlow 1.14 저수준 실습 (tf114/)
- **목적**: Placeholder, Variable, GradientDescent 등 저수준 연산을 직접 다루며 TensorFlow 1.x 그래프 실행 모델을 이해합니다.
- **핵심 기술**: 세션 기반 학습, 손실 시각화, 다양한 데이터셋 적용.

### 3.5 PyTorch 딥러닝 베이스라인 (torch/)
- **목적**: GPU 환경 설정, `nn.Sequential`을 통한 다층 퍼셉트론 구현, 표준화 등 PyTorch 워크플로우를 익힙니다.
- **핵심 기술**: CUDA 디바이스 전환, 사용자 정의 `train`/`evaluate` 루프, 텐서 스케일링.

### 3.6 전통 머신러닝 파이프라인 (ml/)
- **목적**: Iris, Breast Cancer, Kaggle Bank Marketing 등 표준 데이터셋으로 분류 알고리즘을 비교하고 모델 직렬화를 연습합니다.
- **핵심 기술**: `train_test_split`, 전처리(결측치/스케일링), `LinearSVC`, `RandomForestClassifier` 등 다중 모델 평가.

### 3.7 데이터 전처리 실습 (pandas/)
- **목적**: 결측치 보간, `IterativeImputer`, MICE 등 다양한 결측치 처리 전략을 탐구합니다.
- **핵심 기술**: `DataFrame` 조작, 통계 기반 값 대체, 사이킷런 Imputer 도구 연동.

### 3.8 Python 기초 및 모듈 시스템 (python/, python_import/)
- **목적**: 반복문, 클래스 초기화, 모듈 임포트 패턴 등 Python 기본기를 정리합니다.
- **핵심 기술**: 사용자 정의 클래스, `__init__` 메서드, `import`/`from` 구문 구조.

### 3.9 Seaborn 탐색적 데이터 분석 (seaborn/)
- **목적**: Figure-level/Axes-level 함수, 긴/넓은 포맷 데이터를 활용한 고급 시각화 패턴을 실습합니다.
- **핵심 기술**: Seaborn의 선언적 API, 다중 뷰 구성, 스타일 커스터마이징.

### 3.10 웹 프론트엔드 기초 (html/, js/)
- **목적**: 기본 HTML 입력 폼과 조건문, JavaScript 함수 및 DOM 조작을 연습합니다.
- **핵심 기술**: 이벤트 처리, 조건부 렌더링, 외부 JS 스크립트 연동.

### 3.11 FastAPI 마이크로 서비스 (fastapi/)
- **목적**: 간단한 REST 엔드포인트를 구현해 FastAPI의 라우팅과 경로 파라미터 처리 방식을 학습합니다.
- **핵심 기술**: 경로/쿼리 파라미터 정의, JSON 응답 생성.

### 3.12 LLM 기반 챗봇 (llm/)
- **목적**: OpenAI API를 Streamlit 앱과 연계해 맞춤형 대화형 챗봇을 구축합니다.
- **핵심 기술**: `.env` 환경 변수 관리, 세션 상태 기반 메시지 히스토리, Chat Completions API 호출.

## 4. 학습 여정과 성과
1. **기초 다지기**: Python 언어와 모듈 시스템을 정리해 이후 프로젝트의 기반을 마련했습니다.
2. **데이터 이해**: pandas/Seaborn 실습으로 데이터를 정제하고 시각화하는 역량을 강화했습니다.
3. **모델링 확장**: Scikit-learn 전통 모델부터 TensorFlow, PyTorch 딥러닝 모델까지 구현하며 전반적인 AI 스택을 경험했습니다.
4. **서비스 연동**: FastAPI와 Streamlit을 사용해 모델/LLM을 서비스 형태로 노출하는 엔드투엔드 흐름을 완성했습니다.

## 5. 향후 계획
1. **학습내용 문서화**
