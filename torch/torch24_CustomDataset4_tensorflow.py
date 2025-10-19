### <<49>>

# mnist로 커스텀데이터셋 만들기

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# ======================= [수정 포인트 1] =========================
# TensorFlow 대신 numpy를 사용하여 로컬 파일을 직접 로드합니다.

# 1. mnist.npz 파일의 정확한 경로를 지정합니다.
#    (사용자 환경에 맞게 경로를 수정해야 합니다)
#    예: 'C:/Users/사용자명/.keras/datasets/mnist.npz'
#    os.path.expanduser('~')는 홈 디렉토리 경로를 자동으로 찾아줍니다.
path_to_mnist_npz = os.path.join(os.path.expanduser('~'), '.keras', 'datasets', 'mnist.npz')
print(path_to_mnist_npz)

# 2. numpy.load로 데이터 로드
try:
    with np.load(path_to_mnist_npz) as data:
        # .npz 파일은 딕셔너리처럼 키 값으로 데이터에 접근합니다.
        x_train = data['x_train']  # 훈련 이미지
        y_train = data['y_train']  # 훈련 라벨
        x_test = data['x_test']    # 테스트 이미지
        y_test = data['y_test']    # 테스트 라벨
    print(f"'{path_to_mnist_npz}'에서 데이터 로드 성공!")
except FileNotFoundError:
    print(f"'{path_to_mnist_npz}' 경로에 파일이 없습니다. 경로를 확인해주세요.")
    # 파일이 없을 경우 스크립트 종료
    exit()

# 3. NumPy 배열을 PyTorch 텐서로 변환
#    데이터 타입을 float32로 변환하고 0~1 사이로 스케일링하는 것이 일반적입니다.
x_train_tensor = torch.tensor(x_train / 255.0, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long) # 라벨은 보통 long 타입 사용
# ==============================================================


# 1. 커스텀 데이터셋 만들기 (기존과 거의 동일)
class MyDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # CNN 모델에 넣기 좋게 채널 차원을 추가 (28, 28) -> (1, 28, 28)
        return image.unsqueeze(0), label
    
# ======================= [수정 포인트 2] =========================
# 2. 인스턴스 생성 시, 준비된 텐서를 정확하게 전달합니다.
dataset = MyDataset(
    images=x_train_tensor,
    labels=y_train_tensor
)
# ==============================================================


# 3. DataLoader에 쏙
loader = DataLoader(dataset, batch_size=4, shuffle=True)


# 4. 출력 확인
print("\nDataLoader 정상 동작 확인:")
for batch_idx, (images, labels) in enumerate(loader):
    if batch_idx > 2: # 너무 많이 출력되지 않도록 3개 배치만 확인
        break
    print(f"======== 배치 : {batch_idx} =========")
    print("Images batch shape:", images.shape) # e.g., torch.Size([4, 1, 28, 28])
    print("Labels batch shape:", labels.shape) # e.g., torch.Size([4])
