### <<49>>

# mnist로 커스텀데이터셋 만들기

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST

path = './_data/torch/'
train_dataset = MNIST(path, train=True, download=True)
test_dataset = MNIST(path, train=False, download=True)

# 1. 커스텀 데이터셋 만들기
class MyDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images      # 이미지 데이터 텐서 (e.g., shape: [60000, 28, 28])
        self.labels = labels      # 라벨 데이터 텐서 (e.g., shape: [60000])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        return image, label
    
# 2. 인스턴스 생성
dataset = MyDataset(
        train_dataset.data/255,
        train_dataset.targets
)

# 3. DataLoader에 쏙
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 4. 출력
for batch_idx, (image, labels) in enumerate(loader):
    print("========배치 :", batch_idx, "=========")
    print("image :", image)
    print("labels :", labels)
    
