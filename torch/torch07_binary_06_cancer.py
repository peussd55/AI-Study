import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
seed = 2497
random.seed(seed)
np.random.seed(seed)

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

###### prepare data
from sklearn.datasets import load_breast_cancer
x,y = load_breast_cancer(return_X_y=True)

    ### transform to tensor data for torch
x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    ### train test split
from sklearn.model_selection import train_test_split
x_tr, x_ts, y_tr, y_ts = train_test_split(x,y, train_size=0.8,
                                          shuffle=True, random_state=seed)
    ### scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# x_tr = sc.fit_transform(x_tr)
# x_ts = sc.transform(x_ts)

###### learn
    ### model
model = nn.Sequential(
    nn.Linear(30, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.SiLU(),
    nn.Linear(16, 1),
    nn.Sigmoid(),
).to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

    ### fit
def train(model, crit_, optim_, x, y):
    optim_.zero_grad()
    hyp_ = model(x)
    loss = crit_(hyp_,y)
    loss.backward()
    optim_.step()
    return loss.item()
epochs = 100
for i in range(1,epochs+1):
    loss = train(model, criterion, optimizer, x_tr, y_tr)
    print('epoch_{} Loss : {}'.format(epochs, loss))

###### eval / pred
    ### eval
def evaluation(model, crit_, x, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        score = crit_(y, y_pred)
        return score.item()

print('score :', evaluation(model, criterion, x_ts, y_ts))

    ### pred
from sklearn.metrics import accuracy_score
y_pred = torch.round(model(x_ts))
y_pred = y_pred.reshape(-1,1)
acc = accuracy_score(y_ts.tolist(), y_pred.tolist())

y_predict = np.round(y_pred.detach().cpu().numpy())
acc1 = accuracy_score(y_ts.detach().cpu().numpy(), y_predict)
print(acc, acc1)