import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch:',torch.__version__, 'use DEVICE:', DEVICE)
DEVICE = 'cuda'

#1 data
x = np.array([1,2,3])
y = np.array([1,2,3])
xx = torch.tensor([1,2,3])
x = torch.FloatTensor(x)
x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)
x_mean = torch.mean(x)
x_std = torch.std(x)
    ### standard scaling
x = (x - x_mean)/x_std
print('after scaling :', x)

#2 model
# model = nn.Linear(1,1).to(DEVICE) # input, output order
model = nn.Sequential(
    nn.Linear(1,5),
    nn.Linear(5,4),
    nn.Linear(4,1)
).to(DEVICE)

#3 compile / learn
criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
optimizer = optim.SGD(model.parameters(), lr=0.001)

def train(model, criterion, optimizer, x, y):
    # model.train()
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    
    loss.backward() # calculate gradient
    optimizer.step() # renew weight
    
    return loss.item()

epochs = 1000
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x, y)
    print('eopch: {}, loss: {}'.format(epoch, loss[0]))

print('************************************************')
#4 eval / predict
def evaluate(model, criterion, x, y):
    model.eval()
    with torch.no_grad():
        # torch.set_grad_enabled(False)
        y_predict = model(x)
        loss2 = criterion(y, y_predict)
        # torch.set_grad_enabled(True)
    return loss2.item()

loss2 = evaluate(model, criterion, x, y)
print('final_loss :', loss2)

x_pred = (torch.Tensor([[4]]).to(DEVICE)-x_mean)/x_std
result = model(x_pred)
print('prediction of \'4\' :', result.item())



