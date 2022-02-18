
 
import numpy as np
import pandas as pd 

import tushare as ts
 
data = ts.get_k_data('300542')['close'].values
print(data.shape)
 
import matplotlib.pyplot as plt
 
data = data.astype('float32')
mx = np.max(data)
mn = np.min(data)
data = (data - mn) / (mx - mn)
 
input_len = 1
 
def generate_dataset(data, days_for_train):
    dataset_x, dataset_y = [], []
    for i in range(len(data) - days_for_train):
        cur_x = data[i:(i + days_for_train)]
        cur_y = data[i + days_for_train]
        dataset_x.append(cur_x)
        dataset_y.append(cur_y)
    return np.array(dataset_x), np.array(dataset_y)
 
import torch
import torch.nn as nn
import torch.optim as optim
 
print(torch.cuda.is_available())
print(torch.cuda.device_count())
 
dataset_x, dataset_y = generate_dataset(data, input_len)
train_len = int(len(dataset_x) * 0.7)
train_x, train_y = dataset_x[:train_len], dataset_y[:train_len]
train_x, train_y = torch.from_numpy(train_x), torch.from_numpy(train_y)
train_x = train_x.reshape(-1, 1, input_len)
train_y = train_y.reshape(-1, 1, 1)
 
class Regression(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        #self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, _x):
        x, _ = self.gru(_x)
        s, b, h = x.shape
        x = x.reshape(s * b, h)
        x = self.fc(x)
        x = x.reshape(s, b, 1)
        #x = self.dropout(x)
        return x
 
loss_function = nn.MSELoss()
epochs = 1000
model = Regression(input_len, hidden_size=10, output_size=1, num_layers=1)
opt = optim.SGD(model.parameters(), lr=0.2)
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
model = model.to(device)
train_x = train_x.to(device)
train_y = train_y.to(device)
 
model.train()
 
for epoch in range(epochs):
    opt.zero_grad()
    
    out = model.forward(train_x)
    loss = loss_function(out, train_y)
    
    loss.backward()
    opt.step()
    
    if (epoch + 1) % 100 == 0:
        print("Epoch", epoch+1)
        print("Loss:", loss.item())
 
model = model.eval()
dataset_x = dataset_x.reshape(-1, 1, input_len)
dataset_x = torch.from_numpy(dataset_x).to(device)
 
pred = model.forward(dataset_x)
pred = pred.reshape(len(dataset_x))
pred = torch.cat((torch.zeros(input_len), pred.cpu()))
pred = pred.detach().numpy()
 
assert len(pred) == len(data)
 
plt.plot(pred, 'r', label='prediction')
plt.plot(data, 'b', label='data')
plt.plot((train_len, train_len), (0, 1), 'g--')
plt.legend()
plt.show()
 
data *= (mx - mn)
data += mn
pred *= (mx - mn)
pred += mn
 
plt.plot(pred, 'r', label='prediction')
plt.plot(data, 'b', label='data')
plt.plot((train_len, train_len), (0, 1), 'g--')
plt.legend()
plt.show()
 
print("Predicted price of 2020.10.26:", pred[-1])
 

 
