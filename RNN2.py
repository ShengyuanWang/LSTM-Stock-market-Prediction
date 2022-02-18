import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as Data
from torch.utils.data import DataLoader

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device) #cuda

data = pd.read_csv("EtherPriceHistory(USD).csv")
data.tail()

plt.figure(figsize = (12, 8))
plt.plot(data["Date(UTC)"], data["Value"])
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Ethereum Price History")
plt.show()

# Hyper parameters
threshold = 116
window = 30

input_size = 1
hidden_size = 50
num_layers = 3
output_size = 1

learning_rate = 0.001
batch_size = 16

train_data = data['Value'][:len(data) - threshold]
test_data = data['Value'][len(data) - threshold:]

def create_sequences(input_data, window):
    length = len(input_data)
    
    x = input_data[0:window].values
    y = input_data[1:window+1].values
    
    for i in range(1, length - window):
        x = np.vstack((x, input_data[i:i+window].values))
        y = np.vstack((y, input_data[i+1:window+1+i].values))
        
        sequence = torch.from_numpy(x).type(torch.FloatTensor)
        label = torch.from_numpy(y).type(torch.FloatTensor)
        
        sequence = Data.TensorDataset(sequence, label)
        
    return sequence
train_data = create_sequences(train_data, window)
train_loader = Data.DataLoader(train_data, 
                               batch_size = batch_size, 
                               shuffle = False, 
                               drop_last = True)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        self.hidden = torch.zeros(num_layers, 1, hidden_size)
        
        self.rnn = nn.RNN(input_size, 
                          hidden_size, 
                          num_layers,             # number of recurrent layers
                          batch_first = True,    # Default: False
                                                  # If True, layer does not use bias weights
                          nonlinearity = 'relu',  # 'tanh' or 'relu'
                          #dropout = 0.5
                         )
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # input shape of (batch, seq_len, input_size)
        # output shape of (batch, seq_len, hidden_size)
        out, hidden = self.rnn(x, self.hidden)
        self.hidden = hidden
        
        # output shape of (batch_, seq_len, output_size)
        out = self.fc(out)
        return out
    
    def init_hidden(self, batch_size):
        # hidden shape of (num_layers, batch, hidden_size)
        self.hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
rnn = RNN(input_size, hidden_size, num_layers, output_size).to(device)
rnn

def train(model, num_epochs):
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    
    for epoch in range(num_epochs):

        for i, (sequences, labels) in enumerate(train_loader):

            model.init_hidden(batch_size)

            sequences = sequences.view(-1, window, 1)
            labels = labels.view(-1, window, 1)
            
            pred = model(sequences)
            cost = criterion(pred[-1], labels[-1])
            
            optimizer.zero_grad()
            cost.backward()
            
            #防止梯度爆炸问题
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            
            optimizer.step()
    
        print("Epoch [%d/%d] Loss %.4f"%(epoch+1, num_epochs, cost.item()))
    
    print("Training Finished!")

train(rnn, 10)

def evaluation(model):
    model.eval()
    model.init_hidden(1)
    
    val_day = 30
    dates = data['Date(UTC)'][1049+window:1049+window+val_day]
    
    pred_X = []
    
    for i in range(val_day):
        X = torch.from_numpy(test_data[i:window+i].values).type(torch.FloatTensor)
        X = X.view(1, window, 1).to(device)

        pred = model(X)
        pred = pred.reshape(-1)
        pred = pred.cpu().data.numpy()

        pred_X.append(pred[-1])

    y = test_data[window:window+val_day].values
    
    plt.figure(figsize = (12, 8))
    plt.plot(dates, y, 'o-', alpha = 0.7, label = 'Real')
    plt.plot(dates, pred, '*-', alpha = 0.7, label = 'Predict')
    
    plt.xticks(rotation = 45)
    plt.xlabel("Date")
    plt.ylabel("Ethereum Price (USD)")
    plt.legend()
    
    plt.title("Comparison between Prediction and Real Ethereum BitCoin Price")
    plt.show()