#encoding=utf-8
import torch
import torch.nn  as nn
import torch.optim
from torch.autograd import Variable

from collections import Counter

import matplotlib 
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

#首先构建RNN网络
class SimpleRnn(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,num_layers=1):
        super(SimpleRnn,self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        
        #一个embedding层
        self.embedding = nn.Embedding(input_size,hidden_size)
        
        ##
        #pytorch的RNN层，batch_first表示可以让输入的张量表示第一个维度batch的指标
        #在定义这个部件的时候，需要制定输入节点的数量input_size 隐藏层节点的数量hidden_size
        #和RNN层的数目
        #我们世界上使用nn.RNN 来构造多层RNN的，只不过每一层RNN的神经元的数量要相同
        self.rnn = nn.RNN(hidden_size,hidden_size,num_layers,batch_first= True)
        
        #输出全连接层
        self.fc = nn.Linear(hidden_size,output_size)
        
        #最后的logsoftmax层
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self,inputs,hidden):
        '''
         基本运算过程：先进行embedding层的计算
         在将这个向量转换为一个hidden_size维度的向量。
         input的尺寸为 batch_size,num_step,data_dim
        '''
        x = self.embedding(inputs)
        #此时x的size为batch_size,num_step.hidden_size
        
        #从输入层到隐藏层的计算
        output,hidden = self.rnn(x,hidden)
        #从输出中选择最后一个时间步的数值，output中包含了所有时间步的结果
        #output的输出size batch_size,num_step,hidden_size
        
        output = output[:,-1,:]
        #此时output的尺寸为[batch_szie,hidden_size]
        
        #最后输入全连接层
        output = self.fc(output)
        #此时output的尺寸为 batch_size,hidden_size
            
        #输入softmax
        output = self.softmax(output)
            
        return output,hidden
        #对隐藏层进行初始化
        #注意尺寸的大小为layer_size,batch_size,hidden_size
    def initHidden(self):
        return Variable(torch.zeros(self.num_layers,1,self.hidden_size))


#下面是训练以及校验的过程
#首先生成01字符串类的数据以及样本的数量
train_set = []
validset = []
sample = 2000

#训练样本中最大的n值
sz = 10

#定义n的不同权重，我们按照10:6:4:3:1:1....来配置n=1，2，3，4，5
probablity = 1.0 *np.array([10,6,4,3,1,1,1,1,1,1])

#保证n的最大值是sz
probablity = probablity[:sz]

#归一化，将权重变成概率
probablity = probablity / sum(probablity)



#开始生成sample这么多个样本,
#每一个样本的长度是根据概率来定义的
for m in range(2000):
    #对于随机生成的字符串，随机选择一个n，n被选择的权重记录在probablity
    #n表示长度
    #range生成序列，p表示通过之前定义的probablity的概率分布进行抽样
    n = np.random.choice(range(1,sz+1),p=probablity)
    #生成程度为2n这个字符串，用list的形式完成记录
    inputs = [0]*n + [1]*n
    #在最前面插入3表示开始字符，在结尾插入2表示结束符
    inputs.insert(0,3)
    inputs.append(2)
    train_set.append(inputs)
    
#在生成sample/10的校验样本
for m in range(sample // 10):
    n =np.random.choice(range(1,sz+1),p=probablity)
    inputs = [0] * n + [1] *n
    inputs.insert(0,3)
    inputs.append(2)
    validset.append(inputs)
    
#再生成若干个n特别大的样本用于校验
for m in range(2):
    n = sz + m
    inputs = [0] * n + [1] *n
    inputs.insert(0,3)
    inputs.append(2)
    validset.append(inputs)

#下面是训练过程
#输入的size是4，可能的值为0,1,2,3
#输出size为3 可能为 0,1 2

rnn = SimpleRnn(input_size=4, hidden_size=2, output_size=3)
criterion = torch.nn.NLLLoss() #定义交叉熵函数
optimizer = torch.optim.Adam(rnn.parameters(),lr=0.001) #采用Adam算法

#重复进行50次试验
num_epoch = 50
results = []
for epoch in range(num_epoch):
    train_loss = 0
    np.random.shuffle(train_set)
    #对每一个序列进行训练
    for i,seq in enumerate(train_set):
        loss = 0
        hidden = rnn.initHidden() #初始化隐含层的神经元、
        #对于每一个序列的所有字符进行循环
        for t in range(len(seq)-1):
            #当前字符作为输入，下一个字符作为标签
            x = Variable(torch.LongTensor([seq[t]]).unsqueeze(0))
            # x的size为 batch_size=1。time_steps=1，data_dimension = 1
            y = Variable(torch.LongTensor([seq[t+1]]))
            #y的size batch_size =1 data_dimension =1
            output,hidden = rnn(x,hidden)
            #output 的size：batch_size,output_size=3
            #hidden尺寸 layer_size = 1,batch_size = 1,hidden_size
            loss += criterion(output,y)
        
        #计算每一个字符的损失数值
        loss = 1.0 * loss / len(seq)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss
        
        #打印结果
        if i>0 and i % 500 ==0:
            print('第{}轮，第{}个，训练平均loss：{:.2f}'.format(epoch,i,train_loss.data.numpy()/i))
            
    #下面在校验集上进行测试
    valid_loss = 0
    errors = 0
    show_out_p =''
    show_out_t = ''
    for i,seq in enumerate(validset):
        loss = 0
        outstring = ''
        targets = ''
        diff = 0
        hidden = rnn.initHidden()
        for t in range(len(seq)-1):
            x = Variable(torch.LongTensor([seq[t]]).unsqueeze(0))
            y = Variable(torch.LongTensor([seq[t+1]]))
            output,hidden = rnn(x,hidden)
            data = output.data.numpy()
            print("the output is ",data)
            #获取最大概率输出
            mm = torch.max(output,1)[1][0]
            #以字符的形式添加到outputstring中
            outstring += str(mm.data.numpy())
            targets += str(y.data.numpy()[0])
            loss += criterion(output,y) #计算损失函数
            #输出模型预测字符串和目标字符串之间差异的字符数量
            diff += 1 - mm.eq(y).data.numpy()[0]
        loss = 1.0 * loss / len(seq)
        valid_loss += loss
        #计算累计的错误数
        errors += diff
        if np.random.rand() < 0.1:
            #以0,1的概率记录一个输出的字符串
            show_out_p += outstring 
            show_out_t += targets
        #打印结果
        print(output[0][2].data.numpy())
        print('第{}轮，训练loss: {:.2f},校验loss：{:.2f},错误率：{:.2f}'.format(epoch,train_loss.data.numpy()/len(train_set),
                                                                                    valid_loss.data.numpy()/len(validset)
                                                                                    ,1.0*errors/len(validset)))
        print("the show output is: ",show_out_p)
        print("the show taget is: ",show_out_t)
        results.append([train_loss.data.numpy()/len(train_set),valid_loss/len(train_set),1.0*errors/len(validset)])
        
