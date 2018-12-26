import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import sys
sys.path.append('../')
import common.CreditDataSet as CD

epoch = 50


#torch.set_default_tensor_type('torch.DoubleTensor')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(24, 200)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

custdata=CD.CreditDataSet("../data/train.csv")
datalen = len(custdata)
split_pc = 0.2
split_index = int( split_pc * datalen)

indices = list(range(datalen))
np.random.shuffle(indices)

train_indices, test_indices = indices[split_index:], indices[:split_index]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
trainloader = torch.utils.data.DataLoader(custdata, batch_size=4, shuffle=False, sampler=train_sampler )
testset = torch.utils.data.DataLoader(custdata, batch_size=4, shuffle=False, sampler=test_sampler)


net = Net()
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

running_loss = 0.0

for loopidx in range(epoch):
    for i , data in enumerate(trainloader, 0):
        input,label = data

        optimizer.zero_grad()

        output = net(input) 
        #print ("label", label)
        #print ("output", output)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 20 == 19:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (loopidx + 1, i + 1, running_loss / 20))
            running_loss = 0.0

net.train(False)

correct = 0
total = 0
with torch.no_grad():
    for data in testset:
        input,label = data
        output = net(input)

        diff = abs(label - output)
        total += len(diff)
        result = diff < 0.5
        correct += torch.sum(result)

print ("Correct Prediction: ", correct.item(), "\nTotal Samples", total)

x = custdata.getrawinputs()
x=torch.from_numpy(x)

torch_output = torch.onnx._export(net, x, "../models/cust_credit.onnx", export_params=True)
