import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('../')
import common.CreditDataSet as CD



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
testset = torch.utils.data.DataLoader(custdata, batch_size=1, shuffle=False)

net = Net()
net.load_state_dict(torch.load("../models/model.pt"))
net.eval()

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
