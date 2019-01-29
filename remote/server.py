import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('../')
from flask import request,jsonify,Flask

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

net = Net()
net.load_state_dict(torch.load("../models/model.pt"))
net.eval()

net.train(False)

app = Flask(__name__)

def model_predict(testdata):
    #print("model_predict", testdata)
    with torch.no_grad():
        output = net(testdata)
        return output

@app.route("/predict", methods=['POST'])
def predict():
        params = request.get_json(silent=True)
        parameters = params[0]
        testdata = np.array(parameters['features'])
        testdata = torch.FloatTensor(testdata)
        prediction = model_predict(testdata)
        return jsonify([{'prediction':prediction.tolist()}])

def main():
    app.run(threaded=False,debug=True)

if __name__ == '__main__':
    main()

