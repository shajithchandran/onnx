import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import requests
sys.path.append('../')
import common.CreditDataSet as CD

custdata=CD.CreditDataSet("../data/train.csv")
testset = torch.utils.data.DataLoader(custdata, batch_size=1, shuffle=False)

def predict_using_rest_api(features):
    url = 'http://127.0.0.1:5000/predict'
    headers = {'Content-Type': 'application/json', 'Accept':'application/json'}
    data = [ { 'features': features.tolist(), }]
    prediction = requests.post(url, json=data, headers=headers).json()
    prediction = np.squeeze(prediction[0]['prediction'])
    prediction = torch.FloatTensor(prediction)
    return prediction

correct = 0
total = 0
for data in testset:
        input,label = data

        prediction = predict_using_rest_api(input)
        #printdiff = abs(label - prediction)
        #print (prediction, label)
        diff = abs(label - prediction)
        total += len(diff)
        result = diff < 0.5
        correct += torch.sum(result)

print ("Correct Prediction: ", correct.item(), "\nTotal Samples", total)


        
