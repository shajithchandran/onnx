import onnx
from onnx_tf.backend import prepare

import numpy as np

import sys
sys.path.append('../')
from common.CreditDataSet import CreditDataSet


dataset = CreditDataSet("../data/train.csv")
inputs = dataset.getrawinputs()
expected_output = dataset.getrawoutputs()

onnx_model = onnx.load("../models/cust_credit.onnx")
predicted_output = prepare(onnx_model).run(inputs)

#print (output)

diff = abs(expected_output - predicted_output)

correct = 0
total = 0
for i in diff[0]:
    total += 1
    if i <0.2:
        correct += 1

print("Correct: ", correct, " Total: " , total)


