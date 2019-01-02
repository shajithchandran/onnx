import onnx
import onnx_caffe2.backend
import caffe2.python.onnx.backend
import torch
from caffe2.python import core, workspace
import numpy as np

import sys
sys.path.append('../')
from common.CreditDataSet import CreditDataSet

dataset = CreditDataSet("../data/train.csv")
inputs = dataset.getrawinputs()
expected_output = dataset.getrawoutputs()

lmodel = onnx.load("../models/cust_credit.onnx")

predicted_output = caffe2.python.onnx.backend.run_model(lmodel, inputs)

diff = abs(expected_output - predicted_output)

correct = 0
total = 0
for i in diff[0]:
    total += 1
    if i <0.2:
        correct += 1

print("Correct: ", correct, " Total: " , total)
