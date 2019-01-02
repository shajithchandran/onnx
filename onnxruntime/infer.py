import sys
sys.path.append('../')
from common.CreditDataSet import CreditDataSet
import onnxruntime as rt
import numpy

dataset = CreditDataSet("../data/train.csv")
inputs = dataset.getrawinputs()
expected_output = dataset.getrawoutputs()

rtmodel = rt.InferenceSession("../models/cust_credit.onnx")

#print(rtmodel.__dict__)

input_name = rtmodel.get_inputs()[0].name
label_name = rtmodel.get_outputs()[0].name

predicted_output = rtmodel.run([label_name], {input_name: inputs})

diff = abs(expected_output - predicted_output)

correct = 0
total = 0
for i in diff[0]:
    total += 1
    if i <0.2:
        correct += 1

print("Correct: ", correct, " Total: " , total)

