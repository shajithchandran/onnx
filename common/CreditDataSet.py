import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from common.process_data import read_customer_history
import numpy as np


class CreditDataSet(Dataset):

    def __init__(self, csv_file):
        #(features, Y, le, mm_scalar) = read_customer_history(csv_file)
        #self.customer_records = features
        #self.Y = Y
        #self.le = le
        #self.mm_scalar = mm_scalar

        #For now, lets get the transformed data directly.
        self.customer_records = pd.read_csv(csv_file)
        self.customer_records = self.customer_records.apply(np.float32).values #Caffe2 onnx doesn't handle double
        self.inputs = self.customer_records[:,:-1]
        self.labels = self.customer_records[:,-1:]

    def __len__(self):
        return len(self.customer_records)

    def __getitem__(self, idx):
        t = (self.inputs[idx], self.labels[idx])
        return t

    def getrawinputs(self):
        return self.inputs

    def getrawoutputs(self):
        return self.labels
