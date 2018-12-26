import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from process_data import read_customer_history


class CreditDataSet(Dataset):

    def __init__(self, csv_file):
        #(features, Y, le, mm_scalar) = read_customer_history(csv_file)
        #self.customer_records = features
        #self.Y = Y
        #self.le = le
        #self.mm_scalar = mm_scalar

        #For now, lets get the transformed data directly.
        self.customer_records = pd.read_csv(csv_file)
        self.recordlen = len(self.customer_records.iloc[0])

    def __len__(self):
        return len(self.customer_records)

    def __getitem__(self, idx):
        t = (self.customer_records.iloc[idx,].values[:self.recordlen-1], self.customer_records.iloc[idx,].values[self.recordlen-1:])
        return t

    def getrawinputs(self):
        return self.customer_records.loc[:, self.customer_records.columns != '25'].values
