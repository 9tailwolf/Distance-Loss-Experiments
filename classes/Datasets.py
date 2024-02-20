'''
Requirement Libraries
'''
import pandas as pd
import torch
import re
from transformers import AutoTokenizer
from sklearn.datasets import make_regression
from torch.utils.data import Dataset
from classes.Utils import get_data_info

class DatasetforVirtualData(Dataset):
    def __init__(self,size,label,seed):
        self.label = label
        self.preX,self.preY = make_regression(n_samples=size,random_state=seed)
        self.data = [self.preX[i] + [self.preY[i]] for i in range(size)]
        self.make_labels(sep=label)
        self.preprocessing()
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return {
             'X':self.X[index],
             'Y':self.Y[index]
        }
    
    def preprocessing(self):
        self.X = [torch.Tensor(i[:-1]) for i in self.data]
        self.Y = [torch.Tensor(self.get_classification_preds(int(i[-1]))) for i in self.data]

    def make_labels(self,sep):
        self.data.sort(key=lambda x:x[-1])
        sep_point = (max(self.preY)*1.01 - min(self.preY)) / sep
        min_vaule = min(self.preY)
        keep = []
        for i in range(len(self.data)):
            label = int((self.data[i][-1] - min_vaule) // sep_point)
            self.data[i][-1] = label
            keep.append(label)

    def get_classification_preds(self, i):
        temp = [0 for _ in range(self.label)]
        temp[i] = 1
        return temp

class DatasetforData(Dataset):
    def __init__(self,name):
        self.data = pd.read_csv('./Dataset/' + name + '.csv',encoding='cp949',dtype='float')
        self.info = get_data_info(name)
        self.preprocessing()
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return {
             'X':self.X[index],
             'Y':self.Y[index]
        }
    
    def preprocessing(self):
        self.label = self.info['label']
        self.X = [torch.Tensor(self.data.iloc[i][self.info['index']:-1]) for i in range(len(self.data))]
        self.Y = [torch.Tensor(self.get_classification_preds(int(self.data[self.info['Y']][i]) - int(self.info['minus']))) for i in range(len(self.data))]

    def get_classification_preds(self, i):
        temp = [0 for _ in range(self.label)]
        try:
            temp[i] = 1
        except:
            print(i)
        return temp
    
class DatasetforTextData(Dataset):
    def __init__(self,name):
        self.data = pd.read_csv('./Dataset/' + name + '.csv',encoding='utf-8')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.preprocessing()
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return {
            'input_ids': torch.LongTensor(self.X[index]['input_ids']),
            'attention_mask': torch.LongTensor(self.X[index]['attention_mask']),
            'labels': torch.Tensor(self.Y[index])
        }
    
    def preprocessing(self):
        length = 100
        temp = ['[cls]' + self.data['text'][i][:length-2] + '[sep]' for i in range(len(self.data))]
        self.X = [self.tokenizer(temp[i], padding = "max_length", max_length=length) for i in range(len(temp))]
        self.Y = [self.get_classification_preds(int(self.data['y'][i]) - 1) for i in range(len(self.data))]

    def get_classification_preds(self, i):
        temp = [0 for _ in range(5)]
        try:
            temp[i] = 1
        except:
            print(i)
        return temp