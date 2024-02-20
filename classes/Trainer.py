'''
Requirement Libraries
'''
import time
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from transformers import get_linear_schedule_with_warmup
from torch.nn.utils import clip_grad_norm_
from loguru import logger

'''
Local Classes
'''
from classes.Utils import get_device
from classes.Loss import MSELoss, CrossEntropyLoss, L1Loss, DiMSLoss, DiMALoss, ADiMSLoss, ADiMALoss

class Trainer:
    def __init__(self,train_data,test_data,model,lr,eps,epochs,loss,text=False):
        self.text = text
        self.device = get_device()
        self.model = model
        self.dataloader_train = train_data
        self.dataloader_test = test_data
        self.epochs = epochs
        self.performance = (0,0)

        if loss=='MSE':
            self.loss_fn = MSELoss()
        elif loss=='DiMS':
            self.loss_fn = DiMSLoss()
        elif loss=='CE':
            self.loss_fn = CrossEntropyLoss()
        elif loss=='MAE':
            self.loss_fn = L1Loss()
        elif loss=='ADiMS':
            self.loss_fn = ADiMSLoss()
        elif loss=='DiMA':
            self.loss_fn = DiMALoss()
        elif loss=='ADiMA':
            self.loss_fn = ADiMALoss()
    
        self.optimizer = AdamW([{'params': self.model.parameters()}, {'params': self.loss_fn.parameters(), 'lr': lr}], lr=lr)
        if False:
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps = 0, num_training_steps = epochs * len(self.dataloader_train))
        else:
            self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)

        
    def train(self):
        self.model.zero_grad()
        self.model.train()
        for epoch in tqdm((range(self.epochs))):
            total_loss = 0
            self.current_time = time.time()
            for d in self.dataloader_train:
                if self.text:
                    x_ids = d['input_ids'].to(self.device)
                    x_mask =  d['attention_mask'].to(self.device)
                    Y = d['labels'].to(self.device)
                    outputs = self.model(x_ids,attention_mask = x_mask)
                else:
                    X,Y = d['X'].to(self.device),d['Y'].to(self.device)
                    outputs = self.model(X)
                loss = self.loss_fn(outputs, Y)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            self.eval(epoch, total_loss)
            

    def eval(self,e,l):
        correct = 0
        count = 0
        self.model.eval()
        for d in self.dataloader_test:
            if self.text:
                x_ids = d['input_ids'].to(self.device)
                x_mask =  d['attention_mask'].to(self.device)
                Y = d['labels'].to('cpu').numpy()
                
            else:
                X = d['X'].to(self.device)
                Y = d['Y']

            with torch.no_grad():
                if self.text:
                    outputs = self.model(x_ids,attention_mask = x_mask)
                else:
                    outputs = self.model(X)
            outputs = outputs.detach().cpu().numpy()
            correct += self.eval_accuracy(outputs, Y)
            count += len(outputs)
        accuracy = correct / count
        if accuracy > self.performance[0]:
            self.performance = (accuracy,l)
        self.model.train()
        return round(accuracy,5)
    
    def eval_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = np.array(np.argmax(labels, axis=1)).flatten()
        res =  np.sum(pred_flat == labels_flat)
        return res

    def get_performance(self):
        logger.info(f"Accuracy : {self.performance[0]}, Loss : {self.performance[1]}")
        return self.performance