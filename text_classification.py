import argparse
import torch
import json
from classes.Datasets import DatasetforTextData
from classes.Dataloaders import DataLoader, data_spliter
from classes.Models import BERT_NN
from classes.Trainer import Trainer
from classes.Utils import get_device, str2loss

def get_argparse():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--loss', default='dims', type=str2loss, help="Enter the loss functions : mse, ce, mae, dims, adims, dima, adima.")
    parser.add_argument('--data', default='ARD', type=str, help='Type the name of data')
    parser.add_argument('--seed', default=0, type=int, help='Type any intenger without 0 when you set identical condition for training, model. Or type 0 for random state.')
    return parser.parse_args()

def main(args=None):
    if args.seed:
        torch.manual_seed(seed=args.seed)
    device = get_device()
    dataset = DatasetforTextData(name=args.data)
    train_data, test_data = data_spliter(dataset, ratio=0.7)
    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True) # You can edt batch_size(intenger).
    test_dataloader = DataLoader(test_data, batch_size=len(test_data),shuffle=False)
    model = BERT_NN(linear=256,dropout=0.3,labels=5).to(device=device) # You can edit linear(intenger), dropout(float f, 0<= f <=1).
    trainer = Trainer(train_data=train_dataloader, test_data=test_dataloader, model=model,lr=1e-5,eps=1e-8,epochs=10, loss=args.loss,text=True) # You can edit lr(learning rate), eps(epsilons that prevent gradient vanishing problem), and epochs(intenger)
    trainer.train()
    performance = trainer.get_performance()
    

if __name__=='__main__':
    args = get_argparse()
    main(args)