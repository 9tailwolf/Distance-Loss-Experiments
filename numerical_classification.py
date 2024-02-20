import argparse
import torch
import json
from classes.Datasets import DatasetforData
from classes.Dataloaders import DataLoader, data_spliter
from classes.Models import NN
from classes.Trainer import Trainer
from classes.Utils import get_device, str2loss, get_data_info

def get_argparse():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--loss', default='dims', type=str2loss, help="Enter the loss functions : mse, ce, mae, dims, adims, dima, adima.")
    parser.add_argument('--data', default='MPG', type=str, help='Type the name of data')
    parser.add_argument('--seed', default=0, type=int, help='Type any intenger without 0 when you set identical condition for training, model. Or type 0 for random state.')
    return parser.parse_args()

def main(args=None):
    if args.seed:
        torch.manual_seed(seed=args.seed)
    device = get_device()
    data_info = get_data_info(args.data)
    dataset = DatasetforData(name=args.data)
    train_data, test_data = data_spliter(dataset, ratio=0.7)
    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True) # You can edt batch_size(intenger).
    test_dataloader = DataLoader(test_data, batch_size=len(test_data),shuffle=False)
    model = NN(inp=data_info['feature'],linear=40,dropout=0.3,labels=data_info['label']).to(device=device) # You can edit linear(intenger), dropout(float f, 0<= f <=1).
    trainer = Trainer(train_data=train_dataloader, test_data=test_dataloader, model=model,lr=1e-3,eps=1e-8,epochs=100, loss=args.loss) # You can edit lr(learning rate), eps(epsilons that prevent gradient vanishing problem), and epochs(intenger)
    trainer.train()
    performance = trainer.get_performance()

if __name__=='__main__':
    args = get_argparse()
    main(args)