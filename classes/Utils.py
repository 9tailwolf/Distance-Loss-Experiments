import torch
import json
import numpy as np
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:1'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device

def str2loss(v):
    if v.lower() in ('mse','ms','m'):
        return 'MSE'
    elif v.lower() in ('mae','ma'):
        return 'MAE'
    elif v.lower() in ('ce','crossentropy','c'):
        return 'CE'
    elif v.lower() in ('dims','d'):
        return 'DiMS'
    elif v.lower() in ('adims'):
        return 'ADiMS'
    elif v.lower() in ('dima'):
        return 'DiMA'
    elif v.lower() in ('adima'):
        return 'ADiMA'

    else:
        raise 'Loss Error'

def get_data_info(name):
    if name=='DIA':
        d = {
            'label':7,
            'feature':9,
            'Y':'y',
            'index':2,
            'minus':0,
        }
    elif name=='MPG':
        d = {
            'label':6, # change
            'feature':7,
            'Y':"'class'",
            'index':2,
            'minus':0
        }
    elif name=='BIKE':
        d = {
            'label':10, # change
            'feature':14,
            'Y':"cnt",
            'index':2,
            'minus':0
        }
    else:
        raise 'Name Error'

    return d