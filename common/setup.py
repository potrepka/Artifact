import os
import torch
from torch.utils.data import DataLoader

def torch_version():
    return torch.__version__

def cuda_is_available():
    return torch.cuda.is_available()

def cuda_device_count():
    return torch.cuda.device_count()

def cpu():
    return torch.device('cpu')

def gpu(device=0):
    return torch.device('cuda:{}'.format(device))

def device():
    return gpu() if torch.cuda.is_available() else cpu()

def load(dataset, batch_size, shuffle=True):
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=os.cpu_count(),
                      pin_memory=True)
