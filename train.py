
#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import torch
import resnet3d
import torch.nn as nn
from MakeData import myImageFloder

device = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')

# 
train_dataset = myImageFloder(root = "data_test")
#test_dataset = myImageFloder(root = 'data')

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size = 1,
                                           shuffle = True)
#test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size = 2,
#                                           shuffle = True)

resnet = resnet3d.Resnet3d(resnet3d.Basicblock)
net = resnet.to(device)

def train(epoch):
    print('epoch:',epoch)
    for batch_idx,(inputs,targets) in enumerate(train_loader):
        print(inputs)
        inputs = torch.from_numpy(inputs)
train(1)