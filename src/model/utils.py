# -*- coding: utf-8 -*-      
#--------------------------------------------
# Author:chenhao
# Date:2020-12-13 12:05
# Description:  
#--------------------------------------------
import torch
import torch.nn as nn

def weight_init(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)
    if isinstance(layer, nn.Embedding):
        # layer.weight.data.normal_(0, 0.01)
        layer.weight.data.uniform_(0.001, 0.01)