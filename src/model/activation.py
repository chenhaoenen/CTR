# -*- coding: utf-8 -*-      
#--------------------------------------------
# Author:chenhao
# Date:2020-12-22 16:34
# Description:  
#--------------------------------------------
import torch
from torch import nn

def activation(act:str):
    if act == 'sigmoid':
        return nn.Sigmoid()
    if act == 'relu':
        return nn.ReLU()
    if act == 'tanh':
        return nn.Tanh()
    if act == 'prelu':
        return nn.PReLU()
    raise ValueError("act must be sigmoid, relu, tanh")

class Dice(nn.Module):
    def __init__(self):
        super(Dice, self).__init__()

    def forward(self, input):
        pass
