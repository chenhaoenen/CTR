# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2021/4/12 17:03 
# Description:  
# --------------------------------------------
import torch
a = torch.arange(18).view(2, 1, 3, 3)
print(a)

b = torch.arange(18).view(1, 3,2,3)
print(b)
b = torch.arange(6).view(1, 1, 2,3)
b = b.repeat(1,3,1,1)

print(b)

c = torch.matmul(b, a)
print(c)
print(c.size())

a = torch.tensor([5])
print(a.size())
b = torch.Tensor([1,2,3,4])
print(a*b)

from torch import nn
a = torch.empty(2,3)
print(a)
b = nn.Parameter(nn.init.xavier_uniform_(torch.empty(2,3)))
print(b)