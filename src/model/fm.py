# -*- coding: utf-8 -*-      
#--------------------------------------------
# Author:chenhao
# Date:2020-12-13 12:04
# Description:  
#--------------------------------------------
import torch
from torch import nn
from src.model.base import Linear


class FM(nn.Module):
    def __init__(self, sparse_feat_and_nums, embed_dim, sigmoid_out=False):
        super(FM, self).__init__()

        #linear
        self.linear = Linear(sparse_feat_and_nums)

        #interaction
        self.embed = nn.ModuleDict(
            {
                feat: nn.Embedding(num, embed_dim)
                for feat, num in sparse_feat_and_nums
            }
        )
        self.interaction = FMInteraction()
        self.sigmoid_out = sigmoid_out
        if self.sigmoid_out:
            self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        sparse_feat = input['sparse']

        #linear
        linear_out = self.linear(sparse_feat)

        #interaction
        x = [
            self.embed[feat](x.long())
            for feat, x in list(sparse_feat.items())
        ]
        x = torch.cat(x, dim=1)
        inter_out = self.interaction(x)

        #out
        out = linear_out + inter_out
        if self.sigmoid_out:
            out = self.sigmoid(out)

        return out

class FMInteraction(nn.Module):
    def __init__(self):
        super(FMInteraction, self).__init__()

    def forward(self, input):
        x = input  #[B, num, dim]
        square_of_sum = torch.pow(torch.sum(x, dim=1, keepdim=True), exponent=2) #[B, 1, dim]
        sum_of_square = torch.sum(x * x, dim=1, keepdim=True) #[B, 1, dim]
        cross = square_of_sum - sum_of_square #[B, 1, dim]

        out = 0.5 * torch.sum(cross, dim=2, keepdim=False).squeeze(1) #[B]

        return out #[B]

