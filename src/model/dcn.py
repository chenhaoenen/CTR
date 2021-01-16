# -*- coding: utf-8 -*-      
#--------------------------------------------
# Author:chenhao
# Date:2020-12-16 18:12
# Description:  
#--------------------------------------------
import torch
from torch import nn
from src.model.base import DNN, EmbeddingLayer, DenseFeatCatLayer

class DCN(nn.Module):
    """
    Deep & Cross Network for Ad Click Predictions
    """
    def __init__(self, sparse_feat_and_nums, cross_layers, dense_feat, embed_dim, deep_layers, sigmoid_out=False):
        super(DCN, self).__init__()
        x0_dim = len(sparse_feat_and_nums) * embed_dim + len(dense_feat)

        #embed
        self.embed = EmbeddingLayer(feat_and_nums=sparse_feat_and_nums, embed_dim=embed_dim)
        self.dense = DenseFeatCatLayer()

        #cross
        self.crossNet = CrossNet(x0_dim, cross_layers)

        #deep
        self.deep = DNN(input_dim=x0_dim, layers=deep_layers, act='relu', drop=0.2, bn=False)

        #output
        self.linear = nn.Linear(deep_layers[-1]+x0_dim, 1)
        self.drop = nn.Dropout()
        self.sigmoid_out = sigmoid_out
        if self.sigmoid_out:
            self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        sparse_feat = input['sparse']
        dense_feat = input['dense']

        #embed
        sparse_stack_embed_x = self.embed(sparse_feat, axis=2) #[B, sparse_num*embed_dim]
        dense_stack_x = self.dense(dense_feat) #[B, dense_num]

        #stack
        stack_x = torch.cat([sparse_stack_embed_x, dense_stack_x], dim=-1) #[B, sparse_num*embed_dim+dense_num]

        #cross
        cross_out = self.crossNet(stack_x) #[B, sparse_num*embed_dim+dense_num]
        #deep
        deep_out = self.deep(stack_x) #[B, deep_layers[-1]]

        #out
        cat_out = torch.cat([cross_out, deep_out], dim=-1) #[B, sparse_num*embed_dim+dense_num + deep_layers[-1]]
        linear_out = self.linear(cat_out)  # [B, 1]
        linear_out = self.drop(linear_out)
        out = linear_out.squeeze(1)  # [B]

        if self.sigmoid_out:
            out = self.sigmoid(out)

        return out

class CrossNet(nn.Module):
    def __init__(self, x0_dim, cross_layers):
        super(CrossNet, self).__init__()

        self.weights = nn.ParameterList(
            [
                nn.Parameter(nn.init.xavier_normal_(torch.empty(x0_dim, 1))) for _ in range(len(cross_layers))
            ]
        )
        self.bias = nn.ParameterList(
            [
                nn.Parameter(nn.init.xavier_normal_(torch.empty(x0_dim, 1))) for _ in range(len(cross_layers))
            ]
        )

    def forward(self, input):
        '''
        :param input: #[B, dim]
        :return: #[B, dim]
        '''
        x0 = input #[B, dim]

        x0_ = x0.unsqueeze(2)  #[B, dim, 1]
        xl = x0 #[B, dim]
        for w, b in zip(self.weights, self.bias):
            x0_xl = torch.matmul(x0_, xl.unsqueeze(1))  #[B, dim, 1] x  [B, 1, dim] = [B, dim, dim]
            x0_xl_w = torch.matmul(x0_xl, w) + b #[B, dim, dim] x [dim, 1] + [dim, 1] = [B, dim, 1]
            xl = x0_xl_w.squeeze(2) + xl #[B, dim]

        out = xl

        return out
