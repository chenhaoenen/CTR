# -*- coding: utf-8 -*-      
#--------------------------------------------
# Author:chenhao
# Date:2021-01-01 18:06
# Description:  
#--------------------------------------------
import torch
from torch import nn
from src.model.base import SparseEmbeddingLayer, DenseFeatCatLayer

class DeepCrossing(nn.Module):
    '''
    Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features
    '''
    def __init__(self, sparse_feat_and_nums, dense_feat, embed_dim, res_layers, sigmoid_out=False):
        super(DeepCrossing, self).__init__()

        #shared embed
        self.embed = SparseEmbeddingLayer(feat_and_nums=sparse_feat_and_nums, embed_dim=embed_dim)
        self.dense = DenseFeatCatLayer()

        #Residual_Units
        self.stack_dim = len(sparse_feat_and_nums) * embed_dim + len(dense_feat)
        self.resnet = nn.Sequential(*[ResNet(self.stack_dim, layer) for layer in res_layers])

        #output
        self.linear = nn.Linear(self.stack_dim, 1)
        self.drop = nn.Dropout()
        self.sigmoid_out = sigmoid_out
        if self.sigmoid_out:
            self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        sparse_feat = input['sparse'] #[B]
        dense_feat = input['dense'] #[B]

        #embed
        sparse_embed_x = self.embed(sparse_feat, axis=2) #[B, sparse_num*embed_dim]
        dense_x = self.dense(dense_feat) #[B, dense_num]

        #resnet
        res_x = torch.cat([sparse_embed_x, dense_x], dim=-1) #[B, sparse_num*embed_dim+dense_num)]
        res_out = self.resnet(res_x) #[B, sparse_num*embed_dim+dense_num)]

        #out
        linear_out = self.linear(res_out) #[B, 1]
        linear_out = self.drop(linear_out)
        out = linear_out.squeeze(1) #[B]

        if self.sigmoid_out:
            out = self.sigmoid(out)

        return out

class ResNet(nn.Module):
    def __init__(self, input_dim, hidden_unit, drop=0.5):
        super(ResNet, self).__init__()

        #linear
        layers = []
        layers.append(nn.Linear(input_dim, hidden_unit))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(drop))
        layers.append(nn.Linear(hidden_unit, input_dim))

        self.layers = nn.Sequential(*layers)
        self.activation = nn.ReLU()

    def forward(self, input):
        '''
        :param input: [B, dim]
        :return: [B, dim]
        '''
        x = input
        out = self.layers(x)
        out = out + x
        out = self.activation(out)

        return out
