# -*- coding: utf-8 -*-      
#--------------------------------------------
# Author:chenhao
# Date:2020-12-23 09:38
# Description:  
#--------------------------------------------
import torch
from torch import nn
from src.model.base import DNN, Linear, SparseEmbeddingLayer, DenseFeatCatLayer
from src.model.activation import activation

class XDeepFM(nn.Module):
    '''
    xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems
    '''
    def __init__(self, sparse_feat_and_nums,
                 dense_feat,
                 embed_dim,
                 deep_layers,
                 cin_layers=[128, 128],
                 sigmoid_out=False):
        super(XDeepFM, self).__init__()

        #shared embed
        self.embed = SparseEmbeddingLayer(feat_and_nums=sparse_feat_and_nums, embed_dim=embed_dim)
        self.dense = DenseFeatCatLayer()

        #linear
        self.linear = Linear(sparse_feat_and_nums, dense_feat, bias=False)

        #cin
        self.cin = CIN(h_layers=cin_layers, sparse_feat_num=len(sparse_feat_and_nums))
        cin_cores = sum([num-num//2 for num in cin_layers])
        self.cin_w = nn.Linear(cin_cores, 1)

        #deep
        self.deep_input_dim = len(sparse_feat_and_nums) * embed_dim + len(dense_feat)
        self.deep = DNN(input_dim= self.deep_input_dim, layers=deep_layers, act='tanh')
        self.deep_w = nn.Linear(deep_layers[-1], 1)

        #bias
        self.b = nn.Parameter(torch.zeros(1))

        #output
        self.sigmoid_out = sigmoid_out
        if self.sigmoid_out:
            self.sigmoid = nn.Sigmoid()


    def forward(self, input):
        sparse_feat = input['sparse'] #[B]
        dense_feat = input['dense'] #[B]

        #embed
        # sparse_cin_embed_x [B, sparse_num, embed_dim]
        # sparse_deep_embed_x [B, sparse_num*embed_dim]
        sparse_cin_embed_x, sparse_deep_embed_x = self.embed(sparse_feat, axis=-1)
        dense_deep_x = self.dense(dense_feat) #[B, dense_num]

        #linear
        linear_out = self.linear(sparse_feat, dense_feat) #[B]

        #cin
        cin_out = self.cin(sparse_cin_embed_x)  # [B, l]
        cin_out = self.cin_w(cin_out).squeeze(1) #[B]

        #deep
        deep_embed_x = torch.cat([sparse_deep_embed_x, dense_deep_x], dim=-1) #[B, sparse_num*embed_dim+dense_num)]

        deep_out = self.deep(deep_embed_x) #[B, deep_layers[-1]]
        deep_out = self.deep_w(deep_out).squeeze(1) #[B]

        #out
        out = linear_out + cin_out + deep_out + self.b #[B]

        if self.sigmoid_out:
            out = self.sigmoid(out)

        return out

class CIN(nn.Module):
    def __init__(self, h_layers, sparse_feat_num, direct=False, act='sigmoid'):
        super(CIN, self).__init__()
        self.h_layers = h_layers
        self.direct = direct
        self.m = sparse_feat_num

        #conv
        self.convs = nn.ModuleList()
        h_k = self.m
        for h_k_next in self.h_layers:
            self.convs.append(nn.Conv1d(self.m*h_k,  h_k_next, 1))
            h_k = h_k_next//2

        self.activate = activation(act)

    def forward(self, input: torch.Tensor()):
        '''
        :param input: #[B, m, dim]
        :return:
        '''
        result = []
        batch, m, dim = input.size()
        x_0 = input #[B, m, dim]
        x_k_pre = x_0 #[B, h_k_pre, dim]
        h_k_pre = m

        for conv, h_k_next in zip(self.convs, self.h_layers):
            hp = torch.einsum('bhd,bmd->bhmd', x_k_pre, x_0) #hp: Hadamard product [B, h_k_pre, m, dim]
            conv_x = hp.reshape(batch, h_k_pre*m, dim) #[B, h_k_pre*m, dim]
            x_k_next = conv(conv_x) #[B, h_k_next, dim]
            x_k_next = self.activate(x_k_next) #[B, h_k_next, dim]

            if self.direct:
                x_k_pre = x_k_next
                h_k_pre = h_k_next
                result.append(x_k_next)
            else:
                x_k_next_hidden, x_k_next_direct = torch.split(x_k_next, h_k_next//2, dim=1)
                x_k_pre = x_k_next_hidden #[B, h_k_next//2, dim]
                h_k_pre = h_k_next//2
                result.append(x_k_next_direct) #[B, h_k_next - h_k_next//2, dim]

        out = torch.cat(result, dim=1) #[B, l, dim]  l = \sum h_k_next_i-h_k_next_i//2
        out = torch.sum(out, dim=-1, keepdim=False) #[B, l]

        return out
