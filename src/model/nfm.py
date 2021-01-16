# -*- coding: utf-8 -*-      
#--------------------------------------------
# Author:chenhao
# Date:2020-12-22 17:15
# Description:  
#--------------------------------------------
import torch
from torch import nn
from src.model.base import Linear, DNN, EmbeddingLayer, DenseFeatCatLayer

class NFM(nn.Module):
    '''
    Neural Factorization Machines for Sparse Predictive Analytics∗
    '''
    def __init__(self, sparse_feat_and_nums, dense_feat, embed_dim, deep_layers, sigmoid_out=False):
        super(NFM, self).__init__()

        #linear
        self.linear = Linear(sparse_feat_and_nums=sparse_feat_and_nums, dense_feat=dense_feat)

        #embed
        self.embed = EmbeddingLayer(feat_and_nums=sparse_feat_and_nums, embed_dim=embed_dim)
        self.dense = DenseFeatCatLayer()

        #Bi-Interaction pooling
        self.interaction = BiInteractionPooling()
        self.bi_drop = nn.Dropout()

        #deep
        assert deep_layers[-1] == 1, "last hidden dim must be 1"
        self.deep_input_dim = embed_dim + len(dense_feat)
        self.deep = DNN(input_dim= self.deep_input_dim, layers=deep_layers, act='tanh')

        #output
        self.sigmoid_out = sigmoid_out
        if self.sigmoid_out:
            self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        sparse_feat = input['sparse'] #[B]
        dense_feat = input['dense'] #[B]

        #embed
        bi_x = self.embed(sparse_feat, axis=1)  # [B, sparse_num, embed_dim]
        dense_deep_x = self.dense(dense_feat) #[B, dense_num]

        #linear
        linear_out = self.linear(sparse_feat, dense_feat) #[B]

        # Bi-Interaction pooling
        bi_out = self.interaction(bi_x) #[B, 1, embed_dim]
        bi_out = self.bi_drop(bi_out) #[B, 1, embed_dim]

        #deep
        bi_deep_x = bi_out.squeeze(1) #[B, embed_dim]
        deep_x = torch.cat([bi_deep_x, dense_deep_x], dim=-1) #[B, embed_dim+dense_num]
        deep_out = self.deep(deep_x).squeeze(1) #[B]

        #out
        out = linear_out + deep_out #[B]
        if self.sigmoid_out:
            out = self.sigmoid(out)

        return out

class BiInteractionPooling(nn.Module):
    def __init__(self):
        super(BiInteractionPooling, self).__init__()

    def forward(self, input):
        x = input #[B, num, dim]
        square_of_sum = torch.pow(torch.sum(x, dim=1, keepdim=True), exponent=2) #[B, 1, dim]
        sum_of_square = torch.sum(x * x, dim=1, keepdim=True) #[B, 1, dim]
        cross = square_of_sum - sum_of_square #[B, 1, dim]
        out = 0.5 * cross #[B, 1, dim]

        return out


