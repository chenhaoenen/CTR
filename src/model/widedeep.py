# -*- coding: utf-8 -*-      
#--------------------------------------------
# Author:chenhao
# Date:2020-12-12 17:47
# Description:  
#--------------------------------------------
import torch
from torch import nn
from src.model.base import Linear, DNN, SparseEmbeddingLayer, DenseFeatCatLayer

class WideDeep(nn.Module):
    '''
    Wide & Deep Learning for Recommender Systems
    '''
    def __init__(self, sparse_feat_and_nums, cross_feat_and_nums, dense_feat, embed_dim, deep_layers, sigmoid_out=False):
        super(WideDeep, self).__init__()
        sparse_cross_feat_and_nums = sparse_feat_and_nums + cross_feat_and_nums

        #wide
        self.wide = Linear(sparse_feat_and_nums=sparse_cross_feat_and_nums, dense_feat=dense_feat)

        #deep
        self.embed = SparseEmbeddingLayer(feat_and_nums=sparse_feat_and_nums, embed_dim=embed_dim)
        self.dense = DenseFeatCatLayer()
        assert deep_layers[-1] == 1, "last hidden dim must be 1"
        deep_dim = len(sparse_cross_feat_and_nums) * embed_dim + len(dense_feat)
        self.deep = DNN(input_dim=deep_dim, layers=deep_layers, act='relu', drop=0.2, bn=False)

        #output
        self.sigmoid_out = sigmoid_out
        if self.sigmoid_out:
            self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        sparse_feat = input['sparse']
        dense_feat = input['dense']
        cross_feat = input['cross']

        sparse_cross_feat = dict(list(sparse_feat.items())+list(cross_feat.items()))

        #wide
        wide_out = self.wide(sparse_cross_feat, dense_feat) #[B]

        #deep
        deep_sparse_cross_x = self.embed(sparse_cross_feat, axis=2) #[B, sparse_cross_num*embed_dim]
        deep_dense_x = self.dense(dense_feat) #[B, dense_num]
        deep_x = torch.cat([deep_sparse_cross_x, deep_dense_x], dim=-1) #[B, sparse_cross_num*embed_dim+dense_num]
        deep_out = self.deep(deep_x).squeeze(1) #[B]

        #out
        out = wide_out + deep_out #[B]
        if self.sigmoid_out:
            out = self.sigmoid(out)

        return out
