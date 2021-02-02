# -*- coding: utf-8 -*-      
#--------------------------------------------
# Author:chenhao
# Date:2020-12-15 09:25
# Description:  
#--------------------------------------------
import torch
from torch import nn
from src.model.fm import FMInteraction
from src.model.base import DNN, Linear, SparseEmbeddingLayer, DenseFeatCatLayer

class DeepFM(nn.Module):
    '''
    DeepFM: A Factorization-Machine based Neural Network for CTR Prediction
    '''
    def __init__(self, sparse_feat_and_nums, dense_feat, embed_dim, deep_layers, sigmoid_out=False):
        super(DeepFM, self).__init__()

        #shared embed
        self.embed = SparseEmbeddingLayer(feat_and_nums=sparse_feat_and_nums, embed_dim=embed_dim)
        self.dense = DenseFeatCatLayer()

        #fm interaction
        self.linear = Linear(sparse_feat_and_nums=sparse_feat_and_nums, dense_feat=dense_feat)
        self.fm_interaction = FMInteraction()

        #deep
        assert deep_layers[-1] == 1, "last hidden dim must be 1"
        self.deep_input_dim = len(sparse_feat_and_nums) * embed_dim + len(dense_feat)
        self.deep = DNN(input_dim= self.deep_input_dim, layers=deep_layers, act='tanh')

        #output
        self.sigmoid_out = sigmoid_out
        if self.sigmoid_out:
            self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        sparse_feat = input['sparse'] #[B]
        dense_feat = input['dense'] #[B]

        #embed
        # sparse_fm_embed_x [B, sparse_num, embed_dim]
        # sparse_deep_embed_x [B, sparse_num*embed_dim]
        sparse_fm_embed_x, sparse_deep_embed_x = self.embed(sparse_feat, axis=-1)
        dense_deep_x = self.dense(dense_feat) #[B, dense_num]

        #fm linear
        linear_out = self.linear(sparse_feat, dense_feat) #[B]

        #fm interaction
        inter_out = self.fm_interaction(sparse_fm_embed_x)  # [B]

        #deep
        deep_embed_x = torch.cat([sparse_deep_embed_x, dense_deep_x], dim=-1) #[B, sparse_num*embed_dim+dense_num)]
        deep_out = self.deep(deep_embed_x).squeeze(1) #[B]

        #out
        out = linear_out + inter_out + deep_out #[B]

        if self.sigmoid_out:
            out = self.sigmoid(out)

        return out
