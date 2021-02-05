# -*- coding: utf-8 -*-      
#--------------------------------------------
# Author:chenhao
# Date:2021-02-02 09:05
# Description:  
#--------------------------------------------
import torch
from torch import nn
import torch.nn.functional as F
from src.model.base import DNN, SparseEmbeddingLayer, DenseEmbeddingLayer, DenseFeatCatLayer

class AFN(nn.Module):
    '''
    Adaptive Factorization Network: Learning Adaptive-Order Feature Interactions
    '''
    def __init__(self, sparse_feat_and_nums, dense_feat, embed_dim, deep_layers, num_log_neurons=64, sigmoid_out=False):
        super(AFN, self).__init__()

        #afn
        self.afn_embed = Embedding(sparse_feat_and_nums, dense_feat, embed_dim)
        self.afn_ltl = LogarithmicTransformerLayer(len(sparse_feat_and_nums)+len(dense_feat), num_log_neurons)
        self.drop_out1 = nn.Dropout()
        self.afn_deep_input_dim = num_log_neurons*embed_dim
        self.drop_out2 = nn.Dropout()
        self.afn_bn1 = nn.BatchNorm1d(num_log_neurons*embed_dim)
        assert deep_layers[-1] == 1, "last hidden dim must be 1"
        self.afn_deep = DNN(self.afn_deep_input_dim, layers=deep_layers, act='tanh')

        #deep
        self.deep_sparse_embed = SparseEmbeddingLayer(sparse_feat_and_nums,  embed_dim)
        self.deep_dense = DenseFeatCatLayer()
        self.deep_input_dim = len(sparse_feat_and_nums) * embed_dim + len(dense_feat)
        self.deep = DNN(input_dim= self.deep_input_dim, layers=deep_layers, act='tanh')

        #ensemble
        self.afn_w = nn.Parameter(torch.tensor([0.5]))
        self.deep_w = nn.Parameter(torch.tensor([0.5]))
        self.bias = nn.Parameter(torch.zeros(1))

        #output
        self.sigmoid_out = sigmoid_out
        if self.sigmoid_out:
            self.sigmoid = nn.Sigmoid()


    def forward(self, input):
        sparse_feat = input['sparse'] #[B]
        dense_feat = input['dense'] #[B]

        #afn
        afn_embed_out = self.afn_embed(sparse_feat, dense_feat) #[B, feat_num, embed_dim]
        # afn_embed_out = self.drop_out1(afn_embed_out) #[B, feat_num, embed_dim]

        afn_ltl_out = self.afn_ltl(afn_embed_out)  #[B, num_log_neurons*embed_dim]
        afn_ltl_out = self.afn_bn1(self.drop_out2(afn_ltl_out)) #[B, num_log_neurons*embed_dim]
        afn_deep_out = self.afn_deep(afn_ltl_out) #[B, 1]

        #deep
        deep_sparse_embed_x = self.deep_sparse_embed(sparse_feat, axis=2) #[B, sparse_num*embed_dim]
        deep_dense_x = self.deep_dense(dense_feat) #[B, dense_num]
        deep_x = torch.cat([deep_sparse_embed_x, deep_dense_x], axis=1) #[B, deep_input_dim]
        deep_out = self.deep(deep_x) #[B, 1]

        #ensemble
        print('afn_deep_out=', afn_deep_out)
        print('deep_out=', deep_out)
        out = (self.afn_w * afn_deep_out + self.deep_w * deep_out + self.bias).squeeze(1)

        if self.sigmoid_out:
            out = self.sigmoid(out)

        return out

class Embedding(nn.Module):
    def __init__(self, sparse_feat_and_nums, dense_feat, embed_dim):
        super(Embedding, self).__init__()
        self.sparse_embed = SparseEmbeddingLayer(feat_and_nums=sparse_feat_and_nums, embed_dim=embed_dim)
        self.dense_embed = DenseEmbeddingLayer(dense_feat=dense_feat, embed_dim=embed_dim)

    def init_weight(self): #keep all the values in the embeddings to be positive
        for key, layer in self.sparse_embed.embed:
            # layer.weight.data.uniform_(0.001, 0.01)
            layer.weight.data.uniform_(0.8, 0.9)
        for key, layer in self.dense_embed.embed:
            # layer.weight.data.uniform_(0.001, 0.01)
            layer.weight.data.uniform_(0.8, 0.9)

    def forward(self, sparse_feat, dense_feat):
        '''
        :param sparse_feat: [B] dict
        :param dense_feat: [B] dict
        :return:
        '''
        for key, value in sparse_feat.items():
            sparse_feat[key] = torch.clamp(value, min=0.001, max=1.)

        sparse_embed_x = self.sparse_embed(sparse_feat, axis=1) #[B, sparse_num, embed_dim]
        dense_embed_x = self.dense_embed(dense_feat, axis=1) #[B, dense_num, embed_dim]
        out = torch.cat([sparse_embed_x, dense_embed_x], axis=1) #[B, feat_num, embed_dim] feat_num=sparse_num+dense_num

        return out

class LogarithmicTransformerLayer(nn.Module):
    def __init__(self, feat_num, num_log_neurons):
        super(LogarithmicTransformerLayer, self).__init__()
        self.w = nn.Parameter(nn.init.xavier_normal_(torch.empty(num_log_neurons, feat_num)))
        self.b = nn.Parameter(torch.zeros(num_log_neurons))

    def forward(self, x):
        '''
        :param x: [B, feat_num, embed_dim]
        :return:
        '''
        batch = x.size(0)
        x = x.permute(0 ,2, 1) #[B, embed_dim, feat_num]
        log_out = torch.log(x) #[B, embed_dim, feat_num]
        print(log_out, 'log_out'*5)
        exp_x = F.linear(log_out, self.w, bias=self.b) #[B, embed_dim, num_log_neurons]
        exp_out = torch.exp(exp_x) #[B, embed_dim, num_log_neurons]
        out = exp_out.view(batch, -1) #[B, num_log_neurons*embed_dim]

        return out
