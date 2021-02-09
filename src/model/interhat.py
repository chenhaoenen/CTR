# -*- coding: utf-8 -*-      
#--------------------------------------------
# Author:chenhao
# Date:2021-02-04 10:47
# Description:  
#--------------------------------------------
import torch
from torch import nn
import torch.nn.functional as F
from src.model.transformer import TransformerBase
from src.model.base import DNN, SparseEmbeddingLayer, DenseEmbeddingLayer

class InterHAt(nn.Module):
    '''
    Interpretable Click-Through Rate Prediction through Hierarchical Attention
    '''
    def __init__(self,
                 sparse_feat_and_nums,
                 dense_feat,
                 embed_dim,
                 hidden_size,
                 num_attention_heads,
                 context_dim,
                 k_layers,
                 deep_layers,
                 sigmoid_out=False):
        super(InterHAt, self).__init__()
        #embed
        self.sparse_embed = SparseEmbeddingLayer(feat_and_nums=sparse_feat_and_nums, embed_dim=embed_dim)
        self.dense_embed = DenseEmbeddingLayer(dense_feat=dense_feat, embed_dim=embed_dim)

        #transformer
        self.transformer = TransformerBase(input_size=embed_dim,
                                           hidden_size=hidden_size,
                                           num_attention_heads=num_attention_heads)

        #Hierarchical Attention
        self.hier_att = HierarchicalAttention(embed_dim, context_dim, k_layers)

        #deep
        assert deep_layers[-1] == 1, "last hidden dim must be 1"
        self.deep_input_dim = embed_dim
        self.deep = DNN(input_dim= self.deep_input_dim, layers=deep_layers, act='tanh')

        #output
        self.sigmoid_out = sigmoid_out
        if self.sigmoid_out:
            self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        sparse_feat = input['sparse'] #[B]
        dense_feat = input['dense'] #[B]

        # embed
        sparse_embed_x = self.sparse_embed(sparse_feat, axis=1) #[B, sparse_num, embed_dim]
        dense_embed_x = self.dense_embed(dense_feat, axis=1) #[B, dense_num, embed_dim]
        embed_out = torch.cat([sparse_embed_x, dense_embed_x], axis=1) #[B, feat_num, embed_dim] feat_num=sparse_num+dense_num

        # transformer
        trans_out = self.transformer(embed_out) #[B, feat_num, embed_dim]

        # Hierarchical Attention
        hier_out = self.hier_att(trans_out) #[B, input_dim]

        #deep
        out = self.deep(hier_out).squeeze(1) #[B]

        if self.sigmoid_out:
            out = self.sigmoid(out)

        return out


class HierarchicalAttention(nn.Module):
    def __init__(self, input_dim, context_dim, k_layers):
        super(HierarchicalAttention, self).__init__()
        self.att_layers = nn.ModuleList(
            [
                AttentionalAgg(input_dim, context_dim) for _ in range(k_layers)
            ]
        )

        self.final_att = AttentionalAgg(input_dim, context_dim)

    def forward(self, x):
        '''
        :param x: [B, feat_num, input_dim]
        :return:
        '''
        U = []
        X1 = x #[B, feat_num, input_dim]
        Xi = X1
        for i, layer in enumerate(self.att_layers):
            ui = layer(Xi)  #[B, 1, input_dim]
            Xi = X1 * ui + Xi  #[B, feat_num, input_dim]
            U.append(ui)

        U = torch.cat(U, dim=1) #[B, k_layers, input_dim]

        out = self.final_att(U).squeeze(1) #[B, input_dim]

        return out


class AttentionalAgg(nn.Module):
    def __init__(self, input_dim, context_dim):
        super(AttentionalAgg, self).__init__()
        self.linear = nn.Linear(input_dim, context_dim)
        self.act = nn.ReLU()
        self.context = nn.Parameter(torch.ones(context_dim))

    def forward(self, x):
        '''
        :param x: [B, feat_num, input_dim]
        :return:
        '''
        linear_out = self.linear(x) #[B, feat_num, context_dim]
        act_out = self.act(linear_out) #[B, feat_num, context_dim]
        score_out = torch.sum(act_out * self.context, dim=-1, keepdim=False) #[B, feat_num]
        score_out = F.softmax(score_out, dim=-1).unsqueeze(1) #[B, 1, feat_num]

        out = torch.matmul(score_out, x)  #[B, 1, input_dim]

        return out
