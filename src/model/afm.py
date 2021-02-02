# -*- coding: utf-8 -*-      
#--------------------------------------------
# Author:chenhao
# Date:2021-01-02 13:57
# Description:  
#--------------------------------------------
import torch
from torch import nn
import torch.nn.functional as F
from src.model.base import Linear, SparseEmbeddingLayer

class AFM(nn.Module):
    '''
    Attentional Factorization Machines:Learning the Weight of Feature Interactions via Attention Networks∗
    '''
    def __init__(self, sparse_feat_and_nums, dense_feat, embed_dim,attention_factor, sigmoid_out=False):
        super(AFM, self).__init__()

        #embed
        self.embed = SparseEmbeddingLayer(feat_and_nums=sparse_feat_and_nums, embed_dim=embed_dim)
        # linear
        self.linear = Linear(sparse_feat_and_nums=sparse_feat_and_nums, dense_feat=dense_feat)

        #fm attention
        self.fmAttention = FMAttention(embed_dim, attention_factor)

        #output
        self.sigmoid_out = sigmoid_out
        if self.sigmoid_out:
            self.sigmoid = nn.Sigmoid()


    def forward(self, input):
        sparse_feat = input['sparse'] #[B]
        dense_feat = input['dense'] #[B]

        #embed
        sparse_fm_embed_x = self.embed(sparse_feat, axis=1)  # [B, sparse_num, embed_dim]

        #linear
        linear_out = self.linear(sparse_feat, dense_feat) #[B]

        #fm attention
        fm_att_out = self.fmAttention(sparse_fm_embed_x) #[B]

        #out
        out = linear_out + fm_att_out #[B]

        if self.sigmoid_out:
            out = self.sigmoid(out)

        return out

class FMAttention(nn.Module):
    def __init__(self, embed_dim, attention_factor):
        super(FMAttention, self).__init__()

        attention = [
            nn.Linear(embed_dim, attention_factor),  # bias=True
            nn.ReLU(),
            nn.Linear(attention_factor, 1, bias=False)
        ]
        self.att = nn.Sequential(*attention)

        self.p = nn.Linear(embed_dim, 1, bias=False)


    def forward(self, input):
        '''
        :param input: #[B, num, dim]
        :return:
        '''
        x = input

        # inner product
        num_feat = input.size(1)
        inner_x1, inner_x2 = [], []
        for i in range(num_feat):
            for j in range(i+1, num_feat):
                inner_x1.append(x[:,i,:].unsqueeze(1)) #[B, 1, dim]
                inner_x2.append(x[:,j,:].unsqueeze(1)) #[B, 1, dim]
        inner_x1 = torch.cat(inner_x1, dim=1) #[B, m, dim]  m=num(num-1)/2
        inner_x2 = torch.cat(inner_x2, dim=1) #[B, m, dim]

        att_x = inner_x1 * inner_x2 #[B, m, dim]

        #attention
        att_wx = self.att(att_x) #[B, m, 1]
        att_score = F.softmax(att_wx, dim=1) #[B, m, 1]
        att_out = att_x * att_score #[B, m, dim]
        att_out = torch.sum(att_out, dim=1, keepdim=False) #[B, dim]
        out = self.p(att_out).squeeze(1) #[B]

        return out #[B]
