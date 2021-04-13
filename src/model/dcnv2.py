# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2021/4/12 15:30 
# Description:  
# --------------------------------------------
import torch
from torch import nn
from src.model.base import DNN, SparseEmbeddingLayer, DenseFeatCatLayer

class DCNV2(nn.Module):
    '''
    DCN V2: Improved Deep & Cross Network and Practical Lessons forWeb-scale Learning to Rank Systems
    '''
    def __init__(self, sparse_feat_and_nums, cross_layers, dense_feat, embed_dim, deep_layers, k_Experts, r_dim, sigmoid_out=False):
        super(DCNV2, self).__init__()
        x0_dim = len(sparse_feat_and_nums) * embed_dim + len(dense_feat)

        #embed
        self.embed = SparseEmbeddingLayer(feat_and_nums=sparse_feat_and_nums, embed_dim=embed_dim)
        self.dense = DenseFeatCatLayer()

        #cross
        self.crossNet = MixtureCrossNet(x0_dim, cross_layers, k_Experts, r_dim)

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

class MixtureCrossNet(nn.Module):
    def __init__(self, input_dim, cross_layer, k_Experts, r_dim):
        super(MixtureCrossNet, self).__init__()

        self.U = nn.ParameterList(
            [
                nn.Parameter(nn.init.xavier_uniform_(torch.empty(k_Experts, r_dim, input_dim))) for _ in range(cross_layer)
            ]
        )

        self.V = nn.ParameterList(
            [
                nn.Parameter(nn.init.xavier_uniform_(torch.empty(k_Experts, input_dim, r_dim))) for _ in range(cross_layer)
            ]
        )

        self.C = nn.ParameterList(
            [
                nn.Parameter(nn.init.xavier_uniform_(torch.empty(k_Experts, r_dim, r_dim))) for _ in range(cross_layer)
            ]
        )

        self.Gate = nn.Parameter(nn.init.xavier_uniform_(torch.empty(k_Experts, input_dim, 1)))

        self.g = nn.ReLU()

        self.b = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(input_dim)) for _ in range(cross_layer)
            ]
        )

    def forward(self, input):
        '''
        param: input: [B, input_dim]
        return: [B, input_dim]
        '''

        x0 = input.unsqueeze(1).unsqueeze(1) #[B, 1, 1, input_dim]
        xl = x0 #[B, 1, 1, input_dim]
        for u, v, c, b in zip(self.U, self.V, self.C, self.b):

            v = v.unsqueeze(0) #[1, k_Experts, input_dim, r_dim]
            e_out = self.g(torch.matmul(xl, v)) #[B, k_Experts, 1, r_dim]

            c = c.unsqueeze(0) #[1, k_Experts, r_dim, r_dim]
            e_out = self.g(torch.matmul(e_out, c)) #[B, k_Experts, 1, r_dim]

            u = u.unsqueeze(0) #[1, k_Experts, r_dim, input_dim]
            e_out = torch.matmul(e_out, u) + b #[B, k_Experts, 1, input_dim]

            e_out = e_out * x0 #[B, k_Experts, 1, input_dim]

            #gate
            gate = torch.matmul(xl, self.Gate) #[B, k_Experts, 1, 1]
            sum_e_out = torch.sum(e_out * gate, dim=1, keepdim=True) #[B, 1, 1, input_dim]

            xl = sum_e_out + xl #[B, 1, 1, input_dim]

        out = xl.squeeze(1).squeeze(1) #[B, input_dim]

        return out
