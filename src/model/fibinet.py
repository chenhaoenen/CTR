# -*- coding: utf-8 -*-      
#--------------------------------------------
# Author:chenhao
# Date:2021-01-04 10:24
# Description:  
#--------------------------------------------
import torch
from torch import nn
from src.model.base import DNN, Linear, SparseEmbeddingLayer, DenseFeatCatLayer

class FiBiNet(nn.Module):
    '''
    FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction
    '''
    def __init__(self, sparse_feat_and_nums, dense_feat, embed_dim, deep_layers, senet_factor, inter_type='all', sigmoid_out=False):
        super(FiBiNet, self).__init__()

        #shared embed
        self.embed = SparseEmbeddingLayer(feat_and_nums=sparse_feat_and_nums, embed_dim=embed_dim)
        self.dense = DenseFeatCatLayer()

        #linear
        self.linear = Linear(sparse_feat_and_nums, dense_feat, bias=False)

        #SeNet
        self.senet = SeNet(len(sparse_feat_and_nums), senet_factor)

        #BilinearInteraction
        self.biLinear = BilinearInteraction(num_feat=len(sparse_feat_and_nums), embed_dim=embed_dim, inter_type=inter_type)

        #deep
        assert deep_layers[-1] == 1, "last hidden dim must be 1"
        self.deep_input_dim = len(sparse_feat_and_nums)*(len(sparse_feat_and_nums)-1) * embed_dim + len(dense_feat)
        self.deep = DNN(input_dim= self.deep_input_dim, layers=deep_layers, act='tanh')

        #output
        self.sigmoid_out = sigmoid_out
        if self.sigmoid_out:
            self.sigmoid = nn.Sigmoid()


    def forward(self, input):
        sparse_feat = input['sparse'] #[B]
        dense_feat = input['dense'] #[B]

        #embed
        embed_bi_x = self.embed(sparse_feat, axis=1)  # [B, sparse_num, embed_dim]
        deep_dense_x = self.dense(dense_feat) #[B, dense_num]

        #linear
        linear_out = self.linear(sparse_feat, dense_feat) #[B]

        #Bilinear-Interaction
        senet_bi_x = self.senet(embed_bi_x) #[B, sparse_num, embed_dim]

        embed_bi_out = self.biLinear(embed_bi_x) #[B, sparse_num*(sparse_num-1)//2, embed_dim]
        senet_bi_out = self.biLinear(senet_bi_x) #[B, sparse_num*(sparse_num-1)//2, embed_dim]

        bi_out = torch.cat([embed_bi_out, senet_bi_out], dim=1) #[B, sparse_num*(sparse_num-1), embed_dim]
        bi_out = torch.flatten(bi_out, start_dim=1) #[B, sparse_num*(sparse_num-1)*embed_dim]

        #deep
        deep_x = torch.cat([bi_out, deep_dense_x], dim=-1) #[B, sparse_num*(sparse_num-1)*embed_dim+dense_num]
        deep_out = self.deep(deep_x).squeeze(1) #[B]

        #out
        out = linear_out + deep_out #[B]

        if self.sigmoid_out:
            out = self.sigmoid(out)

        return out

class SeNet(nn.Module):
    def __init__(self, num_dim, factor):
        super(SeNet, self).__init__()

        layers = []
        layers.append(nn.Linear(num_dim, factor, bias=False))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(factor, num_dim, bias=False))
        layers.append(nn.ReLU())

        self.linear = nn.Sequential(*layers)

    def forward(self, input):
        '''
        :param input: #[B, num, embed_dim]
        :return: #[B, num, embed_dim]
        '''
        x = input #[B, num, embed_dim]

        #Z
        Z = torch.mean(x, dim=-1, keepdim=False) #[B, num]

        #A
        A = self.linear(Z) #[B, num]

        #weight
        out = A.unsqueeze(2) * x  #[B, num, embed_dim]

        return out

class BilinearInteraction(nn.Module):
    '''
    不同于nfm中的BiInteractionPooling
    '''
    def __init__(self, num_feat, embed_dim, inter_type='all'):
        super(BilinearInteraction, self).__init__()
        self.num_feat = num_feat
        self.inter_type = inter_type
        if self.inter_type == 'all':
            self.biLinear = nn.Linear(embed_dim, embed_dim, bias=False)
        elif self.inter_type == 'each':
            self.biLinear = nn.ModuleList(
                [
                    nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(self.num_feat)
                ]
            )
        elif self.inter_type == 'interaction':
            self.biLinear = nn.ModuleList(
                [
                    [
                        nn.Linear(embed_dim, embed_dim, bias=False)
                        for _ in range(i+1, self.num_feat)
                    ]
                    for i in range(self.num_feat)
                ]
            )
        else:
            raise  NotImplementedError("the interaction type must in 'all, each, interaction' but get {}".format(self.inter_type))


    def forward(self, input):
        '''
        :param input: #[B, num, embed_dim]
        :return: #[B, m, dim] m=num*(num_-1)/2
        '''
        x = input

        if self.inter_type == 'all':
            inner_x1 = [self.biLinear(x[:,i,:].unsqueeze(1)) for i in range(self.num_feat) for j in range(i+1, self.num_feat)]
        elif self.inter_type == 'each':
            inner_x1 = [self.biLinear[i](x[:,i,:].unsqueeze(1)) for i in range(self.num_feat) for j in range(i+1, self.num_feat)]
        elif self.inter_type == 'interaction':
            inner_x1 = [self.biLinear[i][j](x[:,i,:].unsqueeze(1)) for i in range(self.num_feat) for j in range(i+1, self.num_feat)]
        else:
            raise NotImplementedError("the interaction type must in 'all, each, interaction' but get {}".format(self.inter_type))
        inner_x2 = [self.biLinear(x[:, j, :].unsqueeze(1)) for i in range(self.num_feat) for j in range(i + 1, self.num_feat)]

        inner_x1 = torch.cat(inner_x1, dim=1) #[B, m, dim]
        inner_x2 = torch.cat(inner_x2, dim=1) #[B, m, dim]

        out = inner_x1 * inner_x2 #[B, m, dim]

        return out
