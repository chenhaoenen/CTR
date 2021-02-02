# -*- coding: utf-8 -*-      
#--------------------------------------------
# Author:chenhao
# Date:2021-01-15 11:50
# Description:  
#--------------------------------------------
import torch
from torch import nn
from functools import reduce
from operator import __add__
from src.model.base import DNN, SparseEmbeddingLayer, DenseFeatCatLayer

class CCPM(nn.Module):
    '''
    A Convolutional Click Prediction Model
    '''
    def __init__(self, sparse_feat_and_nums,
                 dense_feat,
                 embed_dim,
                 deep_layers,
                 con_channels=(64, 32, 16),
                 con_kernel_sizes=(7, 5, 3),
                 sigmoid_out=False):
        super(CCPM, self).__init__()

        #embed
        self.embed = SparseEmbeddingLayer(feat_and_nums=sparse_feat_and_nums, embed_dim=embed_dim)
        self.dense = DenseFeatCatLayer()

        #CNN layer
        conv_layers = []
        self.cur_feat_nums = len(sparse_feat_and_nums)
        n = len(sparse_feat_and_nums)
        l = len(con_channels)

        for i in range(1, l+1):
            k = max(1, int((1 - pow(i / l, l - i)) * n)) if i < l else 3

            conv_layers.append(
                CNNLayer(
                    in_channels=con_channels[i-2] if i != 1 else 1, #first channel is 1
                    out_channels=con_channels[i-1],
                    feat_nums = self.cur_feat_nums,
                    con_kernel=con_kernel_sizes[i-1],
                    topk=k
                )
            )
            self.cur_feat_nums = min(k, self.cur_feat_nums)
        self.convs = nn.Sequential(*conv_layers)

        #deep
        assert deep_layers[-1] == 1, "last hidden dim must be 1"
        self.deep_input_dim = self.cur_feat_nums * con_channels[-1] * embed_dim + len(dense_feat)
        self.deep = DNN(input_dim= self.deep_input_dim, layers=deep_layers, act='tanh', bn=True)

        #output
        self.sigmoid_out = sigmoid_out
        if self.sigmoid_out:
            self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        sparse_feat = input['sparse'] #[B]
        dense_feat = input['dense'] #[B]

        #embed
        sparse_embed_x = self.embed(sparse_feat, axis=1)  # [B, sparse_num, embed_dim]
        dense_deep_x = self.dense(dense_feat)  # [B, dense_num]

        #CNN Layer
        conv_out = self.convs(sparse_embed_x.unsqueeze(1))

        #deep
        batch = sparse_embed_x.size(0)
        deep_conv_x = conv_out.view(batch, -1) # [B, self.feat_nums*con_channels[-1] * embed_dim]
        deep_x = torch.cat([deep_conv_x, dense_deep_x], dim=1) #[B, self.deep_input_dim]
        out = self.deep(deep_x).squeeze(1) #[B]

        #out
        if self.sigmoid_out:
            out = self.sigmoid(out)

        return out


class CNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, feat_nums, con_kernel, topk):
        super(CNNLayer, self).__init__()

        #Convolutional Layer tensorflow 'same'
        self.kernel_size = (con_kernel, 1)
        self.conv_padding = reduce(__add__, [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]])
        self.pad = nn.ZeroPad2d(self.conv_padding)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=self.kernel_size, stride=(1,1))
        self.act = nn.Tanh()

        # Pooling Layer
        self.topk = min(feat_nums, topk)

    def forward(self, x):
        '''
        :param x: [B, C, H, W]
        :return:
        feat_nums = H
        embed_dim = W
        channel = C
        '''
        #Convolutional Layer
        pad_out = self.pad(x)
        con_out = self.act(self.conv(pad_out)) #[B, out_channels, H, W]

        #p-MaxPooling
        pool_out = torch.topk(con_out, dim=2, k=self.topk)[0] #[B, out_channels, H, 1]

        return pool_out
