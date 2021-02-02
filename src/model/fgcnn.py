# -*- coding: utf-8 -*-      
#--------------------------------------------
# Author:chenhao
# Date:2021-01-13 21:15
# Description:  
#--------------------------------------------
import torch
from torch import nn
from functools import reduce
from operator import __add__
from src.model.base import DNN, SparseEmbeddingLayer, DenseFeatCatLayer

class FGCNN(nn.Module):
    def __init__(self, sparse_feat_and_nums,
                 dense_feat,
                 embed_dim,
                 deep_layers,
                 rec_feat_maps=(64,32,16),
                 con_channels=(64,32,16),
                 con_kernel_sizes=(7, 5, 3),
                 pool_kernel_sizes=(2,2,2),
                 is_share_embed=False,

                 sigmoid_out=False):
        super(FGCNN, self).__init__()

        #define
        assert len(con_kernel_sizes) == len(pool_kernel_sizes) and  \
               len(pool_kernel_sizes) == len(con_channels) and  \
               len(con_channels) == len(rec_feat_maps), \
               'convolution layers kernel size, convolution channels size, pool layers kernel size, recommend layers feat maps must equal'

        #Feature Generation embed
        self.fg_embed = SparseEmbeddingLayer(feat_and_nums=sparse_feat_and_nums, embed_dim=embed_dim)
        self.fg_dense = DenseFeatCatLayer()

        #Deep Classifier embed
        if is_share_embed:
            self.dc_embed = self.fg_embed
            # self.dc_dense = self.fg_dense
        else:
            self.dc_embed = SparseEmbeddingLayer(feat_and_nums=sparse_feat_and_nums, embed_dim=embed_dim)
            # self.dc_dense = DenseFeatCatLayer()

        #Feature Generation
        self.inner_feat_nums = len(sparse_feat_and_nums) #raw features
        self.convs_feat_nums_list = [len(sparse_feat_and_nums)]
        self.convs = nn.ModuleList()

        for i in range(len(con_channels)):
            self.convs.append(
                CNNRec(
                    in_channels=con_channels[i-1] if i != 0 else 1, #first channel is 1
                    out_channels=con_channels[i],
                    feat_nums=self.convs_feat_nums_list[-1],
                    con_kernel=con_kernel_sizes[i],
                    pool_kernel=pool_kernel_sizes[i],
                    rec_feat_map=rec_feat_maps[i]
                )
            )
            cur_rec_feat_nums = self.convs_feat_nums_list[-1] // pool_kernel_sizes[i]
            self.inner_feat_nums += cur_rec_feat_nums * rec_feat_maps[i] #new features
            self.convs_feat_nums_list.append(cur_rec_feat_nums)

        #IPNN
        self.Inner = InnerProduct()

        #Deep Classifier
        assert deep_layers[-1] == 1, "last hidden dim must be 1"

        self.deep_input_dim = int(self.inner_feat_nums*(self.inner_feat_nums-1)/2) + len(sparse_feat_and_nums) * embed_dim + len(dense_feat)
        self.deep = DNN(input_dim= self.deep_input_dim, layers=deep_layers, act='tanh', bn=True)

        #output
        self.sigmoid_out = sigmoid_out
        if self.sigmoid_out:
            self.sigmoid = nn.Sigmoid()


    def forward(self, input):
        sparse_feat = input['sparse'] #[B]
        dense_feat = input['dense'] #[B]

        #Feature Generation embed
        sparse_fg_embed_x = self.fg_embed(sparse_feat, axis=1)  # [B, sparse_num, embed_dim]
        deep_fg_dense_x = self.fg_dense(dense_feat) #[B, dense_num]

        #Deep Classifier embed
        sparse_dc_embed_x = self.dc_embed(sparse_feat, axis=2)  # [B, sparse_num*embed_dim]
        # deep_dc_dense_x = self.dc_dense(dense_feat) #[B, dense_num]

        # Feature Generation
        rec_outs = [sparse_fg_embed_x]
        fg_con_x = sparse_fg_embed_x.unsqueeze(1) #[B, 1, sparse_num, embed_dim]
        for conv in self.convs:
            pool_out, rec_out = conv(fg_con_x)
            rec_outs.append(rec_out)
            fg_con_x = pool_out

        # IPNN
        inner_x = torch.cat(rec_outs, dim=1) #[B, self.inner_feat_nums, embed_dim]
        inner_out = self.Inner(inner_x) #[B, self.inner_feat_nums*(self.inner_feat_nums-1)/2]

        #Deep Classifier
        deep_x = torch.cat([sparse_dc_embed_x, inner_out, deep_fg_dense_x], dim=1) #[B, self.deep_input_dim]
        out = self.deep(deep_x).squeeze(1) #[B]

        #out
        if self.sigmoid_out:
            out = self.sigmoid(out)

        return out

class CNNRec(nn.Module):
    def __init__(self, in_channels, out_channels, feat_nums, con_kernel, pool_kernel, rec_feat_map):
        super(CNNRec, self).__init__()

        #Convolutional Layer tensorflow 'same'
        self.kernel_size = (con_kernel, 1)
        self.conv_padding = reduce(__add__, [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]])
        self.pad = nn.ZeroPad2d(self.conv_padding)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=self.kernel_size, stride=(1,1))
        self.act = nn.Tanh()

        # Pooling Layer
        self.pool_kernel = pool_kernel
        self.pool = nn.MaxPool2d(kernel_size=(self.pool_kernel, 1), stride=(self.pool_kernel, 1))
        self.pool_feat_nums = feat_nums // self.pool_kernel

        #recommend layer
        self.rec = nn.Sequential(*[nn.Linear(out_channels*self.pool_feat_nums, rec_feat_map*self.pool_feat_nums), nn.Tanh(), nn.Dropout()])

    def forward(self, x):
        '''
        :param x: [B, C, H, W]
        :return:
        '''

        #Convolutional Layer
        batch, channel, high, width = x.size()
        pad_out = self.pad(x)
        con_out = self.act(self.conv(pad_out)) #[B, out_channels, H, W]

        #Pooling Layer
        pool_out = self.pool(con_out) #[B, out_channels, self.pool_feat_nums, W]

        #Recombination Layer
        rec_x = pool_out.reshape(batch, -1, width).permute(0, 2, 1).contiguous() #[B, out_channels*self.pool_feat_nums, W]
        rec_out = self.rec(rec_x).permute(0, 2, 1) #[B, rec_feat_map*self.pool_feat_nums, W]

        return pool_out, rec_out

class InnerProduct(nn.Module):
    def __init__(self):
        super(InnerProduct, self).__init__()
    def forward(self, input):
        '''
        :param input: [B, feat_num, embed_dim]
        :return #[B, feat_num*(feat_num-1)/2]
        '''
        x = input #[B, feat_num, embed_dim]
        batch = x.size(0)
        inner_out = torch.matmul(x, x.transpose(dim0=1, dim1=2)) #[B, feat_num, feat_num]

        #get triu data
        index = torch.ones_like(inner_out, dtype=inner_out.dtype, device=inner_out.device)
        index = torch.triu(index, diagonal=1).bool()
        triu_out = inner_out[index]
        chunk = triu_out.chunk(batch)
        out = torch.cat([w.unsqueeze(0) for w in chunk], dim=0)

        return out
