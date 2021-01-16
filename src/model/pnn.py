# -*- coding: utf-8 -*-      
#--------------------------------------------
# Author:chenhao
# Date:2020-12-15 16:46
# Description:  
#--------------------------------------------
import torch
from torch import nn
from src.model.base import DNN, EmbeddingLayer, DenseFeatCatLayer

class PNN(nn.Module):
    '''
    Product-based Neural Networks for User Response Prediction
    '''
    def __init__(self, sparse_feat_and_nums, dense_feat, embed_dim, deep_layers, inner_product=True, outer_product=True, sigmoid_out=False):
        super(PNN, self).__init__()
        self.inner_product = inner_product
        self.outer_product = outer_product
        self.D1 = deep_layers[0]

        #embed
        self.embed = EmbeddingLayer(feat_and_nums=sparse_feat_and_nums, embed_dim=embed_dim)
        self.dense = DenseFeatCatLayer()

        #Wz
        self.Wz = nn.Conv1d(in_channels=len(sparse_feat_and_nums), out_channels=self.D1, kernel_size=embed_dim)

        #Wp
        if self.inner_product: #IPNN
            self.Inner = InnerProduct()
            self.Wp_inner = nn.Conv1d(in_channels=len(sparse_feat_and_nums), out_channels=self.D1, kernel_size=len(sparse_feat_and_nums))
        if self.outer_product: #OPNN
            self.Outer = OuterProduct()
            self.Wp_outer = nn.Conv1d(in_channels=embed_dim, out_channels=self.D1, kernel_size=embed_dim)

        #b1
        self.B1 = nn.Parameter(nn.init.xavier_normal_(torch.empty(self.D1, 1)))

        #Deep
        assert deep_layers[-1] == 1, "last hidden dim must be 1"
        self.deep_input_dim = self.D1+len(dense_feat)
        self.deep = DNN(input_dim= self.deep_input_dim, layers=deep_layers, act='tanh')

        #output
        self.sigmoid_out = sigmoid_out
        if self.sigmoid_out:
            self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        sparse_feat = input['sparse'] #[B]
        dense_feat = input['dense'] #[B]

        #embed
        sparse_embed_x = self.embed(sparse_feat, axis=1)  # [B, sparse_num, embed_dim]
        deep_dense_x = self.dense(dense_feat) #[B, dense_num]

        #Lz
        Lz_out = self.Wz(sparse_embed_x) #[B, self.D1, 1]

        #Lp
        if self.inner_product and self.outer_product:
            inner_out = self.Inner(sparse_embed_x) #[B, sparse_num, sparse_num]
            Lp_inner_out = self.Wp_inner(inner_out) #[B, self.D1, 1]

            outer_out = self.Outer(sparse_embed_x) #[B, embed_dim, embed_dim]
            Lp_outer_out = self.Wp_outer(outer_out) #[B, self.D1, 1]

            Lp_out = Lp_inner_out + Lp_outer_out #[B, self.D1, 1]

        elif self.outer_product:
            outer_out = self.Outer(sparse_embed_x) #[B, embed_dim, embed_dim]
            Lp_out = self.Wp_outer(outer_out) #[B, self.D1, 1]

        elif self.inner_product:
            inner_out = self.Inner(sparse_embed_x) #[B, sparse_num, sparse_num]
            Lp_out = self.Wp_inner(inner_out) #[B, self.D1, 1]
        else:
            raise NotImplementedError('dont have others implemented')

        #L1
        L1 = (Lz_out + Lp_out + self.B1).squeeze(2) #[B, self.D1, 1] -> [B, self.D1]

        #Deep
        deep_x = torch.cat([L1, deep_dense_x], dim=1) #[B, self.D1+dense_num]
        out = self.deep(deep_x).squeeze(1) #[B]

        #out
        if self.sigmoid_out:
            out = self.sigmoid(out)

        return out

class InnerProduct(nn.Module):
    def __init__(self):
        super(InnerProduct, self).__init__()
    def forward(self, input):
        '''
        :param input:[B, feat_num, embed_dim]
        :return [B, feat_num, feat_num]
        '''
        embed_x = input
        out = torch.matmul(embed_x, embed_x.transpose(dim0=1, dim1=2)) #[B, feat_num, feat_num]

        return out

class OuterProduct(nn.Module):
    def __init__(self):
        super(OuterProduct, self).__init__()
    def forward(self, input):
        '''
        :param input: [B, feat_num, embed_dim]
        :return [B, embed_dim, embed_dim]
        '''
        x = input #[B, feat_num, embed_dim]
        device = x.device
        batch, num_feat, embed_dim = x.size()
        out = torch.zeros(batch, embed_dim, embed_dim).to(device) #[B, embed_dim, embed_dim]
        for i in range(num_feat):
            fi = x[:, i, :].unsqueeze(1) #[B, 1, embed_dim]
            if i != num_feat - 1:
                for j in range(i+1, num_feat):
                    fj = x[:,j,:].unsqueeze(1) #[B, 1, embed_dim]
                    ij_out_product = torch.matmul(fi.transpose(dim0=1, dim1=2), fj) #[B, embed_dim, 1]*[B, 1, embed_dim]=[B, embed_dim, embed_dim]

                    out += ij_out_product #上三角
                    out += ij_out_product.transpose(dim0=1, dim1=2) #下三角

            ii_out_product = torch.matmul(fi.transpose(dim0=1, dim1=2), fi) #[B, embed_dim, embed_dim]
            out += ii_out_product #对角线

        return out
