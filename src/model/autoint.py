# -*- coding: utf-8 -*-      
#--------------------------------------------
# Author:chenhao
# Date:2021-02-08 09:41
# Description:  
#--------------------------------------------
import torch
from torch import nn
from src.model.transformer import TransformerBase
from src.model.base import DNN, SparseEmbeddingLayer, DenseEmbeddingLayer

class AutoInt(nn.Module):
    '''
    AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks
    '''
    def __init__(self,
                 sparse_feat_and_nums,
                 dense_feat,
                 embed_dim,
                 hidden_size,
                 num_attention_heads,
                 num_att_layers,
                 deep_layers,
                 all_layer_output=False,
                 sigmoid_out=False):
        super(AutoInt, self).__init__()
        #define
        self.all_layer_output = all_layer_output
        self.num_feat = len(sparse_feat_and_nums) + len(dense_feat)
        self.att_output_dim = num_att_layers*embed_dim*self.num_feat if self.all_layer_output else embed_dim*self.num_feat

        #embed
        self.sparse_embed = SparseEmbeddingLayer(feat_and_nums=sparse_feat_and_nums, embed_dim=embed_dim)
        self.dense_embed = DenseEmbeddingLayer(dense_feat=dense_feat, embed_dim=embed_dim)

        #transformer
        self.layers = nn.ModuleList(
            [
                TransformerBase(input_size=embed_dim, hidden_size=hidden_size, num_attention_heads=num_attention_heads, resnet=True)
                for _ in range(num_att_layers)
            ]
        )

        #deep
        assert deep_layers[-1] == 1, "last hidden dim must be 1"
        self.deep = DNN(input_dim= self.att_output_dim, layers=deep_layers, act='tanh')

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
        att_out = []
        ei = embed_out
        for i, layer in enumerate(self.layers):
            ei = layer(ei)
            att_out.append(ei)

        if self.all_layer_output:
            att_out = torch.cat(att_out, dim=-1).view(-1, self.att_output_dim)
        else:
            att_out = att_out[-1].view(-1, self.att_output_dim)

        # deep
        out = self.deep(att_out).squeeze(1) #[B]

        if self.sigmoid_out:
            out = self.sigmoid(out)

        return out
