# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2021/4/12 20:46 
# Description:  
# --------------------------------------------
import torch
from torch import nn
from src.model.fm import FMInteraction
from torch.nn.functional import softmax
from src.model.base import DNN, SparseEmbeddingLayer, DenseFeatCatLayer

class IFM(nn.Module):
    '''
    An Input-aware Factorization Machine for Sparse Prediction
    '''
    def __init__(self,sparse_feat_and_nums, dense_feat, embed_dim, fen_layers, sigmoid_out=False):
        super(IFM, self).__init__()

        #embed
        self.embed = SparseEmbeddingLayer(feat_and_nums=sparse_feat_and_nums, embed_dim=embed_dim)
        self.dense = DenseFeatCatLayer()

        #Factor Estimating Network
        self.fen = FEN(len(sparse_feat_and_nums)*embed_dim, fen_layers=fen_layers, sparse_num=len(sparse_feat_and_nums))

        #FMInteraction
        self.fm = FMInteraction()


    def forward(self, input):
        sparse_feat = input['sparse']
        dense_feat = input['dense']

        #embed
        # sparse_fm_embed_x [B, sparse_num, embed_dim]
        # sparse_fen_embed_x [B, sparse_num*embed_dim]
        sparse_fm_embed_x, sparse_fen_embed_x = self.embed(sparse_feat, axis=-1)
        dense_deep_x = self.dense(dense_feat) #[B, dense_num]

        # Factor Estimating Network
        mx = self.fen(sparse_fen_embed_x) #[B, sparse_num]
        sparse_fm_embed_x = mx.unsqueeze(2) * sparse_fm_embed_x #[B, sparse_num, embed_dim]
        fm_out = self.fm(sparse_fm_embed_x) #[B]








class FEN(nn.Module):
    def __init__(self, input_dim, fen_layers, sparse_num):
        super(FEN, self).__init__()

        self.h = torch.tensor([sparse_num])
        self.fc = DNN(input_dim=input_dim, layers=fen_layers)
        self.P = nn.Parameter(nn.init.xavier_uniform_(fen_layers[-1], sparse_num))


    def forward(self, input):
        '''
        param input: [B, sparse_num*embed_dim]
        return [B, h]
        '''
        device = input.device
        Ux = self.fc(input) #[B, t]
        mx_ = torch.matmul(Ux, self.P) #[B, h]
        score = softmax(mx_, dim=-1) #[B, h]
        mx = self.h * score #[B, h]

        return mx










