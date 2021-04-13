# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2021/4/12 20:46 
# Description:  
# --------------------------------------------
import torch
from torch import nn
from torch import Tensor
from typing import List, Dict, Tuple
from src.model.fm import FMInteraction
from torch.nn.functional import softmax
from src.model.utils import weight_init
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

        #linear
        self.linear = IFMLinear(sparse_feat_and_nums=sparse_feat_and_nums, dense_feat=dense_feat, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))

        self.sigmoid_out = sigmoid_out
        if self.sigmoid_out:
            self.sigmoid = nn.Sigmoid()


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

        #FMInteraction
        sparse_fm_embed_x = mx.unsqueeze(2) * sparse_fm_embed_x #[B, sparse_num, embed_dim]
        fm_out = self.fm(sparse_fm_embed_x) #[B]

        # linear
        linear_out = self.linear(sparse_feat, mx, dense_feat)

        #out
        out = linear_out + fm_out + self.bias
        if self.sigmoid_out:
            out = self.sigmoid(out)

        return out

class FEN(nn.Module):
    def __init__(self, input_dim, fen_layers, sparse_num):
        super(FEN, self).__init__()

        self.fc = DNN(input_dim=input_dim, layers=fen_layers)
        self.P = nn.Parameter(nn.init.xavier_uniform_(torch.empty(fen_layers[-1], sparse_num)))

    def forward(self, input):
        '''
        param input: [B, sparse_num*embed_dim]
        return [B, h]
        '''
        device = input.device
        Ux = self.fc(input) #[B, t]
        mx_ = torch.matmul(Ux, self.P) #[B, h]
        score = softmax(mx_, dim=-1) #[B, h]
        h = torch.tensor(score.size(1)).to(device)
        mx = h * score #[B, h]

        return mx

class IFMLinear(nn.Module):
    def __init__(self, sparse_feat_and_nums:List[Tuple[str, int]], dense_feat: List[str]=None, bias=True):
        super(IFMLinear, self).__init__()

        self.bias = bias

        self.sparse_weight = nn.ModuleDict(
            {
                feat: nn.Embedding(num, 1)
                for feat, num in sparse_feat_and_nums
            }
        )

        self.dense_weight = nn.ModuleDict(
            {
                feat: nn.Embedding(1, 1)
                for feat in dense_feat
            }
        )

        if self.bias:
            self.b = nn.Parameter(torch.zeros(1))

        self.init_weights()

    def init_weights(self):
        for key, layer in self.sparse_weight.items():
            weight_init(layer)
        for key, layer in self.dense_weight.items():
            weight_init(layer)

    def forward(self, sparse_feat:Dict[str, Tensor], mx, dense_feat:Dict[str, Tensor]=None):
        """
        :param sparse_feat: [B]
        :param mx: [B, sparse_num]
        :param dense_deat: [B]
        :return: [B]
        """
        sparse_x = [
            self.sparse_weight[feat](x.long())
            for feat, x in sparse_feat.items()
        ] #[[B, 1], [B, 1], ...]
        sparse_x = torch.cat(sparse_x, dim=-1) #[B, sparse_num*1]

        #ifm weight
        sparse_x = sparse_x * mx #[B, sparse_num]

        dense_x = [
            x.unsqueeze(1).float() * self.dense_weight[feat](torch.Tensor([0]).to(x.device).long())
            for feat, x in dense_feat.items()
        ] #[[B, 1], [B, 1], ...]
        dense_x = torch.cat(dense_x, dim=-1) #[B, dense_num*1]

        out = torch.cat([sparse_x, dense_x], dim=-1) #[B, sparse_num+dense_num]
        out = torch.sum(out, dim=1, keepdim=True) #[B, 1]
        if self.bias:
            out = out + self.b #[B, 1]
        out = out.squeeze(1)

        return out #[B]
