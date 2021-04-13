# -*- coding: utf-8 -*-      
#--------------------------------------------
# Author:chenhao
# Date:2020-12-12 17:39
# Description:  
#--------------------------------------------
import torch
from torch import nn, Tensor
from typing import Dict, List, Tuple
from src.model.activation import activation
from src.model.utils import weight_init

#wx+b
class Linear(nn.Module):
    def __init__(self, sparse_feat_and_nums:List[Tuple[str, int]], dense_feat: List[str]=None, bias=True):
        super(Linear, self).__init__()

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

    def forward(self, sparse_feat:Dict[str, Tensor], dense_feat:Dict[str, Tensor]=None):
        """
        :param sparse_feat: [B]
        :param dense_deat: [B]
        :return: [B]
        """
        sparse_x = [
            self.sparse_weight[feat](x.long())
            for feat, x in sparse_feat.items()
        ] #[[B, 1], [B, 1], ...]
        sparse_x = torch.cat(sparse_x, dim=-1) #[B, sparse_num*1]

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

class DNN(nn.Module):
    def __init__(self, input_dim, layers, act='relu', drop=0.5, bn=False):
        super(DNN, self).__init__()
        dense = [nn.Linear(input_dim, layers[0])]
        if act:
            dense.append(activation(act))
        if bn:
            dense.append(nn.BatchNorm1d(layers[0]))
        if drop:
            dense.append(nn.Dropout(p=drop))

        for i in range(1, len(layers)):
            dense.append(nn.Linear(layers[i-1], layers[i]))
            if act and i != len(layers)-1:  #last layer cant activation
            # if act:
                dense.append(activation(act))
            if bn:
                dense.append(nn.BatchNorm1d(layers[i]))
            if drop:
                dense.append(nn.Dropout(p=drop))

        self.dense = nn.Sequential(*dense)

        self.init_weights()

    def init_weights(self):
        for layer in self.dense:
            weight_init(layer)

    def forward(self, input):
        '''
        :param input: [B, feat_dim]
        :return: [B, layers[-1]]
        '''
        x = input
        out = self.dense(x)

        return out

class SparseEmbeddingLayer(nn.Module):
    def __init__(self, feat_and_nums:List[Tuple[str, int]], embed_dim):
        super(SparseEmbeddingLayer, self).__init__()

        self.embed = nn.ModuleDict(
            {
                feat: nn.Embedding(num, embed_dim)
                for feat, num in feat_and_nums
            }
        )
        self.init_weights()

    def init_weights(self):
        for key, layer in self.embed.items():
            weight_init(layer)

    def forward(self, feats:Dict[str, Tensor], axis=2):
        '''
        :param feats: dict[feat_name: Tensor], Tensor[B]
        :return:
        '''

        if axis == 1:
            embed_out = [
                self.embed[feat](x.long()).unsqueeze(1)
                for feat, x in feats.items()
            ]  # [[B, 1, embed_dim], [B, 1, embed_dim], ...,[B, 1, embed_dim]]

            return torch.cat(embed_out, dim=1) #[B, feat_num, embed_dim]

        elif axis == 2:
            embed_out = [
                self.embed[feat](x.long())
                for feat, x in feats.items()
            ]  # [[B, embed_dim], [B, embed_dim], ...,[B, embed_dim]]

            return torch.cat(embed_out, dim=-1) #[B, feat_num*embed_dim]

        else:
            embed_out = [
                self.embed[feat](x.long()).unsqueeze(1)
                for feat, x in feats.items()
            ]  # [[B, 1, embed_dim], [B, 1, embed_dim], ...,[B, 1, embed_dim]]

            dim1_out = torch.cat(embed_out, dim=1) #[B, feat_num, embed_dim]
            dim2_out = torch.cat(embed_out, dim=-1).squeeze(1) #[B, feat_num*embed_dim]

            return dim1_out, dim2_out

class DenseFeatCatLayer(nn.Module):
    def __init__(self):
        super(DenseFeatCatLayer, self).__init__()
    def forward(self, feat:Dict[str, Tensor]):
        '''
        :param feat: dict[feat_name: Tensor], Tensor[B]
        :return: #[[B, 1], [B, 1], ...,[B, 1]
        '''
        u_dense_x = [
            x.unsqueeze(1).float()
            for feat, x in feat.items()
        ] #[[B, 1], [B, 1], ...,[B, 1]]
        out = torch.cat(u_dense_x, dim=1) #[B, user_dense]

        return out

class DenseEmbeddingLayer(nn.Module):
    def __init__(self, dense_feat:List, embed_dim):
        super(DenseEmbeddingLayer, self).__init__()

        self.embed = nn.ModuleDict(
            {
                feat: nn.Embedding(1, embed_dim)
                for feat in dense_feat
            }
        )
        self.init_weights()

    def init_weights(self):
        for key, layer in self.embed.items():
            weight_init(layer)

    def forward(self, feats:Dict[str, Tensor], axis=2):
        '''
        :param feats: dict[feat_name: Tensor], Tensor:[B]
        :return:
        '''
        device = list(feats.values())[0].device
        idx0 = torch.tensor([0]).long().to(device)
        if axis == 1:
            embed_out = [
                (x.unsqueeze(1).float() * self.embed[feat](idx0)).unsqueeze(1)
                for feat, x in feats.items()
            ]  # [[B, 1, embed_dim], [B, 1, embed_dim], ...,[B, 1, embed_dim]]

            return torch.cat(embed_out, dim=1) #[B, feat_num, embed_dim]

        elif axis == 2:
            embed_out = [
                (x.unsqueeze(1).float() * self.embed[feat](idx0)).unsqueeze(1)
                for feat, x in feats.items()
            ]  # [[B, embed_dim], [B, embed_dim], ...,[B, embed_dim]]

            return torch.cat(embed_out, dim=-1) #[B, feat_num*embed_dim]

        else:
            embed_out = [
                (x.unsqueeze(1).float() * self.embed[feat](idx0)).unsqueeze(1)
                for feat, x in feats.items()
            ]  # [[B, 1, embed_dim], [B, 1, embed_dim], ...,[B, 1, embed_dim]]

            dim1_out = torch.cat(embed_out, dim=1) #[B, feat_num, embed_dim]
            dim2_out = torch.cat(embed_out, dim=-1).squeeze(1) #[B, feat_num*embed_dim]

            return dim1_out, dim2_out
