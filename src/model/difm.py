# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2021/4/19 9:14 
# Description:  
# --------------------------------------------
import torch
from torch import nn
from .ifm import IFMLinear
from .fm import FMInteraction
from .transformer import TransformerBase
from .base import DNN, SparseEmbeddingLayer

class DIFM(nn.Module):
    '''
    A Dual Input-aware Factorization Machine for CTR Prediction
    '''
    def __init__(self,sparse_feat_and_nums, dense_feat, embed_dim,
                 transformer_hidden_dim, transformer_num_attention_heads,
                 bit_layers, sigmoid_out=False):
        super(DIFM, self).__init__()

        #embed
        self.embed = SparseEmbeddingLayer(feat_and_nums=sparse_feat_and_nums, embed_dim=embed_dim)

        #Dual-FEN Layer
        self.dual_fen = DualFENLayer(vec_input_dim=embed_dim,
                                     vec_hidden_dim=transformer_hidden_dim,
                                     vec_num_attention_heads=transformer_num_attention_heads,
                                     bit_input_dim=len(sparse_feat_and_nums)*embed_dim,
                                     bit_layers=bit_layers)

        #Combination Layer
        self.combin = CombinationLayer(h1=len(sparse_feat_and_nums)*embed_dim,
                                       h2=bit_layers[-1],
                                       h=len(sparse_feat_and_nums))

        #FMInteraction vector
        self.fm_vector = FMInteraction()

        #FMInteraction linear
        self.fm_linear = IFMLinear(sparse_feat_and_nums=sparse_feat_and_nums, dense_feat=dense_feat, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))

        #output
        self.sigmoid_out = sigmoid_out
        if self.sigmoid_out:
            self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        sparse_feat = input['sparse']
        dense_feat = input['dense']

        #embed
        Uvec, Ibit = self.embed(sparse_feat, axis=-1) # Uvec [B, sparse_num, embed_dim],  Ibit [B, sparse_num*embed_dim]

        #Dual-FEN Layer
        Ovec, Obit = self.dual_fen(Uvec, Ibit) # Ovec [B, sparse_num, embed_dim], Obit [B, bit_layers[-1]]

        #Combination Layer
        mx = self.combin(Ovec, Obit) #[B, sparse_num]

        #FMInteraction vector and reweighting layer
        sparse_fm_embed_x = mx.unsqueeze(2) * Uvec #[B, sparse_num, embed_dim]
        fm_out = self.fm_vector(sparse_fm_embed_x) #[B]

        #FMInteraction linear and reweighting layer
        linear_out = self.fm_linear(sparse_feat, mx, dense_feat)

        #out
        out = linear_out + fm_out + self.bias
        if self.sigmoid_out:
            out = self.sigmoid(out)

        return out

class DualFENLayer(nn.Module):
    '''
    Dual-FEN Layer
    '''
    def __init__(self, vec_input_dim, vec_hidden_dim, vec_num_attention_heads, bit_input_dim, bit_layers):
        super(DualFENLayer, self).__init__()

        self.vec = VectorWise(input_dim=vec_input_dim, hidden_dim=vec_hidden_dim, num_attention_heads=vec_num_attention_heads)
        self.bit = BitWise(input_dim=bit_input_dim, layers=bit_layers)

    def forward(self, Uvec, Ibit):
        '''
        param Uvec: [B, sparse_num, embed_dim]
        param Ibit: [B, sparse_num*embed_dim]
        '''
        batch = Uvec.size(0)

        Ovec = self.vec(Uvec).view(batch, -1) #[B, sparse_num, embed_dim]
        Obit = self.bit(Ibit) #[B, bit_layers[-1]]

        return Ovec, Obit

class VectorWise(nn.Module):
    '''
    Dual-FEN Layer vector-wise part
    '''
    def __init__(self, input_dim, hidden_dim, num_attention_heads, resnet=True):
        super(VectorWise, self).__init__()

        self.transformer = TransformerBase(input_size=input_dim, hidden_size=hidden_dim, num_attention_heads=num_attention_heads, resnet=resnet)

    def forward(self, Uvec):
        '''
         param Uvec: [B, sparse_num, embed_dim]
        '''
        Ovec = self.transformer(Uvec) #[B, sparse_num, embed_dim]

        return Ovec

class BitWise(nn.Module):
    '''
    Dual-FEN Layer bit-wise part
    '''
    def __init__(self, input_dim, layers):
        super(BitWise, self).__init__()

        self.linear = DNN(input_dim=input_dim, layers=layers)

    def forward(self, Ibit):
        '''
        param Ibit : [B, sparse_num*embed_dim]
        '''
        Obit = self.linear(Ibit) #[B, fen_layers[-1]]

        return Obit

class CombinationLayer(nn.Module):
    '''
    Combination Layer
    '''
    def __init__(self, h1, h2, h):
        super(CombinationLayer, self).__init__()

        self.Pvec = nn.Parameter(nn.init.xavier_uniform_(torch.empty(h1, h)))
        self.Pbit = nn.Parameter(nn.init.xavier_uniform_(torch.empty(h2, h)))

    def forward(self, Ovec, Obit):
        '''
        param Ovec: [B, h1]
        param Obit: [B, h2]
        '''
        m_vec = torch.matmul(Ovec.unsqueeze(1), self.Pvec) #[B, 1, h]
        m_bit = torch.matmul(Obit.unsqueeze(1), self.Pbit) #[B, 1, h]
        m_x = m_vec + m_bit #[B, 1, h]
        m_x = m_x.squeeze(1) #[B, h]

        return m_x
