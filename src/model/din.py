# -*- coding: utf-8 -*-      
#--------------------------------------------
# Author:chenhao
# Date:2020-12-21 14:50
# Description:  
#--------------------------------------------
import torch
from torch import nn
from torch.nn.functional import softmax
from src.model.base import DNN, SparseEmbeddingLayer, DenseFeatCatLayer

class DIN(nn.Module):
    '''
    Deep Interest Network for Click-Through Rate Prediction
    '''
    def __init__(self, user_sparse_and_nums,
                 user_dense,
                 item_sparse_and_nums,
                 item_dense,
                 embed_dim,
                 deep_layers,
                 attention_layers=[512, 128, 1],
                 sigmoid_out=False):
        super(DIN, self).__init__()
        self.user_feat_num = len(user_sparse_and_nums)*embed_dim + len(user_dense)
        self.item_feat_num = len(item_sparse_and_nums)*embed_dim + len(item_dense)

        #embed
        self.user_embed = SparseEmbeddingLayer(feat_and_nums=user_sparse_and_nums, embed_dim=embed_dim)
        self.user_dense = DenseFeatCatLayer()
        self.item_embed = SparseEmbeddingLayer(feat_and_nums=item_sparse_and_nums, embed_dim=embed_dim)
        self.item_dense = DenseFeatCatLayer()

        #Attention
        self.attention = Attention(self.item_feat_num, layers=attention_layers)

        #Deep
        assert deep_layers[-1] == 1, "last hidden dim must be 1"
        deep_input_dim = self.user_feat_num + self.item_feat_num*2   #*2 is item feat and attention feat
        self.deep = DNN(input_dim=deep_input_dim, layers=deep_layers)

        #output
        self.sigmoid_out = sigmoid_out
        if self.sigmoid_out:
            self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        user_sparse_feat = input['user_sparse'] #dict {feat_name:[B]}
        user_dense_feat = input['user_dense'] #dict {feat_name:[B]}
        item_sparse_feat = input['item_sparse'] #dict {feat_name:[B]}
        item_dense_feat = input['item_dense'] #dict {feat_name:[B]}
        seq_item_sparse_feat = input['seq_item_sparse'] #dict {feat_name:[B, max_seq_len]}
        seq_item_dense_feat = input['seq_item_dense'] #dict {feat_name:[B, max_seq_len]}
        seq_item_length = input['seq_item_length'] # tensor[B]

        #user feat
        u_embed_x = self.user_embed(user_sparse_feat, axis=2) #[B, user_sparse*embed_dim]
        u_dense_x = self.user_dense(user_dense_feat) #[B, user_dense]
        u_x = torch.cat([u_embed_x, u_dense_x], dim=1) #[B, user_feat] user_feat=user_sparse*embed_dim+user_dense

        #item feat
        i_embed_x = self.item_embed(item_sparse_feat, axis=2)#[B, item_sparse*embed_dim]
        i_dense_x = self.item_dense(item_dense_feat) #[B, item_dense]
        i_x = torch.cat([i_embed_x, i_dense_x], dim=1) #[B, item_feat] item_feat=item_sparse*embed_dim+item_dense

        #sequence feat
        seq_i_embed_x = self.item_embed(seq_item_sparse_feat, axis=2) #[B, max_seq_len, item_sparse*embed_dim]
        seq_i_dense_x = self.item_dense(seq_item_dense_feat) #[B, max_seq_len, item_dense]]
        seq_i_x = torch.cat([seq_i_embed_x, seq_i_dense_x], dim=-1) #[B, max_seq_len, item_feat] item_feat=item_sparse*embed_dim+item_dense

        #attention
        seq_length_x = seq_item_length.unsqueeze(1) #[B, 1]
        att_out = self.attention(i_x.unsqueeze(1), seq_i_x, seq_length_x).squeeze(1) #[B, item_feat]

        #deep
        deep_x = torch.cat([u_x, i_x, att_out], dim=-1)    #[B, user_feat+item_feat+item_feat]
        out = self.deep(deep_x).squeeze(1)  #[B]

        #out
        if self.sigmoid_out:
            out = self.sigmoid(out)

        return out


class Attention(nn.Module):
    def __init__(self, item_feat, layers):
        super(Attention, self).__init__()
        assert layers[-1] == 1, "last hidden dim must be 1"
        self.linear = DNN(input_dim=4*item_feat, layers=layers)

    def forword(self, query, keys, seq_len):
        '''
        :param query: [B, 1, item_feat]
        :param keys:  [B, max_seq_len, item_feat]
        :param seq_len: [B, 1]
        :return:
        '''
        batch, max_seq_len, item_feat = keys.size()
        device = keys.device; dtype=seq_len.dtype

        #mask
        masks = torch.arange(max_seq_len, device=device, dtype=dtype).repeat(batch, 1) #[B, max_seq_len]
        masks = masks < seq_len
        masks = masks.unsqueeze(1) #[B, 1, max_seq_len]

        #attention
        query = query.expand(-1, max_seq_len, -1) #[B, max_seq_len, item_feat]
        att_x = torch.cat([query, keys, query-keys, query*keys], dim=-1) #[B, max_seq_len, item_feat*4]
        att_out = self.linear(att_x) #[B, max_seq_len, 1]
        att_out = att_out.transpose(dim0=1, dim1=2) #[B, 1, max_seq_len]

        #softmax
        paddings = torch.ones_like(att_out).to(device) * (-2 ** 32 + 1) #[B, 1, max_seq_len]
        score = torch.where(masks, att_out, paddings) #[B, 1, max_seq_len]  condition if else
        score = softmax(score, dim=-1) #[B, 1, max_seq_len] normalization

        #weight sum
        out = torch.matmul(score, keys) #[B, 1, item_feat]

        return out
