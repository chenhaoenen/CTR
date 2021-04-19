# -*- coding: utf-8 -*-      
#--------------------------------------------
# Author:chen hao
# Date:2020-09-09 11:09
# Description:  
#--------------------------------------------
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class TransformerBase(nn.Module):
    def __init__(self, input_size, hidden_size, num_attention_heads, dropout=0.5, act='relu', resnet=False):
        super(TransformerBase, self).__init__()

        self.self = TransformerSelfAttention(input_size, hidden_size, num_attention_heads, dropout=dropout)
        self.output = TransformerSelfOutput(input_size, hidden_size, dropout=dropout, act=act, resnet=resnet)

    def forward(self, x):
        '''
        :param x: [B, feat_num, input_size]
        :return:
        '''
        attention_out = self.self(x) #[B, feat_num, hidden_size]
        output_out = self.output(x, attention_out) #[B, feat_num, input_size]

        return output_out

class TransformerSelfAttention(nn.Module):
    def __init__(self, input_size,  hidden_size, num_attention_heads, dropout):
        super(TransformerSelfAttention, self).__init__()
        assert hidden_size % num_attention_heads == 0,  'multi heads attentions: hidden size must to be divisible by num_attention_heads'

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        :param x: x[B, feat_num, input_size]
        :return:
        '''
        mixed_query_layer = self.query(x) #[B, feat_num, all_head_size]
        mixed_key_layer = self.key(x) #[B, feat_num, all_head_size]
        mixed_value_layer = self.value(x) #[B, feat_num, all_head_size]

        query_layer = self.transpose_for_scores(mixed_query_layer) #[B, num_attention_heads, feat_num, attention_head_size]
        key_layer = self.transpose_key_for_scores(mixed_key_layer) #[B, num_attention_heads, attention_head_size, feat_num]
        value_layer = self.transpose_for_scores(mixed_value_layer) #[B, num_attention_heads, feat_num, attention_head_size]

        attention_scores = torch.matmul(query_layer, key_layer) #[B, num_attention_heads, feat_num, feat_num]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) #[B, num_attention_heads, feat_num, feat_num]

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer) #[B, num_attention_heads, feat_num, attention_head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() #[B, feat_num, num_attention_heads, attention_head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size, )
        context_layer = torch.reshape(context_layer, new_context_layer_shape) #[B, feat_num, all_head_size]

        return context_layer



    def transpose_for_scores(self, x):
        '''
        :param x: [B, feat_num, all_head_size]
        :return:
        '''
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = torch.reshape(x, new_shape) #[B, feat_num, num_attention_heads, attention_head_size]
        x = x.permute(0, 2, 1, 3) #[B, num_attention_heads, feat_num, attention_head_size]

        return x

    def transpose_key_for_scores(self, x):
        '''
         :param x: [B, feat_num, all_head_size]
         :return:
         '''
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = torch.reshape(x, new_shape)  # [B, feat_num, num_attention_heads, attention_head_size]
        x = x.permute(0, 2, 3, 1)  # [B, num_attention_heads, attention_head_size, feat_num]

        return x

class TransformerSelfOutput(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, act, resnet):
        super(TransformerSelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(dropout)
        self.resnet = resnet
        if act == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.GELU()
    def forward(self, x, attention_out):
        '''
        :param x: [B， feat_num, input_size]
        :param attention_out: [B， feat_num, hidden_size]
        :return:
        '''
        hidden_states = self.dense(attention_out) #[B, feat_num, input_size]
        if self.resnet:
            hidden_states = x + hidden_states
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return  hidden_states
