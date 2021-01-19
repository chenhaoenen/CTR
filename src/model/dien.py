# -*- coding: utf-8 -*-      
#--------------------------------------------
# Author:chenhao
# Date:2021-01-08 15:13
# Description:  
#--------------------------------------------
import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from src.model.base import DNN, EmbeddingLayer, DenseFeatCatLayer

class DIEN(nn.Module):
    '''
    Deep Interest Evolution Network for Click-Through Rate Prediction
    Reference: https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/dien.py
    '''
    def __init__(self, user_sparse_and_nums,
                 user_dense,
                 item_sparse_and_nums,
                 item_dense,
                 embed_dim,
                 deep_layers,
                 gru1_hidden_dim=512,
                 gru2_hidden_dim=256,
                 sigmoid_out=False):
        super(DIEN, self).__init__()
        self.user_feat_num = len(user_sparse_and_nums)*embed_dim + len(user_dense)
        self.item_feat_num = len(item_sparse_and_nums)*embed_dim + len(item_dense)

        #embed
        self.user_embed = EmbeddingLayer(feat_and_nums=user_sparse_and_nums, embed_dim=embed_dim)
        self.user_dense = DenseFeatCatLayer()
        self.item_embed = EmbeddingLayer(feat_and_nums=item_sparse_and_nums, embed_dim=embed_dim)
        self.item_dense = DenseFeatCatLayer()

        #InterestExtractorLayer
        self.extractor = InterestExtractorLayer(input_dim=self.item_feat_num, hidden_dim=gru1_hidden_dim)
        #InterestEvolvingLayer
        self.evolving = InterestEvolvingLayer(item_feat_dim=self.item_feat_num,
                                              input_dim=gru1_hidden_dim,
                                              hidden_dim=gru2_hidden_dim,
                                              gru_type='AUGRU')

        #Deep
        assert deep_layers[-1] == 1, "last hidden dim must be 1"
        deep_input_dim = self.user_feat_num + self.item_feat_num + gru2_hidden_dim
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
        neg_item_sparse_feat = input['neg_item_sparse_feat'] #dict {feat_name:[B, max_seq_len]}
        neg_item_dense_feat = input['neg_item_dense_feat'] #dict {feat_name:[B, max_seq_len]}
        #正负样本序列长度确保相等，都等于seq_item_length

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

        #neg sequence feat
        neg_seq_i_embed_x = self.item_embed(neg_item_sparse_feat, axis=2) #[B, max_seq_len, item_sparse*embed_dim]
        neg_seq_i_dense_x = self.item_dense(neg_item_dense_feat) #[B, max_seq_len, item_dense]]
        neg_i_x = torch.cat([neg_seq_i_embed_x, neg_seq_i_dense_x], dim=-1) #[B, max_seq_len, item_feat] item_feat=item_sparse*embed_dim+item_dense

        #InterestExtractorLayer
        gru1_out, aux_loss = self.extractor(seq_i_x, seq_item_length, neg_i_x) #gru_out [B, max_seq_len, gru1_hidden_dim]

        #InterestEvolvingLayer
        gru2_out = self.evolving(i_x, gru1_out, seq_item_length) #[B, gru2_hidden_dim]

        #deep
        deep_x = torch.cat([u_x, i_x, gru2_out], dim=-1)  #[B, user_feat+item_feat+gru2_hidden_dim]
        out = self.deep(deep_x).squeeze(1)  #[B]

        #out
        if self.sigmoid_out:
            out = self.sigmoid(out)

        return out, aux_loss

class InterestExtractorLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, aux_dnn_layers=[32,16,1]):
        super(InterestExtractorLayer, self).__init__()
        self.hidden_size = hidden_dim
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)

        assert aux_dnn_layers[-1] == 1, "last hidden dim must be 1"
        self.dnn = DNN(input_dim+hidden_dim, layers=aux_dnn_layers)

    def forward(self, seq_item, seq_len, neg_items):
        '''
        :param seq_item: [B, max_seq_len, input_dim]
        :param seq_len: [B]
        :param negative_items: [B, max_seq_len, input_dim]
        :return:
        '''
        #gru
        batch, max_seq_len, input_dim = seq_item.size()

        #pack_gru_x: data [sum(seq_len), input_dim] batch_sizes=sorted(seq_len), _, _
        pack_gru_x = rnn_utils.pack_padded_sequence(seq_item, seq_len.cpu(), batch_first=True, enforce_sorted=False)
        pack_gru_out, _ = self.gru(pack_gru_x)  #gru_out [sum(seq_len), hidden_dim]
        pad_gru_out, _ = rnn_utils.pad_packed_sequence(pack_gru_out, batch_first=True, padding_value=0.0, total_length=max_seq_len) #pad_gru_out [B, max_seq_len, hidden_dim]

        aux_loss = self._get_aux_loss(pad_gru_out, seq_item, seq_len.unsqueeze(1), neg_items)

        return pad_gru_out, aux_loss

    def _get_aux_loss(self, gru_out, pos_items, seq_len, neg_items):
        '''
        :param gru_out: [B, max_seq_len, hidden_dim]
        :param pos_items: [B, max_seq_len, input_dim]
        :param seq_len: [B, 1]
        :param neg_items: [B, max_seq_len, input_dim]
        :return:
        '''
        batch, max_seq_len, _ = pos_items.size()
        dtype = pos_items.dtype
        device = pos_items.device

        #shift
        pos_items = pos_items[:,1:,:] #[B, max_seq_len-1, input_dim]
        neg_items = neg_items[:,1:,:] #[B, max_seq_len-1, input_dim]
        gru_out = gru_out[:,:-1,:] #[B, max_seq_len-1, hidden_dim]

        #masks
        masks = torch.arange(max_seq_len-1, dtype=dtype, device=device).repeat(batch, 1) #[B, max_seq_len-1]
        masks = masks < (seq_len-1) #[B, max_seq_len-1]

        #flat
        pos_items_flat = pos_items[masks] #[num_feat, input_dim] num_feat=sum(e-1 for e in seq_len)
        neg_items_flat = neg_items[masks] #[num_feat, input_dim]
        gru_out = gru_out[masks] #[num_feat, hidden_dim]
        assert pos_items_flat.size(0)==neg_items_flat.size(0) and neg_items_flat.size(0)==gru_out.size(0), "positive and negative nums does not match"

        #loss
        pos_x = torch.cat(pos_items_flat, gru_out) #[num_feat, input_dim+hidden+dim]
        neg_x = torch.cat(neg_items_flat, gru_out) #[num_feat, input_dim+hidden+dim]

        pos_x = self.dnn(pos_x).squeeze(1) #[num_feat]
        pos_label = torch.ones(pos_x.size(), dtype=dtype, device=device)
        neg_x = self.dnn(neg_x).squeeze(1) #[num_feat]
        neg_label = torch.zeros(neg_x, dtype=dtype, device=device)

        y = torch.cat(pos_x, neg_x)
        label = torch.cat([pos_label, neg_label])

        loss = F.binary_cross_entropy_with_logits(y, label)

        return loss

class InterestEvolvingLayer(nn.Module):
    def __init__(self, item_feat_dim, input_dim, hidden_dim, gru_type='AUGRU'):
        super(InterestEvolvingLayer, self).__init__()

        #Attention
        self.attention = Attention(gru_hidden_dim=input_dim, item_feat_dim=item_feat_dim)

        #GRU
        if gru_type == 'AIGRU':
            self.gru = AIGRU(input_dim=input_dim, hidden_dim=hidden_dim)
        elif gru_type == 'AGRU':
            self.gru = DynamicGRU(input_dim=input_dim, hidden_dim=hidden_dim, gru_type='AGRU')
        elif gru_type == 'AUGRU':
            self.gru = DynamicGRU(input_dim=input_dim, hidden_dim=hidden_dim, gru_type='AUGRU')
        else:
            raise  NotImplementedError("the gru type must in 'AIGRU, AGRU, AUGRU' but get {}".format(gru_type))


    def forward(self, query, keys, seq_len):
        '''
        :param query: [B, item_feat_dim]
        :param keys: [B, max_seq_len, input_dim]
        :param seq_len: [B]
        :return:
        '''
        batch, max_seq_len, input_dim = keys.size()

        #Attention
        att_x = query.unsqueeze(1) #[B, 1, item_feat]
        seq_len_x = seq_len.unsqueeze(1) #[B, 1]
        att_score = self.attention(att_x, keys, seq_len_x) # [B, max_seq_len]

        #GRU
        gru_out = self.gru(keys, att_score, seq_len) #[B, max_seq_len, hidden_dim]

        #get last time hidden
        mask = torch.arange(max_seq_len, device=seq_len.device).repeat(batch, 1) == (seq_len.view(-1, 1) - 1)
        out = gru_out[mask] #[B, hidden_dim]

        return out

class Attention(nn.Module):
    def __init__(self, keys_dim, query_dim):
        super(Attention, self).__init__()

        self.w = nn.Parameter(nn.init.xavier_normal_(torch.empty(keys_dim, query_dim)))

    def forward(self, query, keys, seq_len):
        '''
        :param query: [B, 1, query_dim]
        :param keys:  [B, max_seq_len, keys_dim]
        :param seq_len: [B, 1]
        :return:
        '''
        batch, max_seq_len, keys_dim = keys.size()
        device = keys.device
        dtype = seq_len.dtype

        # masks
        masks = torch.arange(max_seq_len, device=device, dtype=dtype).repeat(batch, 1)  # [B, max_seq_len]
        masks = masks < seq_len  # [B, max_seq_len]

        # weight
        query = query.expand(-1, max_seq_len, -1)  # [B, max_seq_len, query_dim]
        att_x = F.linear(query, self.w, bias=None) #[B, max_seq_len, query_dim] * [query_dim, keys_dim] = [B, max_seq_len, keys_dim]

        #inner product
        att_out = att_x * keys  # [B, max_seq_len, keys_dim]
        att_out = torch.sum(att_out, dim=-1, keepdim=False)  # [B, max_seq_len]

        # softmax
        paddings = torch.ones_like(att_out).to(device) * (-2 ** 32 + 1)  # [B, max_seq_len]
        score = torch.where(masks, att_out, paddings)  # [B, max_seq_len]
        score = F.softmax(score, dim=1)  # [B, max_seq_len]

        return score

class AIGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AIGRU, self).__init__()

        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, gru_x, att_score, seq_len):
        '''
        :param gru_x: [B, max_seq_len, input_dim]
        :param att_score: [B, max_seq_len]
        :param h0 = [B, hidden_dim]
        :return:
        '''
        batch, max_seq_len, input_dim = gru_x.size()
        dtype = gru_x.dtype
        device = gru_x.device

        #core
        att_score = att_score.unsqueeze(2) #[B, max_seq_len, 1]
        gru_x = gru_x * att_score #[B, max_seq_len, input_dim]

        #gru
        h0 = torch.zeros(batch, max_seq_len, self.hidden_size, dtype=dtype, device=device)
        gru_x = rnn_utils.pack_padded_sequence(gru_x, seq_len.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(gru_x, h0)
        out, _ = rnn_utils.pad_packed_sequence(out, batch_first=True, padding_value=0.0, total_length=gru_x.size(1)) #out [B, max_seq_len, hidden_dim]

        return out

class DynamicGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, gru_type='AUGRU'):
        super(DynamicGRU, self).__init__()
        self.hidden_size = hidden_dim
        if gru_type == 'AGRU':
            self.gru = AGRU(input_dim, hidden_dim)
        elif gru_type == 'AUGRU':
            self.gru = AUGRU(input_dim, hidden_dim)
        else:
            raise NotImplementedError("the gru type must in 'AGRU, AUGRU' but get {}".format(gru_type))

    def forward(self, gru_x, att_score, seq_len):
        '''
        :param gru_x: [B, max_seq_len, input_dim]
        :param att_score: [B, max_seq_len]
        :param seq_len: [B]
        :return:
        '''
        #data [sum(seq_len), input_dim]
        #batch_sizes sorted(seq_len)
        pack_data, batch_sizes, _, _ = rnn_utils.pack_padded_sequence(gru_x, seq_len.cpu(), batch_first=True, enforce_sorted=False)

        #sorce [sum(seq_len)]
        pack_score, _, _, _ = rnn_utils.pack_padded_sequence(att_score, seq_len.cpu(), batch_first=True, enforce_sorted=False) #out [B, max_seq_len, hidden_dim]

        #gru init
        h0 = torch.zeros(batch_sizes[0], self.hidden_size)

        #output
        #pack_out [sum(seq_len), hidden_dim]
        pack_out = torch.zeros(pack_data.size(0), self.hidden_size, dtype=pack_data.dtype, device=pack_data.device)

        #times
        begin = 0
        ht = h0
        for batch in batch_sizes:
            i_x = pack_data[begin:begin+batch] #[batch, input_dim]
            s_x = pack_score[begin:begin+batch] #[batch]
            h_x = ht[:batch] #[batch, hidden_dim]

            h_t_plus_1 = self.gru(i_x, s_x, h_x) #[batch, hidden_dim]

            pack_out[begin:begin+batch] = h_t_plus_1
            ht = h_t_plus_1
            begin += batch

        out, _ = rnn_utils.pad_packed_sequence(pack_out, batch_first=True, padding_value=0.0, total_length=gru_x.size(1)) #out [B, max_seq_len, hidden_dim]

        return out

class AGRU(nn.Module):
    '''
    Attention based GRU
    '''
    def __init__(self, input_dim, hidden_dim):
        super(AGRU, self).__init__()
        #Wr, Wh
        self.w = nn.Parameter(nn.init.xavier_normal_(torch.empty(2*hidden_dim, input_dim)))
        #Ur, Uh
        self.u = nn.Parameter(nn.init.xavier_normal_(torch.empty(2*hidden_dim, hidden_dim)))
        #Br
        self.br = nn.Parameter(nn.init.xavier_normal_(torch.empty(hidden_dim)))
        #Bh
        self.bh = nn.Parameter(nn.init.xavier_normal_(torch.empty(hidden_dim)))


    def forward(self, x, score, ht):
        '''
        :param x: [B, input_dim]
        :param score: [B]
        :param ht = [B, hidden_dim]
        :return:
        '''
        i_x = F.linear(x, self.w, bias=None) #[B, 2*hidden_dim]
        h_x = F.linear(ht, self.u, bias=None) #[B, 2*hidden_dim]
        i_r, i_z = i_x.chunk(chunks=2, dim=1) #i_r [B, hidden_dim], i_z [B, hidden_dim]
        h_r, h_z = h_x.chunk(chunks=2, dim=1) #h_r [B, hidden_dim], h_z [B, hidden_dim]

        r = F.sigmoid(i_r + h_r + self.br) #[B, hidden_dim]
        z = F.tanh(i_z + h_z*r + self.bh) #[B, hidden_dim]

        a = score.unsqueeze(1) #[B, 1]
        out = (1-a) * ht + a * z #[B, hidden_dim]

        return out

class AUGRU(nn.Module):
    '''
    GRU with attentional update gate
    '''
    def __init__(self, input_dim, hidden_dim):
        super(AUGRU, self).__init__()
        #Wu, Wr, Wh
        self.w = nn.Parameter(nn.init.xavier_normal_(torch.empty(3*hidden_dim, input_dim)))
        #Uu, Ur, Uh
        self.u = nn.Parameter(nn.init.xavier_normal_(torch.empty(3*hidden_dim, hidden_dim)))
        #Bu
        self.bu = nn.Parameter(nn.init.xavier_normal_(torch.empty(hidden_dim)))
        #Br
        self.br = nn.Parameter(nn.init.xavier_normal_(torch.empty(hidden_dim)))
        #Bh
        self.bh = nn.Parameter(nn.init.xavier_normal_(torch.empty(hidden_dim)))

    def forward(self, x, score, ht):
        '''
        :param x: [B, input_dim]
        :param att_score: [B]
        :param ht = [B, hidden_dim]
        :return:
        '''
        i_x = F.linear(x, self.w, bias=None) #[B, 3*hidden_dim]
        h_x = F.linear(ht, self.u, bias=None) #[B, 3*hidden_dim]
        i_u, i_r, i_z = i_x.chunk(chunks=3, dim=1) #i_u [B, hidden_dim], i_r [B, hidden_dim], i_z [B, hidden_dim]
        h_u, h_r, h_z = h_x.chunk(chunks=3, dim=1) #h_u [B, hidden_dim], h_r [B, hidden_dim], h_z [B, hidden_dim]

        a = score.unsqueeze(1)  # [B, 1]
        u = F.sigmoid(i_u + h_u + self.bu) #[B, hidden_dim]
        u = a * u #[B, hidden_dim]

        r = F.sigmoid(i_r + h_r + self.br) #[B, hidden_dim]
        z = F.tanh(i_z + h_z*r + self.bh) #[B, hidden_dim]

        out = (1-u) * ht + u * z #[B, hidden_dim]

        return out
