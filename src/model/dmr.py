# -*- coding: utf-8 -*-      
#--------------------------------------------
# Author:chenhao
# Date:2021-01-08 11:37
# Description:  
#--------------------------------------------
import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from src.model.base import DNN, SparseEmbeddingLayer, DenseFeatCatLayer

class DMR(nn.Module):
    '''
    Deep Match to Rank Model for Personalized Click-Through Rate Prediction
    '''
    def __init__(self, user_sparse_and_nums,
                 user_dense,
                 item_sparse_and_nums,
                 item_dense,
                 uer_embed_dim,
                 target_item_embed_dim,
                 seq_item_embed_dim,
                 deep_layers,
                 max_seq_len,
                 position_embed_dim,
                 attention_hidden_dim,
                 is_share_embed=False,
                 sigmoid_out=False):
        super(DMR, self).__init__()
        self.is_share_embed = is_share_embed

        #user embed
        self.user_feat_num = len(user_sparse_and_nums) * uer_embed_dim + len(user_dense)
        self.user_embed = SparseEmbeddingLayer(feat_and_nums=user_sparse_and_nums, embed_dim=uer_embed_dim)
        self.user_dense = DenseFeatCatLayer()

        #item embed
        self.target_item_feat_num = len(item_sparse_and_nums) * target_item_embed_dim + len(item_dense)
        self.target_item_embed = SparseEmbeddingLayer(feat_and_nums=item_sparse_and_nums, embed_dim=target_item_embed_dim)
        self.target_item_dense = DenseFeatCatLayer()
        if not self.is_share_embed:
            self.seq_item_feat_num = len(item_sparse_and_nums) * seq_item_embed_dim + len(item_dense)
            self.seq_item_embed = SparseEmbeddingLayer(feat_and_nums=item_sparse_and_nums, embed_dim=seq_item_embed_dim)
            self.seq_item_dense = DenseFeatCatLayer()
        else:
            assert seq_item_embed_dim == target_item_embed_dim, 'if shared embedding, seq_item_embed_dim num be equal target_item_embed_dim'
            self.seq_item_feat_num = self.target_item_feat_num
            self.seq_item_embed = self.target_item_embed
            self.seq_item_dense = self.target_item_dense

        #seq position embed
        self.pos_embed = nn.Parameter(nn.init.xavier_normal_(torch.empty(max_seq_len, position_embed_dim)))

        #User-to-Item Network
        self.u2i = User2Item(target_item_feat_dim=target_item_embed_dim,
                             seq_item_feat_dim=seq_item_embed_dim,
                             position_feat_dim=position_embed_dim,
                             attention_hidden_dim=attention_hidden_dim)

        #Item-to-Item Network
        self.i2i = Item2Item(target_item_feat_dim=target_item_embed_dim,
                             seq_item_feat_dim=seq_item_embed_dim,
                             position_feat_dim=position_embed_dim,
                             attention_hidden_dim=attention_hidden_dim)

        #Deep
        assert deep_layers[-1] == 1, "last hidden dim must be 1"
        self.deep_input_dim = self.user_feat_num + self.target_item_feat_num + self.seq_item_feat_num + 1 + 1
        self.deep = DNN(input_dim=self.deep_input_dim, layers=deep_layers, act='prelu')

        #output
        self.sigmoid_out = sigmoid_out
        if self.sigmoid_out:
            self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        user_sparse_feat = input['user_sparse'] #dict {feat_name:[B]}
        user_dense_feat = input['user_dense'] #dict {feat_name:[B]}

        target_item_sparse_feat = input['target_item_sparse'] #dict {feat_name:[B]}
        target_item_dense_feat = input['target_item_dense'] #dict {feat_name:[B]}
        pos_item_sparse_feat = input['pos_item_sparse'] #dict {feat_name:[B]}
        pos_item_dense_feat = input['pos_item_dense'] #dict {feat_name:[B]}
        neg_item_sparse_feat = input['neg_item_sparse'] #dict {feat_name:[B, k]}
        neg_item_dense_feat = input['neg_item_dense'] #dict {feat_name:[B, k]}

        seq_item_sparse_feat = input['seq_item_sparse'] #dict {feat_name:[B, max_seq_len]}
        seq_item_dense_feat = input['seq_item_dense'] #dict {feat_name:[B, max_seq_len]}
        seq_item_length = input['seq_item_length'] # tensor[B]

        #user feat
        u_embed_x = self.user_embed(user_sparse_feat, axis=2) #[B, user_sparse*embed_dim]
        u_dense_x = self.user_dense(user_dense_feat) #[B, user_dense]
        u_x = torch.cat([u_embed_x, u_dense_x], dim=1) #[B, self.user_feat_num]

        #target item feat
        target_i_embed_x = self.target_item_embed(target_item_sparse_feat, axis=2)#[B, item_sparse*target_item_embed_dim]
        target_i_dense_x = self.target_item_dense(target_item_dense_feat) #[B, item_dense]
        target_i_x = torch.cat([target_i_embed_x, target_i_dense_x], dim=1) #[B, self.target_item_feat_num]

        #pos item feat
        pos_i_embed_x = self.target_item_embed(pos_item_sparse_feat, axis=2)#[B, item_sparse*target_item_embed_dim]
        pos_i_dense_x = self.target_item_embed(pos_item_dense_feat) #[B, item_dense]
        pos_i_x = torch.cat([pos_i_embed_x, pos_i_dense_x], dim=1) #[B, self.target_item_feat_num]

        #neg item feat
        neg_i_embed_x = self.target_item_embed(neg_item_sparse_feat, axis=2)#[B, k, item_sparse*target_item_embed_dim]
        neg_i_dense_x = self.target_item_embed(neg_item_dense_feat) #[B, k, item_dense]
        neg_i_x = torch.cat([neg_i_embed_x, neg_i_dense_x], dim=-1) #[B, k, self.target_item_feat_num]

        #sequence item feat
        seq_i_embed_x = self.seq_item_embed(seq_item_sparse_feat, axis=2) #[B, max_seq_len, item_sparse*seq_item_embed_dim]
        seq_i_dense_x = self.seq_item_dense(seq_item_dense_feat) #[B, max_seq_len, item_dense]]
        seq_i_x = torch.cat([seq_i_embed_x, seq_i_dense_x], dim=-1) #[B, max_seq_len, self.seq_item_feat_num]

        seq_position_x = self.pos_embed.repeat(seq_i_x.size(0), 1, 1) #[B, max_seq_len, position_embed_dim]

        #User-to-Item Network
        r_u2i, aux_loss = self.u2i(seq_i_x, seq_position_x, seq_item_length, target_i_x, pos_i_x, neg_i_x) #r_u2i [B, 1] aux_loss [1]

        # Item-to-Item Network
        r_i2i, u_i2i = self.i2i(seq_i_x, seq_position_x, seq_item_length, target_i_x) #r_i2i [B, 1], u_i2i [B, self.seq_item_feat_num]

        #deep
        deep_x = torch.cat([u_x, target_i_x, u_i2i, r_u2i, r_i2i], dim=-1)  #[B, self.deep_input_dim]
        out = self.deep(deep_x).squeeze(1)  #[B]

        #out
        if self.sigmoid_out:
            out = self.sigmoid(out)

        return out, aux_loss

class User2Item(nn.Module):
    def __init__(self, target_item_feat_dim, seq_item_feat_dim, position_feat_dim, attention_hidden_dim):
        super(User2Item, self).__init__()

        #Attention
        self.att = U2IAttention(seq_item_feat_dim, position_feat_dim, attention_hidden_dim, target_item_feat_dim)

    def forward(self, seq_items, seq_position, seq_len, target_item, pos_item, neg_items):
        '''
        :param seq_items: [B, max_seq_len, seq_item_feat_dim]
        :param seq_position: [B, max_seq_len, position_feat_dim]
        :param seq_len: [B]
        :param target_item [B, target_item_feat_dim]
        :param pos_item: [B, target_item_feat_dim]
        :param neg_items: [B, k, target_item_feat_dim]
        :return:
        '''

        #Attention
        u_t, u_t_dif_1 = self.att(seq_items, seq_position, seq_len) #u_t:[B, target_item_feat_dim]]  u_t_dif_1:[B, target_item_feat_dim]]

        out = torch.sum(u_t * target_item, dim=1, keepdim=True) #[B, 1]

        #aux_loss
        aux_loss = self._get_aux_loss(pos_item, neg_items, u_t_dif_1) #[1]

        return out, aux_loss

    def _get_aux_loss(self, pos_item, neg_items, u_t_dif_1):
        """
        :param pos_item: [B, target_item_feat_dim]
        :param neg_items: [B, k, target_item_feat_dim]
        :param u_t_dif_1: [B, target_item_feat_dim]]
        :return:
        """

        #pos
        pos_x = torch.sum(pos_item * u_t_dif_1, dim=1, keepdim=False) #[B]
        pos_label = torch.ones(pos_x.size(), dtype=pos_x.dtype, device=pos_x.device) #[B]

        #neg
        neg_x = torch.sum(neg_items * u_t_dif_1.unsqueeze(1), dim=-1, keepdim=False).flatten() #[B*k]
        neg_label = torch.zeros(neg_x.size(), dtype=neg_x.dtype, device=neg_x.device) #[B*k]

        y = torch.cat([pos_x, neg_x]) #[B+B*k]
        label = torch.cat([pos_label, neg_label]) #[B+B*k]

        loss = F.binary_cross_entropy_with_logits(y, label) #[1]

        return loss

class U2IAttention(nn.Module):
    def __init__(self, e_dim, p_dim, h_dim, c_dim):
        super(U2IAttention, self).__init__()
        self.Wp = nn.Parameter(nn.init.xavier_normal_(torch.empty(p_dim, h_dim)))
        self.We = nn.Parameter(nn.init.xavier_normal_(torch.empty(e_dim, h_dim)))
        self.b = nn.Parameter(torch.zeros(h_dim))
        self.z = nn.Parameter(torch.ones(h_dim))
        self.g = nn.Sequential(*[nn.Linear(e_dim, c_dim), nn.PReLU(), nn.Dropout()])

    def forward(self, e, p, seq_len):
        '''
        :param e: [B, max_seq_len, e_dim]
        :param p: [B, max_seq_len, p_dim]
        :param seq_len: [B]
        :return:
        '''
        batch, max_seq_len, e_dim = e.size()
        dtype = e.dtype
        device = e.device

        out_e = torch.matmul(e, self.We) #[B, max_seq_len, h_dim]
        out_p = torch.matmul(p, self.Wp) #[B, max_seq_len, h_dim]
        a = torch.sum(F.tanh(out_p + out_e + self.b) * self.z, dim=-1, keepdim=False) #[B, max_seq_len]

        #t_masks
        masks = torch.arange(max_seq_len, dtype=dtype, device=device).repeat(batch, 1) #[B, max_seq_len]
        masks = masks < seq_len.unsqueeze(1) #[B, max_seq_len]

        #t_score
        paddings = torch.ones_like(a).to(device) * (-2 ** 32 + 1) #[B, max_seq_len]
        score = torch.where(masks, a, paddings) #[B, max_seq_len]
        score = F.softmax(score, dim=1) #[B, max_seq_len]

        #t_weight sum
        u_t = torch.matmul(score.unsequeeze(1), e) #[B, 1, max_seq_len] x [B, max_seq_len, e_dim] = [B, 1, e_dim]
        u_t = self.g(u_t.sequeeze(1)) #[B, c_dim]

        #t_dif_1_masks
        masks_t_dif_1 = masks < (seq_len-1).unsqueeze(1) #[B, max_seq_len]

        #t_dif_1_score
        score_t_dif_1 = torch.where(masks_t_dif_1, a, paddings) #[B, max_seq_len]
        score_t_dif_1 = F.softmax(score_t_dif_1, dim=1) #[B, max_seq_len]

        #t_dif_1_weight sum
        u_t_dif_1 = torch.matmul(score_t_dif_1.unsequeeze(1), e) #[B, 1, max_seq_len] x [B, max_seq_len, e_dim] = [B, 1, e_dim]
        u_t_dif_1 = self.g(u_t_dif_1.sequeeze(1)) #[B, c_dim]

        return u_t, u_t_dif_1

class Item2Item(nn.Module):
    def __init__(self, target_item_feat_dim, seq_item_feat_dim, position_feat_dim, attention_hidden_dim):
        super(Item2Item, self).__init__()

        #Attention
        self.att = I2IAttention(seq_item_feat_dim, position_feat_dim, attention_hidden_dim, target_item_feat_dim)

    def forward(self, seq_items, seq_pos, seq_len, target_item):
        '''
        :param seq_items: [B, max_seq_len, seq_item_feat_dim]
        :param seq_pos: [B, max_seq_len, position_feat_dim]
        :param seq_len: [B]
        :param target_item [B, target_item_feat_dim]
        :param pos_item: [B, target_item_feat_dim]
        :param neg_items: [B, k, target_item_feat_dim]
        :return:
        '''
        u, r = self.att(target_item, seq_items, seq_pos, seq_len)

        return u, r

class I2IAttention(nn.Module):
    def __init__(self, e_dim, p_dim, h_dim, c_dim):
        super(I2IAttention, self).__init__()
        self.Wp = nn.Parameter(nn.init.xavier_normal_(torch.empty(p_dim, h_dim)))
        self.We = nn.Parameter(nn.init.xavier_normal_(torch.empty(e_dim, h_dim)))
        self.Wc = nn.Parameter(nn.init.xavier_normal_(torch.empty(c_dim, h_dim)))
        self.b = nn.Parameter(torch.zeros(h_dim))
        self.z = nn.Parameter(torch.ones(h_dim))
        self.g = nn.Sequential(*[nn.Linear(e_dim, c_dim), nn.PReLU(), nn.Dropout()])

    def forward(self, c, e, p, seq_len):
        '''
        :param c: [B, c_dim]
        :param e: [B, max_seq_len, e_dim]
        :param p: [B, max_seq_len, p_dim]
        :param seq_len: [B]
        :return:
        '''
        batch, max_seq_len, e_dim = e.size()
        dtype = e.dtype
        device = e.device

        out_c = torch.matmul(c, self.Wc) #[B, h_dim]
        out_e = torch.matmul(e, self.We) #[B, max_seq_len, h_dim]
        out_p = torch.matmul(p, self.Wp) #[B, max_seq_len, h_dim]
        a = torch.sum(F.tanh(out_c.unsqueeze(1) + out_p + out_e + self.b) * self.z, dim=-1, keepdim=False) #[B, max_seq_len]

        #masks
        masks = torch.arange(max_seq_len, dtype=dtype, device=device).repeat(batch, 1) #[B, max_seq_len]
        masks = masks < seq_len.unsqueeze(1) #[B, max_seq_len]

        #score
        paddings = torch.ones_like(a).to(device) * (-2 ** 32 + 1) #[B, max_seq_len]
        score = torch.where(masks, a, paddings) #[B, max_seq_len]
        score = F.softmax(score, dim=1) #[B, max_seq_len]

        #weight sum
        u = torch.matmul(score.unsequeeze(1), e) #[B, 1, max_seq_len] x [B, max_seq_len, e_dim] = [B, 1, e_dim]

        #r
        paddings_r = torch.zeros_like(a).to(device) #[B, max_seq_len]
        r = torch.where(masks, a, paddings_r) #[B, max_seq_len]
        r = torch.sum(r, dim=1, keepdim=True) #[B, 1]

        return u, r
