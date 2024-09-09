import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from layers.cross_encoder import Encoder
from layers.cross_decoder import Decoder
from layers.SelfAttention_Family import FullAttention, AttentionLayer, TwoStageAttentionLayer
from layers.Embed import DSW_embedding

from math import ceil

class Model(nn.Module):
    def __init__(self, configs,train=True):
        super(Model, self).__init__()
        self.data_dim = configs.enc_in
        self.in_len = configs.seq_len
        self.out_len = configs.pred_len
        self.seg_len = 24
        self.merge_win = 4
        self.flag=train
        self.baseline = False
        self.pred_len = configs.pred_len
        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * self.in_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(1.0 * self.out_len / self.seg_len) * self.seg_len
        self.in_len_add = self.pad_in_len - self.in_len

        # Embedding
        self.enc_value_embedding = DSW_embedding(self.seg_len, configs.d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, self.data_dim, (self.pad_in_len // self.seg_len), configs.d_model))
        self.pre_norm = nn.LayerNorm(configs.d_model)

        # Encoder
        self.encoder = Encoder(configs.e_layers, self.merge_win, configs.d_model, configs.n_heads, configs.d_ff, block_depth = 1, \
                                    dropout = configs.dropout,in_seg_num = (self.pad_in_len // self.seg_len), factor = 3)
        
        # Decoder
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, self.data_dim, (self.pad_out_len // self.seg_len), configs.d_model))
        self.decoder = Decoder(self.seg_len, configs.e_layers + 1, configs.d_model, configs.n_heads, configs.d_ff, configs.dropout, \
                                    out_seg_num = (self.pad_out_len // self.seg_len), factor = 3)
        
    def forward(self, x_enc):
        if self.flag:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
            _, _, N = x_enc.shape

        if (self.baseline):
            base = x_enc.mean(dim = 1, keepdim = True)
        else:
            base = 0
        batch_size = x_enc.shape[0]
        if (self.in_len_add != 0):
            x_enc = torch.cat((x_enc[:, :1, :].expand(-1, self.in_len_add, -1), x_enc), dim = 1)

        x_enc = self.enc_value_embedding(x_enc)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        
        enc_out = self.encoder(x_enc)

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat = batch_size)
        predict_y = self.decoder(dec_in, enc_out)

        dec_out = base + predict_y[:, :self.out_len, :]

        if self.flag:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out[:, -self.pred_len:, :]