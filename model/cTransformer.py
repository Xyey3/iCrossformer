import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np

class TimesNet_Block(nn.Module):
    def __init__(self, d_model, num_channels, dropout=0.1):
        super(TimesNet_Block, self).__init__()
        self.d_model = d_model
        self.num_channels = num_channels
        # 定义两个CNN层
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=num_channels, out_channels=d_model, kernel_size=3, padding=1)
        # 残差连接
        self.residual_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        # 定义一个全连接层
        self.fc = nn.Linear(d_model, d_model)
        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Dropout
        self.dropout = nn.Dropout(dropout)
        # BatchNorm
        self.batch_norm = nn.BatchNorm1d(d_model)

    def forward(self, x):
        # x 的形状是 [batch_size, seq_len, d_model]
        # 使用 CNN 提取特征
        x = x.permute(0, 2, 1)  # [batch_size, d_model, seq_len]
        residual = self.residual_conv(x)  # [batch_size, d_model, seq_len]
        x = F.gelu(self.conv1(x))  # [batch_size, num_channels, seq_len]
        x = self.conv2(x)  # [batch_size, d_model, seq_len]
        x += residual  # 残差连接
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, d_model]
        x = self.norm1(x)  # [batch_size, seq_len, d_model]
        # 通过全连接层
        x = self.fc(x)  # [batch_size, seq_len, d_model]
        x = self.batch_norm(x.permute(0, 2, 1)).permute(0, 2, 1)  # BatchNorm
        x = F.gelu(x)
        x = self.dropout(x)  # Dropout
        x = self.norm2(x)  # [batch_size, seq_len, d_model] #
        return x

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.decoder = TimesNet_Block(configs.d_model, configs.d_model//2)
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.decoder(enc_out)
        
        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]