import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, CrossEncLayer
from layers.SelfAttention_Family import TwoStageAttentionLayer
from layers.Embed import DSW_embedding
from einops import rearrange, repeat
import numpy as np

class ResNet_Block(nn.Module):
    def __init__(self, d_model, num_channels, dropout=0.1):
        super(ResNet_Block, self).__init__()
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
        self.enc_embedding = DSW_embedding(configs.seq_len // 3, configs.d_model, configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                CrossEncLayer(
                    TwoStageAttentionLayer(3, 10, configs.d_model, configs.n_heads, configs.d_ff, configs.dropout),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.decoder = ResNet_Block(3 * configs.d_model, configs.d_model // 2)
        self.projection = nn.Linear(3 * configs.d_model, configs.pred_len, bias=True)

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
        enc_out = self.enc_embedding(x_enc)
        # [B, N, K, E]

        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # [B, N, K, E]

        dec_out=rearrange(enc_out, 'batch_size Data_dim Seg_num d_model -> batch_size Data_dim (Seg_num d_model)')
        # [B, N, K*E] 

        dec_out = self.decoder(dec_out)
        # [B, N, K*E]

        dec_out=self.projection(dec_out).transpose(1,2)
        # [B, N, S] 
        # [B, S, N] 

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]