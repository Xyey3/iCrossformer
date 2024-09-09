import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange, repeat
from layers.SelfAttention_Family import FullAttention, AttentionLayer, TwoStageAttentionLayer

class DecoderLayer(nn.Module):
    '''
    The decoder layer of Crossformer, each layer will make a prediction at its scale
    '''
    def __init__(self, seg_len, d_model, n_heads, d_ff=None, dropout=0.1, out_seg_num = 10, factor = 10):
        super(DecoderLayer, self).__init__()
        self.self_attention = TwoStageAttentionLayer(out_seg_num, factor, d_model, n_heads, \
                                d_ff, dropout)    
        self.cross_attention = AttentionLayer(FullAttention(False, attention_dropout=dropout), d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_model),
                                nn.GELU(),
                                nn.Linear(d_model, d_model))
        self.linear_pred = nn.Linear(d_model, seg_len)

    def forward(self, x, cross):
        '''
        x: the output of last decoder layer
        cross: the output of the corresponding encoder layer
        '''

        batch = x.shape[0]
        x = self.self_attention(x)

        # x = rearrange(x, 'b ts_d out_seg_num d_model -> (b ts_d) out_seg_num d_model')
        # cross = rearrange(cross, 'b ts_d in_seg_num d_model -> (b ts_d) in_seg_num d_model')
        # 给定的值
        b = x.shape[0]
        ts_d = x.shape[1]
        out_seg_num = x.shape[2]
        d_model = x.shape[3]
        in_seg_num = cross.shape[2]
        # 对 x 的操作
        # 步骤 1：重塑形状为 (b * ts_d, out_seg_num, d_model)
        x = x.reshape(b * ts_d, out_seg_num, d_model)
        # 对 cross 的操作
        # 步骤 1：重塑形状为 (b * ts_d, in_seg_num, d_model)
        cross = cross.reshape(b * ts_d, in_seg_num, d_model)


        tmp, _ = self.cross_attention(
            x, cross, cross,None
        )
        x = x + self.dropout(tmp)
        y = x = self.norm1(x)
        y = self.MLP1(y)
        dec_output = self.norm2(x+y)
        
        # dec_output = rearrange(dec_output, '(b ts_d) seg_dec_num d_model -> b ts_d seg_dec_num d_model', b = batch)
        # 给定的值
        b = batch
        ts_d = dec_output.shape[0] // b
        seg_dec_num = dec_output.shape[1]
        d_model = dec_output.shape[2]
        # 步骤 1：重塑形状为 (b, ts_d, seg_dec_num, d_model)
        dec_output = dec_output.reshape(b, ts_d, seg_dec_num, d_model)


        layer_predict = self.linear_pred(dec_output)

        # layer_predict = rearrange(layer_predict, 'b out_d seg_num seg_len -> b (out_d seg_num) seg_len')
        # 给定的值
        b, out_d, seg_num, seg_len = layer_predict.shape
        # 步骤 1：重塑形状为 (b, out_d * seg_num, seg_len)
        layer_predict = layer_predict.reshape(b, out_d * seg_num, seg_len)

        return dec_output, layer_predict

class Decoder(nn.Module):
    '''
    The decoder of Crossformer, making the final prediction by adding up predictions at each scale
    '''
    def __init__(self, seg_len, d_layers, d_model, n_heads, d_ff, dropout,\
                router=False, out_seg_num = 10, factor=10):
        super(Decoder, self).__init__()

        self.router = router
        self.decode_layers = nn.ModuleList()
        for i in range(d_layers):
            self.decode_layers.append(DecoderLayer(seg_len, d_model, n_heads, d_ff, dropout, \
                                        out_seg_num, factor))

    def forward(self, x, cross):
        final_predict = None
        i = 0

        ts_d = x.shape[1]
        for layer in self.decode_layers:
            cross_enc = cross[i]
            x, layer_predict = layer(x,  cross_enc)
            if final_predict is None:
                final_predict = layer_predict
            else:
                final_predict = final_predict + layer_predict
            i += 1
        
        # final_predict = rearrange(final_predict, 'b (out_d seg_num) seg_len -> b (seg_num seg_len) out_d', out_d = ts_d)
        # 给定的值
        b = final_predict.shape[0]
        out_d = ts_d
        seg_num = final_predict.shape[1] // out_d
        seg_len = final_predict.shape[2]
        # 步骤 1：重塑形状为 (b, seg_num, seg_len, out_d)
        final_predict = final_predict.reshape(b, seg_num, seg_len, out_d)
        # 步骤 2：调整维度顺序为 (b, (seg_num * seg_len), out_d)
        final_predict = final_predict.permute(0, 2, 1, 3).reshape(b, seg_num * seg_len, out_d)

        return final_predict

