import torch
import torch.nn as nn
from math import sqrt
import numpy as np

from functools import partial
from einops import rearrange, repeat


class MultiHeadsAttention(nn.Module):
    '''
    The Attention operation
    '''

    def __init__(self, scale=None, attention_dropout=0.1, returnA=False):
        super(MultiHeadsAttention, self).__init__()
        self.scale = scale
        self.returnA = returnA
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.returnA:
            return V.contiguous(), A.contiguous()
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, d_keys=None, d_values=None, mix=True, dropout=0.1, returnA=False, att_type='full'):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (hid_dim//n_heads)
        d_values = d_values or (hid_dim//n_heads)

        if att_type == 'full' or att_type == 'proxy':
            self.inner_attention = MultiHeadsAttention(
                scale=None, attention_dropout=dropout, returnA=returnA)
        # elif att_type=='prob':
        #     self.inner_attention = ProbAttention(False, prob_factor, attention_dropout=dropout, output_attention=returnA)
        self.query_projection = nn.Linear(hid_dim, d_keys * n_heads)
        self.key_projection = nn.Linear(hid_dim, d_keys * n_heads)
        self.value_projection = nn.Linear(hid_dim, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, hid_dim)
        self.n_heads = n_heads
        self.returnA = returnA
        self.mix = mix

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        keys = keys.view(B, S, H, -1)
        values = values.view(B, S, H, -1)

        out, A = self.inner_attention(
            queries,
            keys,
            values,
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)
        out = self.out_projection(out)
        if self.returnA:
            return out, A
        else:
            return out, None


class Full_EncoderLayer(nn.Module):
    '''
    input shape: [batch_size, T, N, hid_dim]
    output shape: [batch_size, N, T, hid_dim]
    '''

    def __init__(self, hid_dim, n_heads, att_type, d_ff=None, dropout=0.1, att_dropout=0.1, return_att=False, activation='gelu'):
        super().__init__()
        d_ff = d_ff or 4*hid_dim
        self.return_att = return_att
        self.att_layer = AttentionLayer(
            hid_dim, n_heads, dropout=att_dropout, att_type=att_type, returnA=return_att)
        self.dropout = nn.Dropout(dropout)
        assert activation in ['gelu', 'relu', 'GLU']
        activation_func_dict = {'gelu': nn.GELU(),
                                'relu': nn.ReLU(),
                                'GLU': nn.GLU()}
        if activation == 'GLU':
            d_ff1 = d_ff*2
        else:
            d_ff1 = d_ff
        self.MLP1 = nn.Sequential(nn.Linear(hid_dim, d_ff1),
                                  activation_func_dict[activation],
                                  nn.Linear(d_ff, hid_dim))

    def forward(self, data):  # data:BTNC
        batch = data.shape[0]
        T = data.shape[1]
        now = data[:, -1, :, :]
        query = repeat(now, 'b n c -> (b t) n c', t=T)
        data = rearrange(data, 'b t n c-> (b t) n c')
        x = data
        x, A = self.att_layer(query, x, x)
        x = data + self.dropout(x)
        x = x + self.dropout(self.MLP1(x))
        final_out = rearrange(x, '(b T) N d -> b T N d', b=batch)
        if self.return_att:
            A = rearrange(A, '(b t) h l s -> b t h l s', b=batch)
            return final_out, A
        return final_out, None


class EncoderLayer(nn.Module):
    '''
    input shape: [batch_size, T, N, hid_dim]
    output shape: [batch_size, T, N, hid_dim]
    '''

    def __init__(self, factor, hid_dim, n_heads, num_nodes=None,
                 d_ff=None, dropout=0.1, att_dropout=0.1, activation='gelu', return_att=False):
        super().__init__()
        d_ff = d_ff or 4*hid_dim
        self.return_att = return_att

        assert num_nodes is not None
        self.readout_fc = nn.Linear(num_nodes, factor)

        self.node2proxy = AttentionLayer(
            hid_dim, n_heads, dropout=att_dropout, att_type='proxy', returnA=return_att)
        self.proxy2node = AttentionLayer(
            hid_dim, n_heads, dropout=att_dropout, att_type='proxy', returnA=return_att)

        self.dropout = nn.Dropout(dropout)
        assert activation in ['gelu', 'relu', 'GLU']
        activation_func_dict = {'gelu': nn.GELU(),
                                'relu': nn.ReLU(),
                                'GLU': nn.GLU()}
        if activation == 'GLU':
            d_ff1 = d_ff*2
        else:
            d_ff1 = d_ff
        self.MLP2 = nn.Sequential(nn.Linear(hid_dim, d_ff1),
                                  activation_func_dict[activation],
                                  nn.Linear(d_ff, hid_dim))

    def forward(self, data):  # data:BTNC
        now = data[:, -1, :, :]
        batch = data.shape[0]
        T = data.shape[1]

        now = data[:, -1, :, :]
        temp = rearrange(now, 'b n c -> b c n')
        z_proxy = self.readout_fc(temp)
        z_proxy = repeat(z_proxy, 'b c k -> (b repeat) c k', repeat=T)
        z_proxy = rearrange(z_proxy, 'bt c K -> bt K c')

        data = rearrange(data, 'b t n c-> (b t) n c')

        proxy_feature, A1 = self.node2proxy(z_proxy, data, data)
        node_feature, A2 = self.proxy2node(data, proxy_feature, proxy_feature)
        enc_feature = data + self.dropout(node_feature)
        enc_feature = enc_feature + self.dropout(self.MLP2(enc_feature))

        final_out = rearrange(
            enc_feature, '(b T) N hid_dim -> b T N hid_dim', b=batch)
        if self.return_att:
            A1 = rearrange(A1, '(b t) h l s -> b t h l s', b=batch)
            A2 = rearrange(A2, '(b t) h l s -> b t h l s', b=batch)
            return final_out, [A1, A2]
        else:
            return final_out, None


class time_wise_predictor(nn.Module):
    def __init__(self, num_nodes, input_length, predict_length, in_dim, pre_dim,
                 num_of_filters=128, activation='gelu'):
        super(time_wise_predictor, self).__init__()
        self.num_nodes = num_nodes
        self.input_length = input_length
        self.in_dim = in_dim
        self.pre_dim = pre_dim
        self.predict_length = predict_length
        assert activation in ['gelu', 'relu']
        self.predict_unit = nn.Sequential(nn.Linear(input_length*in_dim, num_of_filters),
                                          nn.GELU() if activation == 'gelu' else nn.ReLU())
        self.predict_unit_list = nn.ModuleList(
            [nn.Linear(num_of_filters, pre_dim) for _ in range(predict_length)])

    def forward(self, data):
        data = rearrange(data, 'b t n c -> b n (t c)')
        data = self.predict_unit(data)
        need_concat = []
        for i in range(self.predict_length):
            unit_out = self.predict_unit_list[i](data)
            unit_out = rearrange(unit_out, 'b n c -> b 1 n c')
            need_concat.append(unit_out)
        final_out = torch.cat(need_concat, dim=1)
        return final_out


class ST_Embedding(nn.Module):
    def __init__(self, hid_dim, num_nodes, slice_size_per_day, dropout=0.1, hasTemb=True, hasSemb=True):
        super(ST_Embedding, self).__init__()
        self.hasTemb = hasTemb
        self.hasSemb = hasSemb
        if hasTemb:
            self.time_in_day_embedding = nn.Embedding(
                slice_size_per_day, hid_dim)
            self.day_in_week_embedding = nn.Embedding(7, hid_dim)
        if hasSemb:
            self.spatial_embedding = nn.Embedding(num_nodes, hid_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, t_hour, t_day, spatial_indexs=None):
        if self.hasTemb:
            time_in_day_emb = self.time_in_day_embedding(t_hour)
            day_in_week_emb = self.day_in_week_embedding(t_day)
            x = x + time_in_day_emb + day_in_week_emb
        if self.hasSemb:
            if spatial_indexs is None:
                batch, _,  num_nodes, _ = x.shape
                spatial_indexs = torch.LongTensor(
                    torch.arange(num_nodes))  # (N,)
            spatial_emb = self.spatial_embedding(
                spatial_indexs.to(x.device)).unsqueeze(0)  # (N, d)->(1, N, d)
            x = x + spatial_emb.unsqueeze(1)  # (B, T, N, d) + (1, 1, N, d)
        return self.dropout(x)


class DataEncoding(nn.Module):
    def __init__(self, in_dim, hid_dim, hasCross=True, activation='relu'):
        super().__init__()
        assert activation in ['gelu', 'relu']
        in_units = in_dim*2 if hasCross else in_dim
        self.hasCross = hasCross
        self.linear1 = nn.Linear(in_units, hid_dim)
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.linear2 = nn.Linear(hid_dim, hid_dim)

    def forward(self, x, latestX):
        if self.hasCross:
            data = torch.cat([x, latestX], dim=-1)
        else:
            data = x
        data = self.linear1(data)
        data = self.activation(data)
        data = self.linear2(data)
        return data


class CLiST(nn.Module):
    def __init__(self, input_length, predict_length, num_nodes, in_dim,
                 hid_dim=64, M=10, tau=3, n_heads=4, pre_dim=None,
                 addLatestX=True, hasCross=True, hasTemb=True, hasSemb=True,
                 slice_size_per_day=288, num_layers=1, d_out=512,
                 st_emb_dropout=0.1, spatial_dropout=0.1, spatial_att_dropout=0.1,
                 att_type='proxy', activation_data='relu', activation_enc='gelu', activation_dec='gelu',
                 return_att=False):
        super().__init__()
        self.input_length = input_length
        self.predict_length = predict_length
        self.in_dim = in_dim
        self.pre_dim = in_dim if pre_dim is None else pre_dim
        self.num_nodes = num_nodes
        self.tau = tau
        self.hid_dim = hid_dim
        self.addLatestX = addLatestX
        self.hasCross = hasCross
        self.hasTemb = hasTemb
        self.hasSemb = hasSemb
        self.useTCN = tau > 0
        self.return_att = return_att
        self.num_layers = num_layers

        assert att_type in ['proxy', 'full']

        self.data_encoding = DataEncoding(
            in_dim, hid_dim, hasCross, activation=activation_data)
        self.add_st_emb = ST_Embedding(
            hid_dim, num_nodes, slice_size_per_day, st_emb_dropout, hasTemb=hasTemb, hasSemb=hasSemb)

        if self.useTCN:
            if tau == 2:
                tcn_pad_l = 0
                tcn_pad_r = 1
                self.padding = nn.ReplicationPad2d(
                    (tcn_pad_l, tcn_pad_r, 0, 0))  # x must be like (B,C,N,T)
            elif tau == 3:
                tcn_pad_l = 1
                tcn_pad_r = 1
                self.padding = nn.ReplicationPad2d(
                    (tcn_pad_l, tcn_pad_r, 0, 0))  # x must be like (B,C,N,T)
            self.time_conv = nn.Conv2d(hid_dim, hid_dim, (1, tau))

        if att_type == 'full':
            self.spatial_agg_list = nn.ModuleList([
                Full_EncoderLayer(hid_dim, n_heads=n_heads, dropout=spatial_dropout, att_type=att_type,
                                  att_dropout=spatial_att_dropout, return_att=return_att, activation=activation_enc) for _ in range(num_layers)])
        elif att_type == 'proxy':
            self.spatial_agg_list = nn.ModuleList([
                EncoderLayer(M, hid_dim, n_heads=n_heads, num_nodes=num_nodes, dropout=spatial_dropout,
                             att_dropout=spatial_att_dropout, return_att=return_att, activation=activation_enc) for _ in range(num_layers)])

        self.time_wise_predictor = time_wise_predictor(
            num_nodes, input_length, predict_length, hid_dim, pre_dim=self.pre_dim, num_of_filters=d_out, activation=activation_dec)

    def forward(self, x):
        x, t_hour, t_day = x  # x: (B,T,N,C) t_hour:(B,T,1) t_day:(B,T,1)
        B, T, N, _ = x.shape
        x = x[..., 0:self.in_dim]

        latestX = x[:, -1:, :, :].repeat([1, self.input_length, 1, 1])

        data = self.data_encoding(x, latestX)

        data = self.add_st_emb(data, t_hour, t_day)

        if self.useTCN:
            data = data.transpose(1, 3)  # (B,T,N,C)->(B,C,N,T)
            if self.tau > 1:
                data = self.padding(data)
            data = self.time_conv(data)
            data = data.transpose(1, 3)  # (B,C,N,T)->(B,T,N,C)
            assert data.shape[1] == self.input_length

        skip = data
        A_list = []
        for i in range(self.num_layers):
            data, A = self.spatial_agg_list[i](data)
            A_list.append(A)
        data += skip

        main_output = self.time_wise_predictor(data)
        if self.addLatestX:
            if self.input_length == self.predict_length:
                main_output += latestX
            else:
                main_output += latestX[:, 0:self.predict_length, :, :]

        if self.return_att:
            return main_output, A
        return main_output, None


if __name__ == "__main__":
    x = torch.randn((2, 12, 307, 1))
    t_hour = torch.LongTensor(torch.randint(0, 287, (2, 12, 1)))
    t_day = torch.LongTensor(torch.randint(0, 6, (2, 12, 1)))
    layer = CLiST(12, 1, 307, 1, att_type='proxy')
    y, A = layer([x, t_hour, t_day])
    print(layer)
    print(y.shape)
    print(type(A))
